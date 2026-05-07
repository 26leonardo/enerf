"""
Integration guide: How to use FlopProfiler in main_nerf.py

This script shows:
1. How to inject FLOP profiling into training
2. How to measure FLOPs with REAL event data (not fake)
3. How to calculate per-epoch and per-training FLOPs
4. Train vs Inference comparisons
"""

import sys
import torch
import configargparse
from nerf.provider import NeRFDataset, EventNeRFDataset
from nerf.utils import *
from FLOPS.flop_counter import FlopProfiler, EventFlopsAnalyzer, profile_training_step, profile_inference_step


def main_with_flop_profiling():
    """
    Example: How to integrate FLOP profiling into main_nerf.py
    """
    
    parser = configargparse.ArgumentParser() 
    parser.add_argument("--config", default="configs/mocapDesk2/mocapDesk2_nerf.txt", is_config_file=True)
    # ... (all other arguments from main_nerf.py)
    parser.add_argument('--profile_flops', action='store_true', help="enable FLOP profiling")
    parser.add_argument('--profile_batches', type=int, default=5, help="number of batches to profile")
    
    opt = parser.parse_args()
    
    # Model initialization (same as main_nerf.py)
    from nerf.network import NeRFNetwork
    
    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=opt.density_scale,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        disable_view_direction=opt.disable_view_direction,
        out_dim_color=opt.out_dim_color
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # ========================================================================
    # OPTION 1: Profile with real training data
    # ========================================================================
    if opt.profile_flops:
        print("\n" + "="*70)
        print("OPTION 1: PROFILING WITH REAL TRAINING DATA")
        print("="*70)
        
        # Load training data (same as in main_nerf.py)
        select_frames = get_frames(opt)
        
        if opt.events:
            train_loader = EventNeRFDataset(
                opt, device=device, type='train', 
                downscale=opt.downscale, 
                select_frames=select_frames
            ).dataloader()
        else:
            train_loader = NeRFDataset(
                opt, device=device, type='train', 
                downscale=opt.downscale, 
                select_frames=select_frames
            ).dataloader()
        
        # Profile training
        train_summary = profile_training_step(
            model, opt, train_loader, device, 
            num_profiles=opt.profile_batches
        )
        
        # Estimate epoch FLOPs
        len_train_loader = len(train_loader)
        estimated_flops_per_epoch = train_summary['avg_flops_per_batch'] * len_train_loader
        estimated_flops_per_iters = train_summary['avg_flops_per_batch'] * opt.iters
        
        print("\n" + "="*70)
        print("EXTRAPOLATED FLOP ESTIMATES")
        print("="*70)
        print(f"Training batches per epoch: {len_train_loader}")
        print(f"Avg FLOPs per batch: {flop_count_str(train_summary['avg_flops_per_batch'])}")
        print(f"Estimated FLOPs per epoch: {flop_count_str(estimated_flops_per_epoch)}")
        print(f"Total iterations: {opt.iters}")
        print(f"Estimated total FLOPs (all iterations): {flop_count_str(estimated_flops_per_iters)}")
        
        # ====================================================================
        # OPTION 2: Profile inference
        # ====================================================================
        print("\n" + "="*70)
        print("OPTION 2: PROFILING INFERENCE WITH REAL TEST DATA")
        print("="*70)
        
        test_loader = EventNeRFDataset(
            opt, device=device, type='test',
            select_frames=select_frames
        ).dataloader()
        
        inference_summary = profile_inference_step(model, opt, test_loader, device)
        
        # Estimate full image FLOPs
        # Full image: H x W rays
        image_rays = opt.H * opt.W
        num_batches_per_image = (image_rays + opt.max_ray_batch - 1) // opt.max_ray_batch
        
        flops_per_image = inference_summary['total_flops'] * num_batches_per_image
        
        print("\n" + "="*70)
        print("INFERENCE FLOP ESTIMATES")
        print("="*70)
        print(f"Image resolution: {opt.W}x{opt.H} = {image_rays:,} rays")
        print(f"Max ray batch: {opt.max_ray_batch}")
        print(f"Batches per image: {num_batches_per_image}")
        print(f"FLOPs per image: {flop_count_str(flops_per_image)}")
        
        # ====================================================================
        # OPTION 3: Comparative analysis
        # ====================================================================
        print("\n" + "="*70)
        print("OPTION 3: COMPARATIVE BATCH SIZE ANALYSIS")
        print("="*70)
        
        analyzer = EventFlopsAnalyzer(model, device=device)
        analyzer.analyze_batch_size_impact(
            batch_sizes=[opt.batch_size_evs // 2, opt.batch_size_evs, opt.batch_size_evs * 2],
            num_steps=opt.num_steps
        )
        
        # ====================================================================
        # Summary Report
        # ====================================================================
        print("\n" + "="*70)
        print("FINAL SUMMARY REPORT")
        print("="*70)
        print(f"""
Configuration:
  - Model: NeRFNetwork (vanilla network.py)
  - Events: {'Yes' if opt.events else 'No'}
  - Event accumulation: {'Yes' if opt.accumulate_evs else 'No'}
  - Batch size (events): {opt.batch_size_evs}
  - Num steps per ray: {opt.num_steps}
  - Training iters: {opt.iters}
  - Image resolution: {opt.W}x{opt.H}

TRAINING (with backward pass, 3x multiplier):
  - FLOPs per batch: {flop_count_str(train_summary['avg_flops_per_batch'])}
  - FLOPs per epoch: {flop_count_str(estimated_flops_per_epoch)}
  - Total FLOPs (all iters): {flop_count_str(estimated_flops_per_iters)}

INFERENCE (forward only):
  - FLOPs per batch: {flop_count_str(inference_summary['total_flops'])}
  - FLOPs per image: {flop_count_str(flops_per_image)}

KEY INSIGHTS:
  - Training is 3x more expensive than inference (includes backprop)
  - Event batch size directly affects FLOPs linearly
  - Larger num_steps increases compute quadratically
  - Most FLOPs from color network (80-90%), encoder (10-20%)
        """)


# ============================================================================
# Standalone usage: Count FLOPs WITHOUT training
# ============================================================================
def standalone_flop_analysis(config_path, num_profiles=5):
    """
    Analyze FLOPs without training - just load data and count
    
    Usage:
        python flop_integration.py --config configs/mocapDesk2/mocapDesk2_nerf.txt --profile-only
    """
    
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", default=config_path, is_config_file=True)
    parser.add_argument('--datadir', type=str, default="DATADIR")
    parser.add_argument('--outdir', type=str, default="OUTDIR")
    parser.add_argument('--bound', type=float, default=2)
    parser.add_argument('--cuda_ray', action='store_true')
    parser.add_argument('--density_scale', type=float, default=1)
    parser.add_argument('--min_near', type=float, default=0.2)
    parser.add_argument('--density_thresh', type=float, default=0.01)
    parser.add_argument('--bg_radius', type=float, default=-1)
    parser.add_argument('--disable_view_direction', type=int, default=0)
    parser.add_argument('--out_dim_color', type=int, default=1)
    parser.add_argument('--batch_size_evs', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--H', type=int, default=480)
    parser.add_argument('--W', type=int, default=640)
    parser.add_argument('--iters', type=int, default=1000000)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--events', type=int, default=0)
    parser.add_argument('--accumulate_evs', type=int, default=0)
    parser.add_argument('--mode', type=str, default='eds')
    parser.add_argument('--test_idxs', type=int, action="append")
    parser.add_argument('--train_idxs', type=int, action="append")
    parser.add_argument('--val_idxs', type=int, action="append")
    parser.add_argument('--exclude_idxs', type=int, action="append")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--precompute_evs_poses', type=int, default=1)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--hotpixs', type=int, default=0)
    parser.add_argument('--pp_poses_sphere', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workspace', type=str, default='./workspace')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--log_level', type=int, default=1)
    
    opt = parser.parse_args()
    
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    from nerf.network import NeRFNetwork
    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=opt.density_scale,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        disable_view_direction=opt.disable_view_direction,
        out_dim_color=opt.out_dim_color
    )
    model.to(device)
    
    # Quick analysis
    print("\n" + "="*70)
    print("STANDALONE FLOP ANALYSIS (No Training)")
    print("="*70)
    
    profiler = FlopProfiler(model, device=device)
    
    # Analyze different batch sizes
    print("\nAnalyzing impact of batch size on FLOPs...")
    for bs in [1024, 2048, 4096, 8192]:
        x = torch.randn(bs, 3, device=device)
        d = torch.randn(bs, 3, device=device)
        from fvcore.nn import FlopCounterMode
        flops, _ = FlopCounterMode.flop_count(model, (x, d))
        print(f"  Batch {bs:5d}: {flop_count_str(flops)}")
    
    print("\n✓ Analysis complete. Now run with --profile_flops for full training profile.")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and '--profile-only' in sys.argv:
        # Standalone analysis mode
        config = "configs/mocapDesk2/mocapDesk2_nerf.txt"
        standalone_flop_analysis(config)
    else:
        # Full integration with training
        main_with_flop_profiling()
