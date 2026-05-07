"""
Minimal Integration Example: Add FLOP profiling to main_nerf.py

This file shows the EXACT LINES to add/modify in main_nerf.py to enable FLOP profiling.

STEP 1: Add imports at the top of main_nerf.py (after existing imports)
STEP 2: Add the --profile-flops argument
STEP 3: Add profiling call in training/testing blocks
STEP 4: Run with: python main_nerf.py --config ... --profile-flops
"""

# ============================================================================
# STEP 1: ADD IMPORTS (put at top of main_nerf.py after existing imports)
# ============================================================================

# Add these lines after "from loss import huber_loss":
from flop_counter import FlopProfiler, profile_training_step, profile_inference_step


# ============================================================================
# STEP 2: ADD ARGUMENT PARSER OPTION
# ============================================================================

# In the argument parser section, add these lines:
parser.add_argument('--profile_flops', action='store_true', 
                   help="enable FLOP profiling with real data")
parser.add_argument('--profile_batches', type=int, default=5,
                   help="number of batches to profile for FLOP counting")


# ============================================================================
# STEP 3: MODIFY TRAINING SECTION
# ============================================================================

# ORIGINAL CODE (in else block, non-test mode):
# else:
#     print(f"opt.lr = {opt.lr}")
#     optimizer = lambda model: torch.optim.Adam(...)
#     ... trainer setup ...

# MODIFIED CODE (add profiling):
else:
    print(f"opt.lr = {opt.lr}")
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    trainer = Trainer(opt.expname, opt, model, device=device, optimizer=optimizer, criterion=criterion, 
                      ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, 
                      scheduler_update_every_step=True, metrics=[PSNRMeter(opt, select_frames)], 
                      use_checkpoint=opt.ckpt)

    # NEW CODE: FLOP PROFILING
    if opt.profile_flops:
        print("\n" + "="*70)
        print("PROFILING FLOPS WITH REAL EVENT DATA")
        print("="*70)
        
        # Prepare loaders for profiling (before main training)
        if opt.events:
            profile_loader = EventNeRFDataset(opt, device=device, type='train', 
                                             downscale=opt.downscale, 
                                             select_frames=select_frames).dataloader()
        else:
            profile_loader = NeRFDataset(opt, device=device, type='train',
                                        downscale=opt.downscale,
                                        select_frames=select_frames).dataloader()
        
        # Run profiling
        train_summary = profile_training_step(
            model, opt, profile_loader, device,
            num_profiles=opt.profile_batches
        )
        
        # Calculate extrapolated FLOPs
        len_loader = len(profile_loader)
        flops_per_epoch = train_summary['avg_flops_per_batch'] * len_loader
        flops_total = train_summary['avg_flops_per_batch'] * opt.iters
        
        print("\n" + "="*70)
        print("FLOP ESTIMATES FOR YOUR CONFIGURATION")
        print("="*70)
        print(f"Config: {opt.expname}")
        print(f"Events enabled: {opt.events}")
        print(f"Batch size: {opt.batch_size_evs}")
        print(f"Num steps: {opt.num_steps}")
        print(f"")
        print(f"FLOPs per batch: {flop_count_str(train_summary['avg_flops_per_batch'])}")
        print(f"Batches per epoch: {len_loader}")
        print(f"FLOPs per epoch: {flop_count_str(flops_per_epoch)}")
        print(f"Total iterations: {opt.iters}")
        print(f"Total FLOPs (training): {flop_count_str(flops_total)}")
        print("="*70 + "\n")


# ============================================================================
# STEP 4: MODIFY TEST SECTION
# ============================================================================

# ORIGINAL CODE (in if opt.test block):
# if opt.test:
#     trainer = Trainer(...)
#     ...
#     else:
#         test_loader = NeRFDataset(...)
#         trainer.test(test_loader)

# MODIFIED CODE (add profiling):
if opt.test:
    trainer = Trainer(opt.expname, opt, model, device=device, 
                     criterion=criterion, fp16=opt.fp16, 
                     metrics=[PSNRMeter(opt, select_frames)], 
                     use_checkpoint=opt.ckpt)

    if opt.gui:
        gui = NeRFGUI(opt, trainer)
        gui.render()
    
    else:
        test_loader = NeRFDataset(opt, device=device, type='test', 
                                 select_frames=select_frames).dataloader()
        
        # NEW CODE: FLOP PROFILING FOR INFERENCE
        if opt.profile_flops:
            print("\n" + "="*70)
            print("PROFILING INFERENCE FLOPS")
            print("="*70)
            
            inference_summary = profile_inference_step(model, opt, test_loader, device)
            
            # Calculate FLOPs per full image
            image_rays = opt.H * opt.W
            batches_per_image = (image_rays + opt.max_ray_batch - 1) // opt.max_ray_batch
            
            print(f"\nImage: {opt.W}x{opt.H} = {image_rays:,} rays")
            print(f"FLOPs per image: {flop_count_str(inference_summary['total_flops'] * batches_per_image)}")
            print("="*70 + "\n")
        
        # Continue with actual testing
        if opt.mode == 'blender':
            trainer.evaluate(test_loader)
        else:
            trainer.test(test_loader)
        trainer.save_mesh(resolution=256, threshold=10)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Profile training with events (mocapDesk2 config)
python main_nerf.py \
    --config configs/mocapDesk2/mocapDesk2_nerf.txt \
    --profile_flops \
    --profile_batches 10

Output:
  - FLOPs per batch from real event data
  - Extrapolated FLOPs per epoch
  - Total FLOPs for all training iterations
  - Per-module breakdown


EXAMPLE 2: Profile inference
python main_nerf.py \
    --config configs/mocapDesk2/mocapDesk2_nerf.txt \
    --test \
    --profile_flops \
    --profile_batches 5

Output:
  - FLOPs per inference batch
  - FLOPs per full image (640x480)
  - Memory efficiency metrics


EXAMPLE 3: Compare different batch sizes
python main_nerf.py \
    --config configs/mocapDesk2/mocapDesk2_nerf.txt \
    --profile_flops \
    --batch_size_evs 2048  # Try different values
    --batch_size_evs 4096
    --batch_size_evs 8192

Note: Run separately for each batch size


EXAMPLE 4: Profile with event accumulation
python main_nerf.py \
    --config configs/mocapDesk2/mocapDesk2_nerf.txt \
    --profile_flops \
    --accumulate_evs 1 \
    --acc_max_num_evs 10

Output: FLOPs with accumulated event pairs (2 rays per event)
"""

# ============================================================================
# WHAT GETS MEASURED
# ============================================================================

"""
TRAINING MEASUREMENT:
  - Reads real event data from EventNeRFDataset
  - Counts FLOPs for network.py forward pass
  - Multiplies by 3 for backward pass (grad computation)
  - Formula: FLOPs = (per_batch_flops * 3) * num_batches_per_epoch * num_epochs

INFERENCE MEASUREMENT:
  - Reads real event data from test set
  - Counts FLOPs for forward pass ONLY
  - No backward pass
  - Formula: FLOPs = per_batch_flops * num_batches_per_image

WHAT IS COUNTED:
  ✓ Encoder (hashgrid): converts 3D position to embedding
  ✓ Sigma network: 2-3 layers of Linear + ReLU
  ✓ Color network: 3 layers of Linear + ReLU
  ✗ Data loading/preprocessing (separate from network)
  ✗ Ray marching (handled by renderer.py, not network.py)
  ✗ Rendering equation (small compared to MLP inference)

WHY USE REAL DATA:
  - Actual input dimensions from real events
  - Realistic batch sizes from data loader
  - No guessing about input shapes
  - Directly correlates with actual training/inference

KEY PARAMETERS THAT AFFECT FLOPs:
  - batch_size_evs: number of event samples (linear impact)
  - num_steps: samples per ray (quadratic impact)
  - accumulate_evs: doubles samples (2 rays per event pair)
  - out_dim_color: 1 (grayscale) vs 3 (RGB) - affects color network output
  - disable_view_direction: saves FLOPs if disabled
"""

# ============================================================================
# EXPECTED OUTPUT EXAMPLE
# ============================================================================

"""
Running: python main_nerf.py --config configs/mocapDesk2/mocapDesk2_nerf.txt \\
         --profile_flops --profile_batches 5

Output:

======================================================================
PROFILING TRAINING STEPS WITH REAL DATA
======================================================================

[EVENT FLOP COUNT] Event samples: 4096
[EVENT FLOP COUNT] Samples per ray: 512
[EVENT FLOP COUNT] Total samples: 4194304

Batch 1/5
  Event batch FLOPs: 1.23T

... (4 more batches) ...

======================================================================
TRAINING PROFILE SUMMARY
======================================================================
Batches profiled: 5
Total FLOPs measured: 6.15T
Avg FLOPs per batch: 1.23T

======================================================================
EXTRAPOLATED FLOP ESTIMATES
======================================================================
Training batches per epoch: 38
Avg FLOPs per batch: 1.23T
Estimated FLOPs per epoch: 46.74T
Total iterations: 100000
Estimated total FLOPs (all iterations): 4.674P  (quadrillion FLOPs!)

======================================================================
FLOP ESTIMATES FOR YOUR CONFIGURATION
======================================================================
Config: mocapDesk2_nerf
Events enabled: True
Batch size: 4096
Num steps: 512

FLOPs per batch: 1.23T (with 3x backward multiplier)
Batches per epoch: 38
FLOPs per epoch: 46.74T
Total iterations: 100000
Total FLOPs (training): 4.674P
======================================================================
"""

