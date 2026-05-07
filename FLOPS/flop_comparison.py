"""
QUICK COMPARISON: Network-only vs Full Rendering FLOPs

Questo file mostra il confronto rapido senza training loop.
Esegui con: python FLOPS/flop_comparison.py
"""

from flop_counter_with_rendering import FlopProfilerWithRendering, RenderingFlopsComparison
import torch


def quick_analysis():
    """
    Analisi rapida: calcola FLOPs per config di default
    """
    
    print("\n" + "="*80)
    print("FLOP ANALYSIS: Network.py vs Full Rendering (renderer.py)")
    print("="*80)
    
    # Config default da mocapDesk2_nerf.txt
    batch_size_evs = 4096
    num_steps = 512
    upsample_steps = 0  # Nel file è 0!
    
    print(f"\nConfiguration (mocapDesk2_nerf.txt):")
    print(f"  batch_size_evs: {batch_size_evs}")
    print(f"  num_steps: {num_steps}")
    print(f"  upsample_steps: {upsample_steps}")
    print(f"  out_dim_color: 1 (events only)")
    print(f"  accumulate_evs: 0")
    print(f"  cuda_ray: False")
    
    num_rays = batch_size_evs * 2  # 2 raggi per coppia eventi
    
    # ========================================================================
    # SCENARIO 1: NETWORK ONLY (quello che abbiamo calcolato prima)
    # ========================================================================
    print("\n" + "-"*80)
    print("SCENARIO 1: NETWORK ONLY (network.py forward pass)")
    print("-"*80)
    
    # Senza ray marching, solo network inference
    network_samples = num_rays * num_steps
    flops_per_sample_network = 22432  # encoder + sigma + color
    flops_network_only = network_samples * flops_per_sample_network
    
    print(f"Calculation:")
    print(f"  - Ray pairs: {batch_size_evs} events × 2 = {num_rays} rays")
    print(f"  - Samples per ray: {num_steps}")
    print(f"  - Total samples: {num_rays} × {num_steps} = {network_samples:,}")
    print(f"  - FLOPs per sample: ~{flops_per_sample_network:,}")
    print(f"  - Total FLOPs (forward): {flops_network_only:,.0f}")
    print(f"  - Total FLOPs (with 3x backward): {flops_network_only*3:,.0f}")
    
    # Human readable
    if flops_network_only >= 1e12:
        print(f"\n  → {flops_network_only/1e12:.2f} Trillion FLOPs")
        print(f"  → {flops_network_only*3/1e12:.2f} Trillion FLOPs (training)")
    
    # ========================================================================
    # SCENARIO 2: FULL RENDERING - Python ray marching
    # ========================================================================
    print("\n" + "-"*80)
    print("SCENARIO 2: FULL RENDERING (renderer.py.run + network.py)")
    print("-"*80)
    
    profiler = FlopProfilerWithRendering(None)
    
    # Ray marching FLOPs (approssimativi da renderer.py.run)
    # Operazioni aggiuntive oltre network:
    # - z_vals generation
    # - xyz generation
    # - PDF sampling
    # - Weight calculation (exp + cumprod)
    # - Compositing
    
    linspace_flops = num_rays * num_steps * 2
    xyz_gen_flops = num_rays * (num_steps + upsample_steps) * 3
    weight_flops = num_rays * (num_steps + upsample_steps) * 5
    composite_flops = num_rays * (num_steps + upsample_steps) * 3
    
    flops_raymarching_overhead = linspace_flops + xyz_gen_flops + weight_flops + composite_flops
    
    flops_full_rendering = flops_network_only + flops_raymarching_overhead
    
    print(f"Breakdown:")
    print(f"  Network FLOPs: {flops_network_only:,.0f}")
    print(f"    - z_vals generation: {linspace_flops:,.0f}")
    print(f"    - xyz calculation: {xyz_gen_flops:,.0f}")
    print(f"    - Weight/alpha calc: {weight_flops:,.0f}")
    print(f"    - Compositing: {composite_flops:,.0f}")
    print(f"  Ray marching overhead: {flops_raymarching_overhead:,.0f}")
    print(f"  ---")
    print(f"  TOTAL (forward): {flops_full_rendering:,.0f}")
    print(f"  TOTAL (with 3x backward): {flops_full_rendering*3:,.0f}")
    
    if flops_full_rendering >= 1e12:
        print(f"\n  → {flops_full_rendering/1e12:.2f} Trillion FLOPs")
        print(f"  → {flops_full_rendering*3/1e12:.2f} Trillion FLOPs (training)")
    
    # ========================================================================
    # SCENARIO 3: CUDA Ray Marching
    # ========================================================================
    print("\n" + "-"*80)
    print("SCENARIO 3: CUDA RAY MARCHING (renderer.py.run_cuda - OPTIMIZED)")
    print("-"*80)
    
    # CUDA ha:
    # - Density grid per early termination (~70% efficienza)
    # - Molto meno overhead nel ray marching
    # - Kernel fusi
    
    network_calls_reduced = int(num_rays * num_steps * 0.7)
    flops_network_cuda = network_calls_reduced * flops_per_sample_network
    
    cuda_overhead = num_rays * num_steps * 2  # Molto ridotto
    
    flops_cuda_rendering = flops_network_cuda + cuda_overhead
    
    print(f"Optimization:")
    print(f"  - Density grid culling: ~70% efficiency")
    print(f"  - Network calls (reduced): {network_calls_reduced:,}")
    print(f"  - Network FLOPs: {flops_network_cuda:,.0f}")
    print(f"  - CUDA overhead: {cuda_overhead:,.0f}")
    print(f"  TOTAL (forward): {flops_cuda_rendering:,.0f}")
    print(f"  TOTAL (with 3x backward): {flops_cuda_rendering*3:,.0f}")
    
    if flops_cuda_rendering >= 1e12:
        print(f"\n  → {flops_cuda_rendering/1e12:.2f} Trillion FLOPs")
        print(f"  → {flops_cuda_rendering*3/1e12:.2f} Trillion FLOPs (training)")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON & ANALYSIS")
    print("="*80)
    
    overhead_python = (flops_full_rendering - flops_network_only) / flops_network_only * 100
    overhead_cuda = (flops_cuda_rendering - flops_network_only) / flops_network_only * 100
    speedup_cuda = flops_full_rendering / flops_cuda_rendering
    
    print(f"\nOverhead vs Network-only:")
    print(f"  Python ray marching: +{overhead_python:.1f}% ({flops_raymarching_overhead:,.0f} FLOPs)")
    print(f"  CUDA ray marching: +{overhead_cuda:.1f}%")
    print(f"\nCUDA Optimization:")
    print(f"  Speedup vs Python ray marching: {speedup_cuda:.2f}x")
    print(f"  FLOPs reduction: {(1-flops_cuda_rendering/flops_full_rendering)*100:.1f}%")
    
    # ========================================================================
    # IMPACT ON TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING IMPACT (100,000 iterations)")
    print("="*80)
    
    num_iters = 100000
    batches_per_epoch = 50  # Approssimato
    epochs = num_iters / batches_per_epoch
    
    flops_train_network_only = flops_network_only * 3 * num_iters
    flops_train_full = flops_full_rendering * 3 * num_iters
    flops_train_cuda = flops_cuda_rendering * 3 * num_iters
    
    print(f"\nTotal FLOPs for full training:")
    print(f"  Network-only: {flops_train_network_only/1e15:.2f} Exaflops")
    print(f"  Full rendering (Python): {flops_train_full/1e15:.2f} Exaflops")
    print(f"  Full rendering (CUDA): {flops_train_cuda/1e15:.2f} Exaflops")
    
    # ========================================================================
    # RECOMMENDATION
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    print(f"""
Current Config Analysis:
  - cuda_ray = False (using Python ray marching)
  - upsample_steps = 0 (no hierarchical sampling)
  - This is simpler but adds ~{overhead_python:.0f}% overhead vs network-only

To Enable CUDA:
  python main_nerf.py --config ... --cuda_ray

Benefits of CUDA:
  - {speedup_cuda:.2f}x faster ray marching
  - ~{(1-flops_cuda_rendering/flops_full_rendering)*100:.0f}% fewer FLOPs
  - Better GPU utilization
  - Enables higher batch sizes

FLOPs are LINEAR with:
  ✓ batch_size_evs: more events = more rays = more FLOPs
  ✓ num_steps: more samples per ray = more FLOPs
  ✓ upsample_steps: adds fine-grained sampling
    """)


if __name__ == '__main__':
    quick_analysis()
