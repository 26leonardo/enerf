"""
TECHNICAL REFERENCE: FLOP Counting for NeRF Events

This document explains the mathematical details of FLOP counting for:
- Event-based NeRF (E-NeRF) using vanilla network.py
- Training vs Inference scenarios
- Event accumulation configurations
"""

# ============================================================================
# 1. NETWORK ARCHITECTURE & DATA FLOW
# ============================================================================

"""
NeRFNetwork.forward(x, d) takes:
  x: [N, 3]  - 3D coordinates (from event rays)
  d: [N, 3]  - view directions (normalized)

Returns:
  sigma: [N, 1]     - density (alpha)
  color: [N, C]     - RGB or grayscale (C=1 or 3)

INTERNAL STRUCTURE:
  1. Position Encoding:
     - encoder = hashgrid
     - x_encoded = encoder(x) -> [N, D_enc]
     - D_enc ~ 32-64 dimensions
     
  2. Sigma Network (density):
     - 2 layers of Linear(D_enc -> hidden_dim -> 1+geo_feat_dim)
     - Each layer: matrix multiply + ReLU
     - Output: sigma + geometric features (15 dims)
     
  3. Direction Encoding:
     - encoder_dir = sphere_harmonics
     - d_encoded = encoder_dir(d) -> [N, D_dir]
     - D_dir ~ 16-24 dimensions
     
  4. Color Network:
     - 3 layers of Linear(D_dir+geo_feat_dim -> hidden_dim -> out_dim_color)
     - Output: RGB [N, 3] or grayscale [N, 1]

TOTAL LAYERS:
  Encoding layers: ~2 (hashgrid encoder, SH encoder)
  Linear layers: 2 (sigma) + 3 (color) = 5 total
"""

# ============================================================================
# 2. FLOP CALCULATION: TRAINING SCENARIO
# ============================================================================

"""
STEP 1: Understand batch composition
========================================

Config: mocapDesk2_nerf.txt (from your workspace)
  - batch_size_evs = 4096         # Event samples per batch
  - num_steps = 512               # Ray marching steps
  - accumulate_evs = 0            # No accumulation (just single events)
  - out_dim_color = 1             # Grayscale output

STEP 2: Calculate total samples per forward pass
==================================================

For a single event sample:
  - We march along ray with 512 steps
  - At each step, evaluate network: forward(x_t, d)
  - Total evaluations per event = 512 samples
  
For one batch:
  - 4096 events × 512 steps = 2,097,152 samples per forward pass
  
Full notation:
  N_events = 4096                    # batch_size_evs
  N_steps = 512                      # num_steps per ray
  N_samples_total = N_events × N_steps = 2,097,152


STEP 3: Count FLOPs for a single sample (N=1)
===============================================

Hashgrid Encoder:
  Input: (1, 3) - single 3D position
  Output: (1, D_enc) where D_enc ≈ 32-64
  FLOPs_encoder ≈ 1,000-5,000 ops (depends on hash resolution)
  
Sigma Network: 2 layers
  Layer 1: Linear(D_enc -> hidden_dim=64)
    Input: (1, D_enc=32)
    Output: (1, 64)
    FLOPs = 2 × 32 × 64 = 4,096
    (2x because forward + 1 multiply-add per element)
    
  ReLU: negligible
  
  Layer 2: Linear(64 -> 1+geo_feat_dim=16)
    FLOPs = 2 × 64 × 16 = 2,048
    
  Total sigma network: ~6,144 FLOPs
  
Direction Encoder:
  FLOPs ≈ 500-2,000 ops
  
Color Network: 3 layers
  Layer 1: Linear(dir_dim+geo_feat_dim=16+15=31 -> 64)
    FLOPs = 2 × 31 × 64 = 3,968
    
  Layer 2: Linear(64 -> 64)
    FLOPs = 2 × 64 × 64 = 8,192
    
  Layer 3: Linear(64 -> out_dim_color=1)
    FLOPs = 2 × 64 × 1 = 128
    
  Total color network: ~12,288 FLOPs
  
TOTAL PER SAMPLE:
  FLOPs_forward = encoder + sigma_net + color_net
  FLOPs_forward ≈ 2,000 + 6,144 + 2,000 + 12,288 ≈ 22,432 FLOPs


STEP 4: Calculate batch FLOPs
==============================

Forward pass (inference):
  FLOPs_batch_forward = FLOPs_per_sample × N_samples_total
  FLOPs_batch_forward = 22,432 × 2,097,152
  FLOPs_batch_forward ≈ 47 Trillion FLOPs (47T)

Backward pass (gradient computation):
  - Backward typically costs 2-3x forward
  - FLOPs_batch_backward ≈ 2-3 × FLOPs_batch_forward
  
Total per training batch:
  FLOPs_batch_total ≈ 47T × 3 = 141T FLOPs


STEP 5: Extrapolate to epoch/training
======================================

Batches per epoch:
  Total training events ≈ 10,000-100,000
  Batches per epoch = Total_events / batch_size_evs
                    = ~50,000 / 4,096 ≈ 12 batches
                    
FLOPs per epoch:
  FLOPs_epoch = FLOPs_batch × batches_per_epoch
              = 141T × 12
              = 1.7 Quadrillion (1.7P) FLOPs per epoch
              
Total training FLOPs:
  Total iterations: opt.iters = 100,000
  Batches for all iterations = 100,000 / 12 ≈ 8,333 epochs
  Total FLOPs = 141T × 100,000 ≈ 14.1 Quadrillion FLOPs


FORMULA SUMMARY:
================

  FLOPs_batch_forward = (D_enc + D_sigma + D_dir + D_color) × N_events × N_steps
  
  Where:
    D_enc = encoding dimensions (~2,000-5,000 ops)
    D_sigma = sigma network linear ops (~6,000)
    D_dir = direction encoding (~2,000)
    D_color = color network linear ops (~12,000)
    N_events = batch_size_evs
    N_steps = num_steps per ray
  
  FLOPs_batch_total = FLOPs_batch_forward × 3  (for training)
  FLOPs_epoch = FLOPs_batch_total × batches_per_epoch
  FLOPs_total = FLOPs_epoch × num_epochs
"""

# ============================================================================
# 3. FLOP CALCULATION: INFERENCE SCENARIO
# ============================================================================

"""
INFERENCE differs from training:
  ✓ No backward pass (no gradient computation)
  ✓ No batch norm / dropout variance
  ✓ Simpler computation: just forward
  ✓ FLOPs ≈ 1/3 of training

STEP 1: Image composition
===========================

Config:
  - H = 480, W = 640 resolution
  - Total rays per image = 480 × 640 = 307,200 rays
  - max_ray_batch = 5,096 (OOM avoidance)

STEP 2: Batches needed per image
==================================

Image rays / max_ray_batch = 307,200 / 5,096 ≈ 60 batches
(ceiling division: (307,200 + 5,096 - 1) / 5,096 = 61 batches)

STEP 3: FLOPs per inference batch
==================================

Using same network as training:
  N_samples = max_ray_batch × num_steps = 5,096 × 512 = 2,609,152
  FLOPs_batch = 22,432 × 2,609,152 ≈ 58.5T FLOPs (forward only)

STEP 4: FLOPs per image
=======================

FLOPs_image = FLOPs_per_batch × batches_per_image
            = 58.5T × 61
            = 3.57 Quadrillion FLOPs per 480x640 image

For full video (100 images):
  FLOPs_video = 3.57P × 100 = 357 Quadrillion FLOPs
  
Rendering time estimate (A100 = 312 TFLOPS):
  Time = 357P / 312T = ~1,144 seconds ≈ 19 minutes for 100 frames


FORMULA:
=========

  FLOPs_image = FLOPs_per_sample × N_rays × N_steps × ceil(N_rays / max_batch)
              = 22,432 × 307,200 × 512 × 60
              ≈ 3.57P FLOPs per image
"""

# ============================================================================
# 4. EFFECT OF EVENT ACCUMULATION
# ============================================================================

"""
When accumulate_evs = 1:
  - Instead of single event: (x, y, t, p)
  - Process event pairs: (x, y, t1, p1) -> (x, y, t2, p2)
  - Two rays per event pair: rays_evs_o1/d1 and rays_evs_o2/d2
  - Temporal constraint: events at same pixel

FLOP IMPACT:
  With accumulation:
    N_rays_per_event = 2  (instead of 1)
    N_samples_total = N_events × 2 × N_steps
    FLOPs_batch ≈ 2 × FLOPs_without_accum
    
  Example:
    Without accumulation: 4096 events × 512 steps = 2M samples
    With accumulation: 4096 events × 2 rays × 512 steps = 4M samples
    FLOPs increase: 2x


acc_max_num_evs parameter:
  - Limits temporal window for accumulation
  - acc_max_num_evs = 10: only use up to 10 successor events
  - Smaller value = fewer rays per event pair = fewer FLOPs
  - Formula: N_rays_per_event = min(num_successors, acc_max_num_evs)


EXPECTED FLOP VARIATIONS:
===========================

Config 1: No accumulation
  FLOPs/batch = 141T

Config 2: With accumulation (unlimited)
  FLOPs/batch = 282T (2x)

Config 3: With accumulation (max 5)
  FLOPs/batch = 211.5T (1.5x)
"""

# ============================================================================
# 5. EFFECT OF OUTPUT DIMENSION
# ============================================================================

"""
out_dim_color parameter:

Case 1: out_dim_color = 1 (Grayscale - your config)
  Color network output: [N, 1]
  Final layer: Linear(64 -> 1)
  FLOPs = 2 × 64 × 1 = 128
  Total network FLOPs ≈ 22,432

Case 2: out_dim_color = 3 (RGB)
  Color network output: [N, 3]
  Final layer: Linear(64 -> 3)
  FLOPs = 2 × 64 × 3 = 384 (3x more!)
  Total network FLOPs ≈ 22,688 (slightly higher)
  
FLOP difference:
  RGB vs Grayscale ≈ 1.01x (minimal, mainly in final layer)
  BUT: loss computation may increase
"""

# ============================================================================
# 6. MEASURING WITH FVCORE
# ============================================================================

"""
How FlopCounterMode works:

from fvcore.nn import FlopCounterMode

flops, flops_dict = FlopCounterMode.flop_count(
    model,
    (x_input, d_input)  # tuple of inputs
)

# flops: total FLOPs as integer
# flops_dict: dict mapping module names to their FLOPs

Breakdown by module:
  - 'encoder': hashgrid encoding FLOPs
  - 'sigma_net.0': first sigma layer
  - 'sigma_net.1': second sigma layer
  - 'encoder_dir': direction encoding
  - 'color_net.0': first color layer
  - 'color_net.1': second color layer
  - 'color_net.2': third color layer


Counting matrix multiplication FLOPs:
  For Linear(in_features -> out_features):
    FLOPs = 2 × batch_size × in_features × out_features
    (2x because: multiply + accumulate = 1 MAC = 2 FLOPs)


Validation:
  - Count FLOPs for small batch (e.g., 1024)
  - Extrapolate linearly to larger batches
  - Should scale perfectly (linear complexity)
"""

# ============================================================================
# 7. PRACTICAL ESTIMATION TABLE
# ============================================================================

"""
Quick reference for your configuration:

================================================================================================
Scenario         | Batch Size | Samples     | FLOPs/Sample | FLOPs/Batch | FLOPs with 3x
================================================================================================
Training         | 4096       | 2,097,152   | ~22.5K       | 47T         | 141T
Training (accum) | 4096       | 4,194,304   | ~22.5K       | 94T         | 282T
Inference        | 5096       | 2,609,152   | ~22.5K       | 58.5T       | —
Inference/image  | varying    | 157,286,400 | ~22.5K       | 3.57P       | —
================================================================================================

Performance estimates (on NVIDIA A100 = 312 TFLOPS peak):

Training 100,000 iterations:
  - With 2M samples/batch: ~14.1P FLOPs total
  - Estimated time: 45 hours (assuming 25% peak utilization)
  
Full video inference (100 frames):
  - 3.57P per frame
  - Total: 357P FLOPs
  - Estimated time: 19 minutes per 100 frames


Memory considerations:

Peak GPU memory for batch:
  - Forward activations: ~2GB (for 2M samples)
  - Backward gradients: ~4GB additional
  - Model weights: ~200MB
  - Total: ~6-7GB (fits in RTX 4060 Ti with 8GB)


To reduce FLOPs:

1. Reduce batch_size_evs (e.g., 2048 instead of 4096)
   Impact: -50% FLOPs, but slower training

2. Reduce num_steps (e.g., 256 instead of 512)
   Impact: -50% FLOPs, but lower quality rendering

3. Disable accumulation (already enabled in your config)
   Impact: -50% FLOPs

4. Reduce hidden_dim in sigma/color networks
   Impact: linear reduction in network FLOPs
"""

# ============================================================================
# 8. VALIDATION: HOW TO VERIFY YOUR COUNTS
# ============================================================================

"""
To validate FLOP counts:

1. Simple test:
   x = torch.randn(1024, 3, device='cuda')
   d = torch.randn(1024, 3, device='cuda')
   
   flops, _ = FlopCounterMode.flop_count(model, (x, d))
   
   Expected output: ~22-25M FLOPs for 1024 samples


2. Linear scaling test:
   for batch_size in [1024, 2048, 4096, 8192]:
       x = torch.randn(batch_size, 3)
       flops, _ = FlopCounterMode.flop_count(model, (x, d))
       print(f"{batch_size}: {flops}")
   
   Expected: FLOPs should scale perfectly linearly


3. Compare theoretical vs measured:
   Theoretical FLOPs (from layer dims):
     = sum(2 × in_dim × out_dim for each linear layer)
   
   Measured FLOPs (from fvcore):
     = actual count
   
   Should match within 10% (difference is encoding overhead)


4. Real training validation:
   - Measure FLOPs from first 5 batches
   - Calculate per-batch average
   - Extrapolate to epoch
   - Compare with wall-clock time vs theoretical bandwidth
"""

# ============================================================================
# 9. SUMMARY: KEY EQUATIONS
# ============================================================================

"""
FINAL FORMULAS TO REMEMBER:

For single network forward pass:
  FLOPs = Σ(2 × in_dim × out_dim) for all Linear layers
          + encoding overhead
        ≈ 22,432 FLOPs per sample

For batch of N samples:
  FLOPs_batch = FLOPs_per_sample × N

For training (forward + backward):
  FLOPs_training = FLOPs_batch × 3

For full training:
  FLOPs_total = FLOPs_per_sample × total_samples_across_all_iterations

Where:
  total_samples = num_events × num_steps × num_batches_in_training
                = 4096 × 512 × (100,000 iterations / batches_per_iteration)
                ≈ 2.1 × 10^15 samples


Memory estimate:
  Peak GPU memory ≈ 2 × batch_size × num_steps × (hidden_dim × sizeof(float32))
                  ≈ 2 × 4096 × 512 × 64 × 4 bytes
                  ≈ 4.3 GB
"""

print(__doc__)
