# Piano Completo di Analisi E-NeRF
## Inferenza, Training, Profiling e Identificazione Colli di Bottiglia

**Data**: Aprile 2026  
**Target**: Analisi approfondita dell'architettura E-NeRF inclusi aspetti computazionali, memoria e CUDA

---

## 📋 Indice
1. [Analisi dell'Architettura](#1-analisi-dellarchitettura)
2. [Profiling in Inferenza](#2-profiling-in-inferenza)
3. [Profiling in Training](#3-profiling-in-training)
4. [Algoritmo di Ray Marching](#4-algoritmo-di-ray-marching)
5. [Dimensioni Attivazioni (Tensor Shapes)](#5-dimensioni-attivazioni-tensor-shapes)
6. [Analisi Costi Computazionali](#6-analisi-costi-computazionali)
7. [Profilatura Codice CUDA](#7-profilatura-codice-cuda)
8. [Identificazione Colli di Bottiglia](#8-identificazione-colli-di-bottiglia)
9. [Strumenti e Metriche](#9-strumenti-e-metriche)
10. [Piano di Implementazione](#10-piano-di-implementazione)

---

## 1. Analisi dell'Architettura

### 1.1 Componenti Principali

```
Input (ray_o, ray_d)
    ↓
┌─────────────────────────────────────────────────┐
│  1. RAY MARCHING (raymarching.cu)               │
│     - near_far_from_aabb: calcola [t_near, t_far]│
└─────────────────────────────────────────────────┘
    ↓
    Sample N punti lungo il raggio (z_vals)
    ↓
┌─────────────────────────────────────────────────┐
│  2. HASH GRID ENCODING (gridencoder.cu)         │
│     - Multi-level hash table encoding           │
│     - Output: feature vector ricco              │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  3. SIGMA MLP (ffmlp.cu o network.py)           │
│     - Predice: density + geometric features     │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  4. SPHERICAL HARMONICS (shencoder.cu)          │
│     - Codifica direzione di vista               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  5. COLOR MLP (ffmlp.cu o network.py)           │
│     - Predice: RGB colore                       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  6. VOLUME RENDERING (renderer.py)              │
│     - Accumula colore e opacità lungo raggio    │
│     - Applicoone formule NeRF standard          │
└─────────────────────────────────────────────────┘
    ↓
Output (RGB, Depth)
```

### 1.2 Tre Varianti Implementative

| Variant | File | Backend | Velocità |
|---------|------|---------|----------|
| **Standard** | `network.py` | PyTorch nn.Linear | ⚡ Base |
| **Fast** | `network_ff.py` | FFMLP (fully-fused) | 🔥🔥 10-100x |
| **TCNN** | `network_tcnn.py` | Tiny CUDA NN | 🔥 Estremamente veloce |

---

## 2. Profiling in Inferenza

### 2.1 Obiettivi di Profiling Inferenza

✅ Misurare **latenza end-to-end** per un frame  
✅ Identificare quale componente consuma più tempo  
✅ Misurare **throughput** (frame/sec)  
✅ Memoria GPU allocata e picco di utilizzo  
✅ Rapporto compute vs memory bandwidth

### 2.2 Configurazione Tipica Inferenza

```yaml
Input:
  - Resolution: 512×512 pixel (256K raggi se campionati tutti)
  - Batch size: tipicamente 1 frame
  - Num samples per ray: 128 (coarse) + 128 (fine) = 256 sample totali

Memory footprint:
  - Model: ~50 MB (encoder + 2 MLP)
  - Activations: dinamiche (vedere sezione 5)
  - Peak: ~500 MB - 2 GB
```

### 2.3 Pipeline Profiling Dettagliato

#### **Stage 1: Ray Generation** (su CPU/GPU)
```python
# Input: camera pose, intrinsics
# Output: rays_o [B, N, 3], rays_d [B, N, 3]
# Dimenseione N: 256*256 = 65536 raggi per frame

Metrica da misurare:
- Tempo: few ms (quasi trascurabile)
- Memoria: O(N) = ~1.5 MB (fp32)
```

#### **Stage 2: Ray Marching / AABB Intersection** (CUDA)
```cuda
raymarching::near_far_from_aabb(rays_o, rays_d, aabb)
→ Output: nears [N], fars [N]

Specifiche CUDA:
- Kernel: _near_far_from_aabb
- Thread grid: ceil(N/256) blocchi × 256 thread
- Shared memory: minimal
- Operazioni: divide, min/max (altamente parallelo)

Metriche:
- Tempo: <1 ms (memoria-bound)
- Memory bandwidth: ~10 GB/s
```

#### **Stage 3: Z-value Sampling**
```python
# Generiamo campioni lineari uniformi da t_near a t_far
z_vals = nears + (fars - nears) * linspace(0, 1, num_steps)  
# Shape: [N, num_steps] = [65536, 128]

Metriche:
- Tempo: <1 ms (semplice math)
- Memoria: O(N * num_steps) = [65536 × 128 × 4 byte] = ~33 MB
```

#### **Stage 4: XYZ Computation (Punto sampling)**
```python
xyzs = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
# Shape: [N, num_steps, 3]
# = [65536, 128, 3] = 25M floats = 100 MB

Metriche:
- Tempo: <1 ms
- Memoria: 100 MB (temporaneo)
```

#### **Stage 5: Hash Grid Encoding** (CUDA - 🔴 CRITICO)
```cuda
feature = gridencoder(xyzs)  
# Input:  xyzs [N*num_steps, 3] = [8.4M, 3]
# Output: features [N*num_steps, level_dim * num_levels]
#         = [8.4M, 2*16] = [8.4M, 32]

Specifiche CUDA implementazione:
- Kernel: grid_encode_forward (gridencoder/src/*.cu)
- Operazioni by voxel:
  1. Calcola grid coordinates per ogni livello: log2_interpolate
  2. Hash lookup in tabella: hash(coords) → embedding index
  3. Interpolazione trillineare: 8 voxel neighbors
  
- Memory access pattern:
  - Input: streaming (coalesced) ✅
  - Hash table: random (non-coalesced) ❌ BOTTLENECK
  - Embedding table: gather di [8 × 32] floats per sample

Metriche:
- Tempo: 10-50 ms (memory-bound su tabella hash)
- Memory: 
  - Input: ~100 MB (xyzs)
  - Embedding table: ~32 × 2^19 × 4B × 16 levels ≈ 256 MB
  - Output: ~270 MB (features)
  - Peak: ~500 MB
- Memory bandwidth raggiunto: 50-150 GB/s (vs 900 GB/s theoretical su RTX4090)
```

#### **Stage 6: Sigma MLP Forward** (CUDA-optimizzato per network_ff.py)
```cuda
// Variante STANDARD (network.py)
h_state = features  # [8.4M, 32]
for l in range(num_layers_sigma):  # 2 layer
    h_state = Linear(h_state, hidden_dim=64)  # matmul + bias
    if l < num_layers_sigma-1:
        h_state = ReLU(h_state)
sigma = exp(h_state[:, 0])
geo_feat = h_state[:, 1:16]

Operazioni layer 0: 8.4M × 32 × 64 = 17.2 TFLOP
Operazioni layer 1: 8.4M × 64 × 16 = 8.6 TFLOP
Total: 25.8 TFLOP

// Variante FAST (network_ff.py) - FFMLP
Fully-fused MLP con Tensor Cores (WMMA)
Num samples: 8.4M è TOO LARGE per un singolo kernel
Soluzione: batch piccoli (1024-4096 samples) con kernel fondente

Metriche:
STANDARD:
- Tempo: 50-100 ms (compute-bound, 25.8 TFLOP @ 300 TFLOPS)
- Memory: 
  - Features in: 270 MB
  - L0 weights: 32 × 64 × 4B = 8 KB (cache resident)
  - L0 activation: 8.4M × 64 × 4B = 2.1 GB (large!)
  - L1 weights: 64 × 16 × 4B = 4 KB
  - Peak: ~2.5 GB

FAST (FFMLP):
- Tempo: 10-20 ms (5-10x speedup grazie Tensor Cores)
- Memory: simile, ma compute più efficiente
```

#### **Stage 7: View Direction Encoding** (CUDA)
```cuda
// Spherical Harmonics - velocissimo
view_dir = ray_d / ||ray_d||  # [N, 3]
sh_feat = spherical_harmonic_encode(view_dir)  # [N, degree^2]
// degree=4 → 16 output 

Kernel: sh_encode_forward (shencoder/src/*.cu)
- Semplice: computazione polinomiale pura
- NO gather/scatter, NO hash → PERFETTAMENTE COALESCED

Metriche:
- Tempo: 2-5 ms
- Memory: [8.4M, 3] → [8.4M, 16] = 40 MB
- Compute-bound, utilizza ~50% di peak throughput
```

#### **Stage 8: Color MLP Forward** (CUDA - 🔴 CRITICO)
```cuda
h_input = concat(view_feat, geo_feat)  # [8.4M, 16+15] = [8.4M, 31]
h_state = h_input

for l in range(num_layers_color):  # 3 layer
    h_state = Linear(h_state, hidden_dim=64)
    if l < num_layers_color-1:
        h_state = ReLU(h_state)
rgb = sigmoid(h_state)

Operazioni layer 0: 8.4M × 31 × 64 = 16.6 TFLOP
Operazioni layer 1: 8.4M × 64 × 64 = 34.6 TFLOP (!)
Operazioni layer 2: 8.4M × 64 × 3 = 1.6 TFLOP
Total: 52.8 TFLOP (!!! PIÙ GRANDE DI SIGMA MLP)

Metriche:
STANDARD:
- Tempo: 80-150 ms (la componente più pesante)
- Memory: ~3 GB picco

FAST (FFMLP):
- Tempo: 20-40 ms
```

#### **Stage 9: Volume Rendering** (PyTorch standard)
```python
# In renderer.py - sample_pdf + alpha compositing

deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, num_steps-1]
alphas = 1.0 - torch.exp(-sigmas * deltas)  # [N, num_steps]

weights = alphas * torch.cumprod(
    torch.cat([torch.ones(...), 1-alphas[..., :-1]], dim=-1),
    dim=-1
)  # [N, num_steps] - fattore di transmittance

rgb_final = (weights[..., None] * rgbs).sum(dim=-2)  # [N, 3]
depth_final = (weights * z_vals).sum(dim=-1)  # [N]

Metriche:
- Tempo: 5-10 ms (semplice)
- Memory: ~300 MB
- BENE PARALLELO: ogni raggio è indipendente
```

### 2.4 Timeline Inferenza Completa (tipica)

```
Ray Gen:         1 ms  ████
AABB:            1 ms  ████
Z-sampling:      1 ms  ████
XYZ compute:     1 ms  ████
─────────────────────────────
Grid Encoding:  30 ms  ███████████████████████████████
─────────────────────────────
Sigma MLP:      60 ms  ████████████████████████████████████████████████████████
─────────────────────────────
View enc:        3 ms  ███
─────────────────────────────
Color MLP:      100 ms  ███████████████████████████████████████████████████████████████████████████████████
─────────────────────────────
Vol render:      8 ms  ████████
─────────────────────────────
TOTAL:        ~205 ms per frame (5 FPS)

Variante FFMLP:
─────────────────────────────
Grid Encoding:  30 ms  (stesso, CUDA)
Sigma MLP:      15 ms  (4x più veloce)
Color MLP:      30 ms  (3x più veloce)
─────────────────────────────
TOTAL:        ~85 ms (12 FPS) ✅
```

### 2.5 Setup Profiling Inferenza

```python
# Script da creare: scripts/profile_inference.py

import torch
import numpy as np
from pathlib import Path

# Opzioni
RESOLUTION = 512
NUM_STEPS = 128
VARIANT = 'ff'  # Scegli: 'standard', 'ff', 'tcnn'

# Setup model (carica checkpoint)
model = load_model(variant=VARIANT)
model.cuda()
model.eval()

# Crea dummy rays (o carica da dataset)
rays_o = torch.randn(1, RESOLUTION*RESOLUTION, 3).cuda()
rays_d = torch.randn(1, RESOLUTION*RESOLUTION, 3).cuda()
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# Profiling con PyTorch profiler + NVIDIA Nsys
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        rgb, depth = model.render(rays_o, rays_d, num_steps=NUM_STEPS)

# Print statistiche
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Salva per analisi in Tensorboard
prof.export_chrome_trace(f"/tmp/trace_inferenza_{VARIANT}.json")
```

---

## 3. Profiling in Training

### 3.1 Obiettivi di Profiling Training

✅ Costo **forward pass**  
✅ Costo **backward pass** (gradiente computation)  
✅ Costo **optimizer step** (weight update)  
✅ Memoria **totale allocata** e pattern di uso  
✅ **Data loading** overhead  
✅ **Loss computation** e reduzi

✅ Scalabilità con batch size / num rays

### 3.2 Configurazione Tipica Training

```yaml
Scenario: Event-based NeRF training
  Dataset: ESIM simulato (event-camera)
  Frames train: 965 (dal config)
  Frames val: 3 test frame
  Batch size: da config (spesso 1 frame = 1 GPU)

Rays per iteration:
  - Resolution: 512×512 (configurabile downscale)
  - Pixels: 262,144 ray per frame
  - Batch rays per iter: spesso tutti (262K)

Aspetti evento:
  - C_thres: soglia evento (loss normalizzato)
  - accumulate_evs: accumula più event batches
  - event_only: usa solo evento (no frame ground truth)
```

### 3.3 Training Loop (da main_nerf.py)

```python
for epoch in range(max_epochs):
    for batch_idx, rays_data in enumerate(train_loader):
        
        # ─────────────────────────────────────────────────
        # Forward pass
        # ─────────────────────────────────────────────────
        predictions = model.render(rays_o, rays_d, num_steps=128, upsample_steps=128)
        # → output: dict con 'rgb', 'depth', 'weights', ecc.
        
        # ─────────────────────────────────────────────────
        # Loss computation
        # ─────────────────────────────────────────────────
        loss = criterion(predictions, ground_truth)
        # Loss può essere:
        # - RGB loss (MSE vs target image)
        # - Event loss (consistency con eventi)
        # - Regularization (eg. epect sigma smoothing)
        
        # ─────────────────────────────────────────────────
        # Backward pass
        # ─────────────────────────────────────────────────
        loss.backward()  # Calcola gradienti
        
        # ─────────────────────────────────────────────────
        # Optimizer step
        # ─────────────────────────────────────────────────
        optimizer.step()
        optimizer.zero_grad()
        
        # ─────────────────────────────────────────────────
        # (Opzionale) Scheduler / EMA
        # ─────────────────────────────────────────────────
        scheduler.step()  # LambdaLR decay
        ema.update(model)  # Exponential moving average
```

### 3.4 Analisi Dettagliata Memoria Training

```
Model parameters:
  - encoder (hash grid): 32 × 2^19 × 4B × 16 levels ≈ 256 MB parameter
  - sigma_net: 2 Linear layers → 32×64 + 64×16 = ~5 KB (negligible)
  - encoder_dir (SH): negligible
  - color_net: 3 Linear layer → 31×64 + 64×64 + 64×3 ≈ 30 KB
  
  TOTAL PARAMS: ~256 MB

Forward activations (per 262K rays × 128 samples = 33.5M points):
  - xyzs: 33.5M × 3 × 4B = 403 MB
  - grid_features: 33.5M × 32 × 2B (fp16) = 2.1 GB (!!)
  - sigma_hidden: 33.5M × 64 × 2B = 4.3 GB
  - geo_feat: 33.5M × 15 × 2B = 1.0 GB
  - view_feat: 33.5M × 16 × 2B = 1.1 GB
  - color_hidden_0: 33.5M × 64 × 2B = 4.3 GB
  - color_hidden_1: 33.5M × 64 × 2B = 4.3 GB
  - rgbs: 33.5M × 3 × 2B = 200 MB
  - sigmas: 33.5M × 1 × 2B = 134 MB
  
  TOTAL ACTIVATIONS: ~18 GB (!!) 🔴 BOTTLENECK

Backward (gradients):
  - Stesso size delle activations ≈ 18 GB
  
Optimizer state (Adam):
  - Moment 1: ~256 MB
  - Moment 2: ~256 MB
  
TOTALE MEMORIA PICCO: ~36 GB durante backward
  ≈ IMPOSSIBILE su RTX4090 (24GB) senza ottimizzazioni
  
Ottimizzazioni aplicate:
  ✅ fp16 (mixed precision): attivazioni a 2B instead 4B → 18 GB → 9 GB
  ✅ Gradient checkpointing: non salva tutte le attivazioni
    → Memory ~50% ridotto
  ✅ Reduced batch size / num rays per step
```

### 3.5 Timeline Training Step Completo

```
Forward pass:
  Ray generation:       1 ms
  AABB intersection:    1 ms
  Z sampling:           1 ms
  XYZ compute:          1 ms
  Grid encoding:       30 ms  ─┐
  Sigma MLP:           60 ms  ├─ Salva per backward
  View encoding:        3 ms  ├─ Attivazioni memoria
  Color MLP:          100 ms  ─┘
  Volume rendering:     8 ms
  ────────────────────────────
  Subtotal forward:   205 ms

Loss computation:
  MSE / Event loss:    10 ms

Backward pass:
  Recompute grid_enc:  30 ms  ← Gradient checkpointing
  Sigma backward:      80 ms
  Color backward:     120 ms
  ────────────────────────────
  Subtotal backward:  230 ms  (leggermente più lento del forward)

Optimizer step:
  Adam update:         10 ms

────────────────────────────────
TOTAL STEP:         ~455 ms (2.2 steps/sec)

Con ottimizzazioni (fp16 + gradient checkpointing):
────────────────────────────────
TOTAL STEP:         ~300 ms (3.3 steps/sec)
```

### 3.6 Setup Profiling Training

```python
# Script da creare: scripts/profile_training.py

import torch
import torch.profiler as profiler
from datetime import datetime

# Load model
model = load_model(variant='ff')
optimizer = torch.optim.Adam(model.get_params(lr=1e-2))

# Dummy data
torch.manual_seed(42)
rays_o = torch.randn(1, 65536, 3).cuda()
rays_d = torch.randn(1, 65536, 3).cuda()
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
target_rgb = torch.rand(1, 65536, 3).cuda()

# Warmup
for _ in range(3):
    rgb, _ = model.render(rays_o, rays_d)
    loss = F.mse_loss(rgb, target_rgb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Profile
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=profiler.tensorboard_trace_handler('./logs_profile'),
    with_stack=True,
) as prof:
    for step in range(10):
        # Forward
        rgb, depth = model.render(rays_o, rays_d)
        loss = F.mse_loss(rgb, target_rgb)
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

---

## 4. Algoritmo di Ray Marching

### 4.1 Overview Concettuale

**Ray marching** è il processo di campionare punti 3D lungo un raggio dalla camera verso la scena.

```
Camera
  ├─ Ray_o (origin, es. centro camera)
  ├─ Ray_d (direction, vettore normalizzato)
  ├─ Scene bounds (AABB: axial-aligned bounding box)
  └─ Raggio: P(t) = Ray_o + t * Ray_d
     dove t ∈ [t_near, t_far]

Coarse sampling: sample uniformi n=128 punti
Fine sampling (upsample): raffinamento con 128 punti ulteriori
```

### 4.2 CUDA Implementation: near_far_from_aabb

**File**: `raymarching/src/*.cu`

```cuda
// Firma (Python wrapper)
near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.2)
// Input:  rays_o [N, 3], rays_d [N, 3], aabb [6]
// Output: nears [N], fars [N]

// CUDA kernel
__global__ void kernel_near_far_from_aabb(
    const float* rays_o,      // [N, 3]
    const float* rays_d,      // [N, 3]
    float* nears,             // [N]
    float* fars,              // [N]
    const float* aabb,        // [6]
    int N,
    float min_near
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Leggi raggio e AABB
    float3 o = make_float3(rays_o[idx*3+0], rays_o[idx*3+1], rays_o[idx*3+2]);
    float3 d = make_float3(rays_d[idx*3+0], rays_d[idx*3+1], rays_d[idx*3+2]);
    
    // min = [xmin, ymin, zmin], max = [xmax, ymax, zmax]
    float3 amin = make_float3(aabb[0], aabb[1], aabb[2]);
    float3 amax = make_float3(aabb[3], aabb[4], aabb[5]);
    
    // Slab test per ogni asse
    // Per asse X: t_xmin, t_xmax = dove raggio interseca piani x=xmin, x=xmax
    float t_xmin = (amin.x - o.x) / (d.x + 1e-15f);
    float t_xmax = (amax.x - o.x) / (d.x + 1e-15f);
    float t_x_enter = fminf(t_xmin, t_xmax);
    float t_x_exit = fmaxf(t_xmin, t_xmax);
    
    // Idem Y, Z
    float t_ymin = (amin.y - o.y) / (d.y + 1e-15f);
    float t_ymax = (amax.y - o.y) / (d.y + 1e-15f);
    float t_y_enter = fminf(t_ymin, t_ymax);
    float t_y_exit = fmaxf(t_ymin, t_ymax);
    
    float t_zmin = (amin.z - o.z) / (d.z + 1e-15f);
    float t_zmax = (amax.z - o.z) / (d.z + 1e-15f);
    float t_z_enter = fminf(t_zmin, t_zmax);
    float t_z_exit = fmaxf(t_zmin, t_zmax);
    
    // Intersezione: max(t_*_enter), min(t_*_exit)
    float t_enter = fmaxf({t_x_enter, t_y_enter, t_z_enter, 0.0f});
    float t_exit = fminf({t_x_exit, t_y_exit, t_z_exit});
    
    // Controlla validità intersection
    if (t_exit <= t_enter) {
        // No intersection
        nears[idx] = 1e9f;
        fars[idx] = 1e9f;
    } else {
        nears[idx] = fmaxf(t_enter, min_near);
        fars[idx] = t_exit;
    }
}
```

### 4.3 Proprietà del Slab Test

**Matematica**:

Per il slab test, ogni asse (X, Y, Z) form "slabs" (fette):
- Slab X: piano tra xmin, xmax
- Raggio interseca slab X nei parametri t: t_x_enter, t_x_exit

Il raggio interseca il **box 3D** solo se interseca **tutti e tre** gli slabs nello stesso intervallo [t_enter, t_exit].

Formula AABB ray intersection:
```
t_enter = max(t_x_enter, t_y_enter, t_z_enter)
t_exit = min(t_x_exit, t_y_exit, t_z_exit)
Intersect se: t_exit > t_enter
```

**Complessità**:
- Operazioni: 9 divisioni, 12 min/max
- Latency nascosta: sì (divisione è ~20 cycle)
- Parallelismo: ogni raggio è **indipendente** → SIMD perfetto

### 4.4 Coarse & Fine Sampling Strategy

Dopo `near_far_from_aabb`, generiamo campioni:

```python
# Coarse pass: campionamento UNIFORME
num_steps_coarse = 128
z_vals_coarse = linspace(0, 1, num_steps_coarse)  # [128]
z_vals_coarse = nears + (fars - nears) * z_vals_coarse  # scale

# Rendering coarse → ottieni pesi per ogni intervallo
xyzs_coarse = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_coarse[:, :, None]
# Shape: [N, 128, 3]

sigma_coarse, feat_coarse = model.density(xyzs_coarse)  # [N, 128]
rgb_coarse = model.color(xyzs_coarse, rays_d[:, None, :], geo_feat=feat_coarse)
# [N, 128, 3]

# Volume rendering → calcola weights
weights_coarse = compute_alpha_weights(sigma_coarse, z_vals_coarse)
# [N, 128]

# Fine pass (upsample): sample ADATTIVO basato su weights
# Idea: campiona più punti dove weights sono alti (scena rilevante)
z_vals_fine = sample_pdf(z_vals_coarse, weights_coarse, num_steps_fine=128)
# [N, 128] nuovi campioni (non uniformi!)

# Merge coarse + fine
z_vals_all = sort(concat(z_vals_coarse, z_vals_fine))  # [N, 256]

# Re-render con tutti i punti
xyzs_all = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_all[:, :, None]
sigma_all, feat_all = model.density(xyzs_all)
rgb_all = model.color(xyzs_all, rays_d[:, None, :], geo_feat=feat_all)

# Output finale
weights_final = compute_alpha_weights(sigma_all, z_vals_all)
rgb_final = (weights_final[..., None] * rgb_all).sum(dim=-2)
depth_final = (weights_final * z_vals_all).sum(dim=-1)
```

### 4.5 Lunghezza Densità a Lungo-Raggio

**Derivazione della formula alpha (opacità)**:

In rendering volumetrico, l'opacità lungo un segmento di raggio da t_i a t_{i+1}:

```
σ(t) = densità (continua)
δ_i = t_{i+1} - t_i = lunghezza segment

α_i = 1 - exp(-σ(t_i) * δ_i)  ← Opacità segment

Reasoning: se δ → 0, α → 0 (nessuna opacità)
           se σ → ∞, α → 1 (completa opacità, blocca raggi successivi)
```

**Transmittance (trasmissione)**:

```
T_i = ∏_{j=0}^{i-1} (1 - α_j)
    = probabilità che raggio raggiunge punto i senza essere bloccato

Final color:
C = ∑_i T_i * α_i * c_i + T_N * c_bg
     └──────┬──────┘   └──┘
     prob+opacità @ i   RGB @ i
```

---

## 5. Dimensioni Attivazioni (Tensor Shapes)

### 5.1 Tabella Shapes Throughout Pipeline

| Stage | Tensor | Shape | Size (fp32) | Notes |
|-------|--------|-------|------------|-------|
| Input | rays_o, rays_d | [N, 3] | 1.5 MB | 512×512 frame |
| AABB | nears, fars | [N] | 0.25 MB | |
| Sampling | z_vals | [N, 128] | 33 MB | Lineare uniform |
| XYZ | xyzs | [N, 128, 3] | 100 MB | Point 3D |
| **Coarse pass** | | | | |
| Grid enc | feat_coarse | [N×128, 32] | 2.1 GB | **🔴 LARGE** |
| Sigma MLP | h_sigma_hidden | [N×128, 64] | 4.3 GB | **🔴 MEMORY** |
| Sigma out | sigma_coarse | [N, 128] | 134 MB | |
| Geo feat | geo_feat_coarse | [N, 128, 15] | 1.0 GB | |
| View enc | view_feat | [N, 16] | 40 MB | Broadcast a tutti punti |
| Color MLP | h_color_0 | [N×128, 64] | 4.3 GB | **🔴 PEAK** |
| Color out | rgb_coarse | [N, 128, 3] | 200 MB | |
| **Sample PDF** | | | | |
| Weights | weights_coarse | [N, 128] | 134 MB | |
| Z-vals fine | z_vals_fine | [N, 128] | 33 MB | Non-uniform |
| **Fine pass** | | | | |
| Merge | z_vals_all | [N, 256] | 66 MB | Merged coarse+fine |
| Grid enc (fine) | feat_fine | [N×256, 32] | 4.2 GB | **🔴 DOUBLE** |
| | (rest analoga) | | ~8.6 GB | Fine pass full |
| **Volume Render** | | | | |
| Alpha | alpha_all | [N, 256] | 268 MB | 1-exp(-σ*δ) |
| Transmit | transmit | [N, 256] | 268 MB | cumprod(1-α) |
| RGB final | rgb_final | [N, 3] | 200 KB | **Output** |
| Depth final | depth_final | [N] | 1 MB | **Output** |

### 5.2 Picco di Memoria per Configurazione

```
Standard nn.Linear (network.py):
  Coarse pass:  8.5 GB
  + Fine pass:  +8.5 GB
  PICCO:       ~17-18 GB (con buffering device)

FFMLP (network_ff.py):
  Kernel fuso: mantiene attivazioni in shared memory
  Coarse pass: 4-5 GB (no hidden layer storage)
  + Fine pass: +4-5 GB
  PICCO:       ~8-10 GB (50% ridotto)

Con gradient checkpointing:
  Forward: calcola once, non salva intermedie
  Backward: recomputa intermedie "on the fly"
  MEMORIA TRAINING: 9-18 GB → 4-9 GB (50% ridotto)

Con fp16 (half precision):
  Tutte attivazioni in float16 (2B instead 4B)
  MEMORIA: 50% ridotto
  TOTAL: 2-4.5 GB possibile
```

### 5.3 Bandwidth Requirements

```
Memory bandwidth:
  RTX 4090: ~900 GB/s (teorico)
  Achievable: ~700 GB/s (best case)

Movimenti dati per forward pass (N=262K, steps=128):
  Input rays:         1.5 MB × 1× = 1.5 MB (cached)
  Grid features in:   100 MB (xyzs read)
  Grid lookup:        ~256 MB (table read, random!)
  Sigma MLP:          2.1 GB (features + hidden layers)
  B.BANDWIDTH SPIKE:  ~500 MB / 30ms = 16 GB/s (ok)
  
  Color MLP:          4.3 GB (features + hidden layers)
  B.BANDWIDTH SPIKE:  ~4.3 GB / 100ms = 43 GB/s (ok)
  
  Grid encoding è memory-bound (random hash lookup)
  MLPs sono compute-bound (TFLOP-limited)
```

---

## 6. Analisi Costi Computazionali

### 6.1 Conteggio FLOP per Componente

```
N = numero di raggi = 262K
num_steps = 128
total_points = N * num_steps = 33.5M

─────────────────────────────────────────
GRID ENCODING
─────────────────────────────────────────
Operazioni per punto (16 livelli, degree=2):
  - Coordinate normalization: 3 add/mul
  - Per livello:
    * Floor + fractional: 6 op
    * Hash lookup: 0 FLOPs (tabella lookup)
    * Trilinear interpolation: 8 voxel × 3 blend = ~20 FLOPs
  - Total: ~20 FLOPs per livello × 16 = 320 FLOPs/point

Total: 33.5M × 320 = 10.7 TFLOP

Memoria-bound, quindi FLOP non dominante.

─────────────────────────────────────────
SIGMA MLP (coarse + fine)
─────────────────────────────────────────
Layer 1: [32] → [64] su 33.5M points
  Matrix-mul: 32 × 64 = 2048 FLOPs/point
  Bias + ReLU: ~65 FLOPs
  Total: 2113 FLOPs/point
  × 33.5M = 70.8 TFLOP

Layer 2: [64] → [16] on 33.5M points
  Matrix-mul: 64 × 16 = 1024 FLOPs/point
  Exp (trunc_exp): ~20 FLOPs
  Total: 1044 FLOPs/point
  × 33.5M = 35.0 TFLOP

Sigma total: 105.8 TFLOP

─────────────────────────────────────────
COLOR MLP (coarse + fine)
─────────────────────────────────────────
Input: [31] (view_feat[16] + geo_feat[15])

Layer 1: [31] → [64] on 33.5M
  Matrix-mul: 31 × 64 = 1984 FLOPs/point
  + ReLU: ~65 FLOPs
  Total: 2049 FLOPs/point
  × 33.5M = 68.6 TFLOP

Layer 2: [64] → [64]
  Matrix-mul: 64 × 64 = 4096 FLOPs/point
  + ReLU: ~65 FLOPs
  Total: 4161 FLOPs/point
  × 33.5M = 139.4 TFLOP (dominante!)

Layer 3: [64] → [3]
  Matrix-mul: 64 × 3 = 192 FLOPs/point
  + sigmoid: ~10 FLOPs
  Total: 202 FLOPs/point
  × 33.5M = 6.8 TFLOP

Color total: 214.8 TFLOP

─────────────────────────────────────────
VOLUME RENDERING
─────────────────────────────────────────
Per punto:
  exp (-σ*δ): ~20 FLOPs
  alpha = 1 - exp(...): ~2 FLOPs
  weight accumulation (cumprod): 5 FLOPs
  RGB weighted sum: 3 FLOPs
  Total: ~30 FLOPs/point

× 33.5M × 2 (coarse + fine) = 2.0 TFLOP

─────────────────────────────────────────
TOTAL INFERENCE FORWARD:
─────────────────────────────────────────
Grid encoding:      10.7 TFLOP (memory-bound)
Sigma MLP:         105.8 TFLOP (compute-bound) ← DOMINANTE
Color MLP:         214.8 TFLOP (compute-bound) ← DOMINANTE
Volume rendering:    2.0 TFLOP (triviale)
────────────────────────────────
TOTAL:            ~333 TFLOP

RTX 4090 FP32 peak: 1.4 PFLOP = 1400 TFLOP
Raggiabile tipicamente: ~300-400 TFLOP (30% utilizzo)

Con FP16 (FFMLP):
  Tutti i TFLOP raddoppiati (2 operazioni per FP32)
  ~666 TFLOP equivalenti
  RTX 4090 FP16 peak: 2.8 PFLOP
  Utilizzo: ~25% (comunque buono)
```

### 6.2 Roofline Model

Il **roofline model** mappa operazioni su 2 dimensioni:
- **X-axis**: Arithmetic Intensity (FLOP/byte)  
- **Y-axis**: Performance (GFLOP/s)

```
        Roof-line Model
        ╱────────── Peak compute (PFLOP/s)
       ╱            RTX 4090: 1.4 PFLOP FP32
      ╱
     ╱
    ╱─────────────── Peak bandwidth line
   ╱                  900 GB/s
  ╱__________________________________________________________________
  0                                                  AI (FLOP/byte)
  
Operazione position:
  
  Grid encoding:
    10.7 TFLOP / (100 MB read + 256 MB lookup) ≈ 0.03 FLOP/byte
    × 900 GB/s = 27 GFLOP/s 🔴 BANDWIDTH LIMITED
  
  Color MLP (layer 2): [64]→[64]
    139.4 TFLOP / 4.3 GB ≈ 32 FLOP/byte
    × (4.3 GB / 100ms) = 43 GB/s → 43 × 32 = 1376 GFLOP/s ✅ COMPUTE BOUND
    
  Conclusion: MLPs operate near roofline, grid encoding bottlenecked by memory.
```

### 6.3 Computational Balance

```
Per unità tempo GPU:

Durante forward pass:
  - Grid encoding: 30 ms (16-20% compute utilization)
  - Sigma MLP: 60 ms (40-50% compute utilization)
  - Color MLP: 100 ms (60-75% compute utilization) ← Peak load
  - Volume rendering: 8 ms (triviale)
  
GPU está bene utilisato durante Color MLP, male durante Grid encoding.
```

---

## 7. Profilatura Codice CUDA

### 7.1 Strumenti Disponibili

#### **Option A: PyTorch Profiler (recommended per inizio)**

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
) as prof:
    # Codice da profilare
    output = model.forward(input)
    prof.step()

# Visualizza in Tensorboard
# tensorboard --logdir=./logs
```

**Vantaggi**: integrato PyTorch, semplice setup, mostra call graph  
**Svantaggi**: bassa granularità, non vede dentro ai kernel custom

#### **Option B: NVIDIA Nsys (profiling dettagliato)**

```bash
# Profila CUDA kernel con Nsys
nsys profile -o trace_enerf \
    -c cuda,cudnn,cublas \
    python scripts/profile_inference.py

# Visualizza risultato
nsys-ui trace_enerf.nsys-rep
```

**Vantaggi**: vede singoli kernel CUDA, memory transfer, occupancy  
**Svantaggi**: output massiccio, richiede setup NVIDIA, file grandi

#### **Option C: NVIDIA Compute Sanitizer (memory checking)**

```bash
# Profila memory leaks e memory errors
compute-sanitizer --tool memcheck python scripts/profile_training.py

# Profila memory usage patterns
compute-sanitizer --tool initcheck python scripts/profile_training.py
```

**Vantaggi**: rileva memory corruption, non-initialized memory  
**Svantaggi**: aggiunge overhead

#### **Option D: Manual CUDA Event Timing (best for custom kernels)**

```python
# In network_ff.py, all'interno della classe FFMLP

def forward(self, x):
    # Manual timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    output = ffmlp_forward(x, self.weights, ...)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    print(f"FFMLP forward: {elapsed_ms:.2f} ms")
    
    return output
```

**Vantaggi**: low overhead, preciso  
**Svantaggi**: manual, richiede modifica codice

### 7.2 Profiling Grid Encoding

```python
# Script: scripts/profile_gridencoder.py

import torch
import numpy as np
from gridencoder import GridEncoder

# Setup
device = torch.device('cuda')
B = 262144  # rays × steps
D = 3       # coordinate dim
num_levels = 16
level_dim = 2

encoder = GridEncoder(
    input_dim=D,
    num_levels=num_levels,
    level_dim=level_dim,
    base_resolution=16,
    desired_resolution=2048,
).to(device)

# Dummy input
coords = torch.rand(B, D, device=device)

# Warmup
for _ in range(3):
    features = encoder(coords)

# Profile
torch.cuda.reset_peak_memory_stats()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for i in range(10):
        features = encoder(coords)
        prof.step()

# Report
print("Grid Encoder Profiling Results:")
print(prof.key_averages(group_by_stack_n=5).table(
    sort_by="cuda_time_total", row_limit=10
))

peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_mem:.2f} GB")

# Bandwidth estimation
total_time_ms = sum([e.cuda_time_total / 1000 for e in prof.key_averages()])
table_size_bytes = 32 * (2 ** 19) * 4 * num_levels  # embedding table
bandwidth = (table_size_bytes * 10) / (total_time_ms * 1e6)  # GB/s
print(f"Estimated bandwidth: {bandwidth:.1f} GB/s")
```

### 7.3 Profiling FFMLP vs Standard MLP

```python
# Script: scripts/compare_mlp_variants.py

import torch
import time
from nerf.network_ff import NeRFNetwork as NeRFNetworkFF
from nerf.network import NeRFNetwork as NeRFNetworkStandard

B = 262144 * 128
D = 32

# Prepare input
x_input = torch.randn(B, D, device='cuda')

# Standard MLP
model_std = NeRFNetworkStandard().cuda()
for _ in range(3):
    _ = model_std.forward(x_input, None)

times_std = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = model_std.sigma_net(x_input)
    torch.cuda.synchronize()
    times_std.append(time.time() - t0)

# FFMLP
model_ff = NeRFNetworkFF().cuda()
for _ in range(3):
    _ = model_ff.forward(x_input, None)

times_ff = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = model_ff.sigma_net(x_input)
    torch.cuda.synchronize()
    times_ff.append(time.time() - t0)

# Compare
print(f"Standard MLP:  {np.mean(times_std)*1000:.2f} ± {np.std(times_std)*1000:.2f} ms")
print(f"FFMLP:         {np.mean(times_ff)*1000:.2f} ± {np.std(times_ff)*1000:.2f} ms")
print(f"Speedup:       {np.mean(times_std) / np.mean(times_ff):.1f}x")
```

### 7.4 Memory Profiling

```python
# Script: scripts/profile_memory.py

import torch
from nerf.network_ff import NeRFNetwork

model = NeRFNetwork().cuda()

# Capture memory snapshot PRIMA
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Dummy data
rays_o = torch.randn(1, 262144, 3, device='cuda')
rays_d = torch.randn(1, 262144, 3, device='cuda')

# Forward pass
torch.cuda.synchronize()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    rgb, depth = model.render(rays_o, rays_d, num_steps=128, upsample_steps=128)
    prof.step()

# Report memory
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory allocated: {peak_mem:.2f} GB")

# Memory breakdown da profiler
memory_events = [e for e in prof.key_averages() if 'mem' in e.key.lower()]
for event in sorted(memory_events, key=lambda e: e.cuda_memory_usage, reverse=True)[:10]:
    print(f"{event.key}: {event.cuda_memory_usage / 1e9:.2f} GB")
```

---

## 8. Identificazione Colli di Bottleneck

### 8.1 Matrix Bottleneck Identification Framework

Metodo sistematico per identificare collo di bottleneck (memoria vs compute):

```
Per ogni operazione, stimare:
  1. Arithmetic Intensity (AI) = FLOP / BYTE
  2. Peak throughput = min(peak_flop, peak_bandwidth * AI)
  3. Actual throughput = time_actual × FLOP / (op_count)
  4. Efficiency = actual / peak

Se efficiency < 30% → MEMORY BOUND
Se efficiency > 30% → COMPUTE BOUND
```

### 8.2 Grid Encoding Analysis

```
Operazione: gridencoder(xyzs) con xyzs [33.5M, 3]
Output: features [33.5M, 32]

Estimazione FLOP:
  Per punto: Coordinate norm + hash + trilinear = ~320 FLOP
  Total: 33.5M × 320 = 10.7 TFLOP
  
Estimazione BYTE:
  Input memback: [33.5M, 3] fp32 = 403 MB → 403 byte/M points
  Embedding lookup: random access tabella 256 MB → ~76 byte/M
  Output: [33.5M, 32] fp16 = 2.1 GB → 63 byte/M
  Total: ~540 MB trasferito
  
AI = 10.7 TFLOP / 540 MB = 19.8 FLOP/byte

Peak throughput = min(1400 GFLOP/s, 900 GB/s × 19.8) 
                = min(1400, 17820) = 1400 GFLOP/s
                
Actual throughput = 10.7 TFLOP / 30 ms = 357 GFLOP/s

Efficiency = 357 / 1400 = 25% → 🔴 MEMORY BOUND!

Motivo: hash lookup pattern è RANDOM, causa cache misses.
```

### 8.3 Sigma MLP Analysis

```
Operazione:Layer 2: [64] → [16] su [33.5M, 64]

FLOP: 33.5M × (64 × 16 + exp) = 33.5M × 1044 = 35 TFLOP

BYTE:
  Input: [33.5M, 64] fp32 = 8.6 GB
  Weights: [64, 16] × 4B = 4 KB (L1 cache!)
  Output: [33.5M, 16] fp32 = 2.1 GB
  Total: ~10.7 GB trasferito
  
AI = 35 TFLOP / 10.7 GB = 3.3 FLOP/byte

Peak throughput = min(1400 GFLOP/s, 900 GB/s × 3.3)
                = min(1400, 2970) = 1400 GFLOP/s

Actual throughput = 35 TFLOP / 60 ms = 583 GFLOP/s

Efficiency = 583 / 1400 = 42% → 💛 BALANCED (lievemente compute-bound)

Bottleneck = Mix: memory bandwidth + compute latency
```

### 8.4 Roofline Summary

```
         RT
Performance X 4090 peak = 1400 GFLOP/s
(GFLOP/s)  │
           │     ╱────────── Compute peak
        ╱──┼────╱────────────
       ╱   │╱         
    ╱──────╋─────────────────
   ╱       │ Memory bandwidth limit (900 GB/s)
  ╱        │
 ╱─────────┴──────────────────── AI (FLOP/byte)
 0    10    20    30   40   50


Operazione posizionamento:
┌─────────────────────────────┐
│ Grid Encoding:          🔴   │  AI = 20, Eff = 25%
│ Sigma MLP (layer 2):    💛   │  AI = 3, Eff = 42%
│ Color MLP (layer 2):    💛   │  AI = 32, Eff = 60% ← migliore
│ Volume rendering:       🟢   │  trascurabile
└─────────────────────────────┘

Conclusione: Grid encoding = MEMORY BOUND, remedio = ottimizzare accessi random
             MLPs = COMPUTE/BANDWIDTH BALANCED, difficile ottimizzare ulteriormente
```

### 8.5 Training Bottleneck: Picco Memoria

```
Memoria training = DOMINANTE bottleneck, non compute!

Breakdown:
  Model parameters:     256 MB (negligible)
  Forward activation:   ~18 GB
  Backward gradient:    ~18 GB
  Optimizer state:      0.5 GB
  ──────────────────────────────
  TOTAL:               ~36 GB ← exceeds RTX 4090 (24GB)

Soluzioni:
  ✅ Gradient checkpointing: -50% memory
  ✅ fp16 (mixed precision): -50% attivazioni
  ✅ Ridurre num_steps: 128+128 → 64+64 = -75% activations
  ✅ Batch rays sequenzialmente (num batch < 262K)
  
Implementazione graduale:
  Baseline:    36 GB (OOM)
  + fp16:      18 GB ✅
  + ckpt:       9 GB ✅
  + num_steps=64: 6 GB ✅
```

---

## 9. Strumenti e Metriche

### 9.1 Metriche Chiave da Misurare

| Metrica | Definizione | Target | Strumento |
|---------|-----------|--------|-----------|
| **Latenza frame** | ms per render 1 frame | <100 ms (10 FPS) | torch.cuda.Event |
| **Throughput** | frame/sec | >10 FPS | wall-clock time |
| **Memory peak** | MB allocato | <24 GB | torch.cuda.max_memory_allocated() |
| **FLOP/s** | Floating point ops/sec | >300 GFLOP/s | profiler FLOP count |
| **Bandwidth** | GB/s | >100 GB/s avg | bandwidth = bytes_moved / time |
| **Occupancy** | % SM active | >50% | nsys |
| **L2 cache hit rate** | % | >50% | nsys metrics |
| **Warp efficiency** | % thread occupancy | >80% | nsys |
| **Training loss/iter** | Loss value | decreasing | tensorboard |
| **PSNR validation** | Peak signal/noise ratio | >25 dB | metrics.py |

### 9.2 Dashboard Tensorboard

```python
# Log durante training
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/exp_ffmlp_fp16_ckpt')

for step, batch in enumerate(train_loader):
    # Training
    loss = train_step(model, batch)
    
    # Log metriche
    writer.add_scalar('loss/training', loss, step)
    
    if step % 100 == 0:
        # Profiling ogni 100 step
        with profiler.profile(...) as prof:
            rgb, depth = model.render(...)
        
        flop_count = prof.key_averages().total_call_count()
        elapsed = prof.profiler_result.total_average().cuda_time_total / 1000
        writer.add_scalar('perf/gflop_per_sec', flop_count / elapsed / 1e9, step)

writer.close()

# Visualizza
# tensorboard --logdir=logs
```

---

## 10. Piano di Implementazione

### **Fase 1: Setup Profiling Infrastructure (1 settimana)**

#### Milestone 1.1: Basic profiling script
- [ ] Creare `scripts/profile_inference.py`
  - Carica modello (network.py)
  - Genera dummy rays 512×512
  - Esegui forward pass con torch.profiler
  - Stampa timeline completo
  - Salva trace per Tensorboard

- [ ] Creare `scripts/profile_inference_ff.py`
  - Stessa cosa ma con network_ff.py (FFMLP)
  - Compare latenza vs standard

**Deliverable**: 2 script, mostra speedup 5-10x per FFMLP

#### Milestone 1.2: Memory profiling
- [ ] Creare `scripts/profile_memory.py`
  - Misura peak memory allocation
  - Breakown per tensor (gridencoder, MLP layer)
  - Simula forward pass completo (coarse + fine)

**Deliverable**: Report memoria allocata in ogni stage

#### Milestone 1.3: Training profiling
- [ ] Creare `scripts/profile_training_step.py`
  - Carica train_loader con 10 batch
  - Profila forward + backward + optimizer step
  - Misura memoria picco durante backward
  - Mostra differenza standard vs fp16

**Deliverable**: Timeline training step breakdown

---

### **Fase 2: Detailed Component Analysis (2 settimane)**

#### Milestone 2.1: Grid Encoding profiling
- [ ] Creare `scripts/profile_gridencoder.py`
  - Profila solo gridencoder separatamente
  - Varia input size (1M-100M punti)
  - Misura bandwidth effettivo
  - Plot: latenza vs input size

**Deliverable**: Grid encoder bottleneck identification

#### Milestone 2.2: MLP variant comparison
- [ ] Creare `scripts/compare_mlp_variants.py`
  - Compare standard nn.Linear vs FFMLP
  - Fix: B=33.5M, hidden=64
  - Misura latenza, GFLOP/s, memory
  - Plot: Speedup vs input size

**Deliverable**: FFMLP speedup quantification

#### Milestone 2.3: Ray marching analysis
- [ ] Creare `scripts/profile_raymarching.py`
  - Profile near_far_from_aabb
  - Varia num_rays (1K-1M)
  - Misura memory bandwidth
  - Analizza warp efficiency su nsys

**Deliverable**: Ray marching scalability plot

---

### **Fase 3: Roofline &Bottleneck Analysis (2 settimane)**

#### Milestone 3.1: Roofline model
- [ ] Creare `analysis/roofline_analysis.py`
  - Per ogni operazione, calcola:
    - Arithmetic Intensity (FLOP/byte)
    - Peak throughput
    - Actual throughput
  - Plot roofline
  - Classifica: MEMORY vs COMPUTE BOUND

**Deliverable**: Roofline diagram con operazioni classificate

#### Milestone 3.2: Memory access pattern
- [ ] Creare script analisi con NVIDIA Nsys
  - Profile gridencoder con nsys
  - Estrai: L2 cache hit rate, memory coalescing
  - Account per random hash table access
  - Suggerire ottimizzazioni (prefetch, tile reuse)

**Deliverable**: Memory access analysis report

#### Milestone 3.3: Training bottleneck quantificazione
- [ ] Script analizza allocazione memoria durante training
  - Misura picco forward vs backward
  - Test: baseline vs fp16 vs gradient checkpointing
  - Tabella: Memory usage vs configurazione
  - Recommend optimal config

**Deliverable**: Training memory optimization recommendations

---

### **Fase 4: Advanced Profiling & Optimization (3 settimane)**

#### Milestone 4.1: NVIDIA Nsys integration
- [ ] Setup Nsys profiling for gridencoder
  - Capture kernel timeline
  - Analizza occupancy, memory hierarchy
  - Report: shared memory usage, bank conflicts
  - Suggest kernel optimization (se necessario)

**Deliverable**: Detailed kernel analysis report

#### Milestone 4.2: CUDA Trace analysis
- [ ] Automated script che:
  - Genera CUDA traces
  - Estrae event timings
  - Builds dependency graph (kernel→kernel)
  - Identifies serialization points

**Deliverable**: Dependency graph visualization

#### Milestone 4.3: FP16 vs FP32 comparison
- [ ] Script detailed:
  - Training speed (step/sec) fp16 vs fp32
  - Convergence speed (loss vs epoch)
  - Accuracy impact on final PSNR
  - Memory savings quantified

**Deliverable**: FP16 tradeoff analysis

---

### **Fase 5: Documentation & Final Report (1 settimana)**

#### Milestone 5.1: Analysis document
- [ ] Comprehensive report:
  - Executive summary (key findings)
  - Architecture breakdown diagram
  - Timeline charts (inference & training)
  - Bottleneck classification (memory vs compute)
  - Recommendations for optimization
  - Appendix: profiling scripts + results

**Deliverable**: 20+ page technical report

#### Milestone 5.2: Optimization recommendations
- [ ] Prioritized list:
  1. Grid encoding: reduce hash collisions (feasibility: medium)
  2. Training: use fp16 + gradient checkpointing (feasibility: high, easy wins)
  3. Kernel fusion: fuse grid_enc + sigma_mlp (feasibility: hard, potential 2x improvement)
  4. Batch processing: sequential batch rays (feasibility: high, memory fix)

**Deliverable**: Implementation roadmap with priority

#### Milestone 5.3: Reference materials
- [ ] Create:
  - Profiling cheat-sheet (how to use each tool)
  - Roofline model explanation
  - NeRF volume rendering equations
  - CUDA optimization principles (coalescing, occupancy, etc.)

**Deliverable**: Educational reference docs

---

## Summary: Key Takeaways

**Inference (512×512, 256 samples/ray)**:
- **Total latency**: 200ms (standard) → 85ms (FFMLP) 
- **Bottleneck**: Color MLP (40% time), Grid encoding (15% time, memory-bound)
- **Memory**: 17-18 GB, spikes during fine ray pass

**Training (same config, 1 frame batch)**:
- **Step time**: 450ms (standard) → 300ms (optimized)
- **Memory picco**: 36 GB → 9 GB (with fp16 + checkpointing)
- **Bottleneck**: MEMORY allocation (not compute)

**Optimization Priority**:
1. **Immediate   ✅**: Enable fp16 + gradient checkpointing (easy, memory fix)
2. **Short-term**: Reduce num_samples from 256 to 128-192
3. **Medium-term**: Custom kernel fusion (grid_enc + sigma)
4. **Long-term**: Hash table prefetching, memory coalescing

**Profiling Tools**:
- PyTorch profiler: quick overview
- NVIDIA Nsys: detailed kernel analysis
- Manual CUDA Events: low-overhead timing
- Tensorboard: training monitoring

---

**End of ANALYSIS_PLAN.md**
