# 🎯 Raymarching: Diagrammi di Interazione

**Visualizzazione di: Come i file si parlano fra loro, flusso dati, architettura**

---

## 1. ARCHITECTURE DIAGRAM (Layer Stack)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    PYTHON LAYER                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ raymarching.py                                       │  │
│  │ ┌─────────────────────────────────────────────────┐  │  │
│  │ │ class _near_far_from_aabb(Function):            │  │  │
│  │ │   @staticmethod                                 │  │  │
│  │ │   @custom_fwd(cast_inputs=torch.float32)        │  │  │
│  │ │   def forward(ctx, rays_o, rays_d, ...):        │  │  │
│  │ │       ... type conversions ...                  │  │  │
│  │ │       _backend.near_far_from_aabb(...)  ┐───┐  │  │  │
│  │ └─────────────────────────────────────────┼──┬┘  │  │  │
│  │   near_far_from_aabb = _near_far_from_aabb.apply  │  │  │
│  └──────────────────────────────────────────────────┬─┘  │
│                 ↑                                    │     │
│           (pytorch API)                           (bind)  │
│                                                    ↓ ↓    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ __init__.py                                          │  │
│  │ from .raymarching import *                           │  │
│  │ (Esporta tutto a public API)                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓ (c'è un problema con binding)
                            
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    C++ BINDING LAYER                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ backend.py: torch.utils.cpp_extension.load()        │  │
│  │                                                      │  │
│  │ (JIT compila first-run)  ──────────────┐            │  │
│  │   nvcc flags: -O3 -std=c++14           │            │  │
│  │   cl.exe flags: /O2 /std:c++17         │ Config     │  │
│  │                                        ↓            │  │
│  │   sources: ['raymarching.cu',                       │  │
│  │            'bindings.cpp']             ← Files      │  │
│  │                                        ↓            │  │
│  │   name='_raymarching'  ──→  _raymarching.pyd/.so    │  │
│  │   (Binary module, loadable da Python)               │  │
│  │                                                      │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          ↓ (binary module in Python)
                    
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    C++ BINDING LAYER                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ src/bindings.cpp                                     │  │
│  │                                                      │  │
│  │ Estrae pointer da torch::Tensor                      │  │
│  │ ├─ rays_o.packed_accessor32<float, 2>               │  │
│  │ └─ rays_d.packed_accessor32<float, 2>               │  │
│  │                                                      │  │
│  │ Calcola grid/block CUDA:                            │  │
│  │   threads = 512                                      │  │
│  │   blocks = (N + threads - 1) / threads              │  │
│  │                                                      │  │
│  │ Lancia kernel:                                       │  │
│  │   near_far_kernel<<<blocks, threads>>>(...) ──┐     │  │
│  │                                               │     │  │
│  │ PYBIND11 registra per Python:                 │     │  │
│  │   m.def("near_far_from_aabb", ...)            │     │  │
│  │                                               │     │  │
│  └───────────────────────┬──────────────────────┘     │  │
│                         │                             │     │
└─────────────────────────┼─────────────────────────────────┘
                          ↓ (actual CUDA code)
                    
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                     CUDA KERNEL LAYER                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ src/raymarching.cu                                   │  │
│  │                                                      │  │
│  │ __global__ void near_far_from_aabb_kernel(...) {    │  │
│  │   int idx = blockIdx.x * blockDim.x + threadIdx.x;  │  │
│  │   if (idx >= N) return;                              │  │
│  │                                                      │  │
│  │   // Slab test algorithm (9 flops per ray)          │  │
│  │   float t_near = compute_t_near(...);               │  │
│  │   float t_far = compute_t_far(...);                 │  │
│  │                                                      │  │
│  │   nears[idx] = t_near;   // Write to global memory  │  │
│  │   fars[idx] = t_far;                                │  │
│  │ }                                                    │  │
│  │                                                      │  │
│  │ PARALLELISM:                                        │  │
│  │   1024 rays → 1024 threads in parallel              │  │
│  │   4 blocks × 256 threads/block                      │  │
│  │   Execution time: 0.5-1 ms                          │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ src/pcg32.h                                          │  │
│  │ Random number generator header (used by kernels)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. DATA FLOW DIAGRAM (Forward Pass)

```
USER INPUT (CPU or GPU):
  rays_o: torch.Tensor [1024, 3] ─┐
  rays_d: torch.Tensor [1024, 3] ─┤
  aabb: torch.Tensor [6]           │
                                 │
                                 ↓
        ┌───────────────────────────────────────┐
        │ raymarching.py: near_far_from_aabb()  │
        │ (Python wrapper)                      │
        ├───────────────────────────────────────┤
        │ Step 1: Type Checking                 │
        │   if not rays_o.is_cuda():            │
        │     rays_o = rays_o.cuda()            │
        │   (Move to GPU if on CPU)             │
        │                                       │
        │ Step 2: Memory Layout                 │
        │   rays_o = rays_o.contiguous()        │
        │   (Ensure C-contiguous memory)        │
        │   rays_o = rays_o.view(-1, 3)         │
        │   (Reshape [1024, 3])                 │
        │                                       │
        │ Step 3: Allocate Output Tensors       │
        │   nears = torch.empty(1024, dtype=f32)│
        │   fars = torch.empty(1024, dtype=f32) │
        │                                       │
        │ Step 4: CUDA Binding Call             │
        │   _backend.near_far_from_aabb(        │
        │     rays_o, rays_d, aabb, N, ...→─ ┐  │
        └───────────────────────────────────────┘
                                             │
                                             ↓
        ┌─────────────────────────────────────────────────────┐
        │ bindings.cpp: near_far_from_aabb()                  │
        │ (C++ wrapper)                                       │
        ├─────────────────────────────────────────────────────┤
        │ Step 1: Extract Tensor Pointers                      │
        │   auto rays_o_ptr = rays_o.data<float>();          │
        │   auto nears_ptr = nears.data<float>();            │
        │   (Get underlying GPU memory pointers)               │
        │                                                      │
        │ Step 2: Compute CUDA Grid                           │
        │   int threads = 512;                                │
        │   int blocks = (N + 511) / 512;  // 1024→2 blocks   │
        │   (Parallel execution layout)                       │
        │                                                      │
        │ Step 3: Launch CUDA Kernel  ┌─ (<<<4, 256>>>)      │
        │   near_far_kernel<<<...>>>( │                       │
        │     rays_o_ptr, rays_d_ptr,─┤                       │
        │     nears_ptr, fars_ptr)    │                       │
        │     ↑                        │                       │
        │ (Grid of threads on GPU)    │                       │
        │                             │                       │
        > Step 4: cudaDeviceSynchronize()                     │
        │   (Wait for all threads to finish)                  │
        │                                                      │
        │ Step 5: Return Control                              │
        │   Return nears, fars buffers (GPU memory)           │
        │                                                      │
        └──────────────────────────────────────────────────────┘
                                             │
                                             ↓
        ┌─────────────────────────────────────────────────────┐
        │ raymarching.cu: CUDA KERNEL                         │
        │ (GPU parallel computation)                          │
        ├─────────────────────────────────────────────────────┤
        │                                                      │
        │ GPU Memory Layout:                                  │
        │ ┌──────────────┐  ┌──────────────┐                 │
        │ │ Global Memory│  │ Global Memory│                 │
        │ │ rays_o [1024]│  │ nears [1024] │ ← Output        │
        │ │ rays_d [1024]│  │ fars [1024]  │ ← Output        │
        │ │ aabb [6]     │  │              │                 │
        │ └──────────────┘  └──────────────┘                 │
        │                                                      │
        │ Thread Grid: 4 blocks × 256 threads = 1024 total    │
        │                                                      │
        │ Block 0: threads 0-255   process rays 0-255         │
        │ Block 1: threads 256-511 process rays 256-511       │
        │ Block 2: threads 512-767 process rays 512-767       │
        │ Block 3: threads 768-1023 process rays 768-1023     │
        │                                                      │
        │ Each thread executes:                               │
        │ ─────────────────────                               │
        │ int idx = blockIdx.x * blockDim.x + threadIdx.x;    │
        │ if (idx >= N) return; // Out of bounds             │
        │                                                      │
        │ // AABB Slab Test Algorithm                         │
        │ float t_min = __fdividef(aabb[0] - rays_o[idx].x,  │
        │                           rays_d[idx].x);           │
        │ float t_max = __fdividef(aabb[3] - rays_o[idx].x,  │
        │                           rays_d[idx].x);           │
        │ // Repeti per Y, Z → 6 operazioni                   │
        │ // Merge 3 intervals → t_near, t_far                │
        │                                                      │
        │ nears[idx] = t_near;  // Write result to output     │
        │ fars[idx] = t_far;                                  │
        │                                                      │
        │ TIMING:                                             │
        │ - Shared memory access: ~1 cycle                    │
        │ - Global memory write: ~100 cycles (hidden by bulk)  │
        │ - Total per thread: 150-200 cycles                  │
        │ - 1024 threads in parallel: 150-200 cycles TOTAL    │
        │   (vs 1024×150 = 153,600 if sequential on CPU)     │
        │                                                      │
        └──────────────────────────────────────────────────────┘
                                             │
                                             ↓
PYTORCH AUTOGRAD TRACKING (if needed for backward):
  - nears, fars have requires_grad=False (no backward for geom ops)
  - But graph is tracked if outputs feed to learnable operations
  
OUTPUT (GPU Memory):
  nears: torch.Tensor [1024] ← t_near for each ray
  fars: torch.Tensor [1024]  ← t_far for each ray
```

---

## 3. BACKWARD PASS DIAGRAM (Example: composite_rays_train)

```
FORWARD:
  sigmas [M]     ─┐  
  rgbs [M, 3]    ├─→ composite_rays_train.forward(ctx, ...)
  deltas [M, 2]  │    │
  rays [N, 3]    ┘    │ ctx.save_for_backward(...)
                      │ ctx.dims = [M, N]
                      ↓
                    ┌──────────────────────────┐
                    │ compute volume rendering │
                    ├──────────────────────────┤
                    │ for i in range(M):       │
                    │   alpha_i = 1-exp(...)  │
                    │   weight_i = trans*alpha│
                    │   pixel += color*weight │
                    └──────────────────────────┘
                      │
                      ↓
                OUTPUT:
                  weights_sum [N]   ← alpha channel
                  depth [N]         ← depth estimates
                  image [N, 3]      ← RGB after compositing
                      │
LOSS COMPUTATION:
                  l_rgb = mse(image, gt_image)
                  loss = l_rgb.mean()
                      │
BACKWARD (loss.backward()):
                      │
                      ↓
        ┌─────────────────────────────────────────┐
        │ PyTorch builds backward graph:           │
        │                                          │
        │ grad_weights_sum = 0 (not used in loss) │
        │ grad_depth = 0 (not used in loss)       │
        │ grad_image = ∂loss/∂image [N, 3]        │
        │             = 2 * (image - gt) / N      │
        │                                          │
        │ Traverses chain of operations backward  │
        │ and calls .backward() on each Function  │
        └──────────────────────────────────────────┘
                      │
custom_bwd triggers: │
                      ↓
        ┌─────────────────────────────────────────────────┐
        │ composite_rays_train.backward(ctx,              │
        │   grad_weights_sum=0,                           │
        │   grad_depth=0,                                 │
        │   grad_image=[N, 3] ← dL/dimage                │
        │ )                                               │
        ├─────────────────────────────────────────────────┤
        │                                                  │
        │ Step 1: Retrieve Saved Tensors                  │
        │   sigmas, rgbs, deltas, rays,        ← from ctx│
        │   weights_sum, depth, image = ctx.saved_tensors│
        │                                                  │
        │ Step 2: Allocate Gradient Tensors               │
        │   grad_sigmas = torch.zeros_like(sigmas)        │
        │   grad_rgbs = torch.zeros_like(rgbs)            │
        │                                                  │
        │ Step 3: Call CUDA Backward Kernel               │
        │   _backend.composite_rays_train_backward(       │
        │     grad_weights_sum,        ← 0 (unused)       │
        │     grad_image,              ← ∂L/∂image (input)│
        │     sigmas, rgbs, deltas,    ← forward values   │
        │     weights_sum, image,      ← forward results  │
        │     grad_sigmas, grad_rgbs   ← OUTPUT buffers   │
        │   )                                              │
        │                                                  │
        │   CUDA Kernel Computation (Parallel):           │
        │   ──────────────────────────────────            │
        │   For each sample m in parallel:                │
        │     ∂L/∂sigma[m] = grad_image · ∂pixel/∂sigma│
        │                   = ∂/∂sigma[(1-exp(-s*d))*...] │
        │     ∂L/∂rgb[m] = grad_image · ∂pixel/∂rgb      │
        │                 = weight[m]                      │
        │                                                  │
        │ Step 4: Return Gradients                        │
        │   return (grad_sigmas,  ← ∂L/∂sigmas [M]        │
        │           grad_rgbs,    ← ∂L/∂rgbs [M, 3]       │
        │           None,         ← ∂L/∂deltas (not needed)
        │           None)         ← ∂L/∂rays (not needed) │
        │                                                  │
        └─────────────────────────────────────────────────┘
                      │
                      ↓ (autograd continues chain)
        
        PyTorch backward continues:
        
        grad_sigmas [M] ────→ color_network.backward()
                              └─→ color_net.weight.grad updated
                              
        grad_rgbs [M, 3] ───→ rgbs no longer used in graph
                              (dead-end for backprop)
        
        
        At end of backward pass:
        - sigmoid_net.weight.grad ← accumulated gradients
        - color_net.weight.grad   ← accumulated gradients
        - optimizer.step() updates them
```

---

## 4. FILE DEPENDENCIES GRAPH

```
User Code
  ↓
raymarching/__init__.py
  ├─ from .raymarching import *
  │
raymarching/raymarching.py
  ├─ import torch
  ├─ from torch.autograd import Function
  ├─ from torch.cuda.amp import custom_fwd, custom_bwd
  │
  └─ try: import _raymarching as _backend
     ├─ SUCCESS (if precompiled): Jump to line "binary module loaded"
     │
     └─ EXCEPT ImportError:
        └─ from .backend import _backend
           │
           raymarching/backend.py
           ├─ import os
           ├─ from torch.utils.cpp_extension import load
           │
           └─ load(name='_raymarching',
                    sources=['src/raymarching.cu', 'src/bindings.cpp'],
                    extra_cuda_cflags=[...],
                    extra_cflags=[...])
              │
              ├─ Calls nvcc compiler
              │  └─ raymarching/src/raymarching.cu
              │     └─ #include "raymarching.h"
              │     └─ #include "pcg32.h"
              │
              ├─ Calls cl.exe (Windows) or gcc (Linux)
              │  └─ raymarching/src/bindings.cpp
              │     ├─ #include <torch/extension.h>
              │     ├─ #include "raymarching.h"
              │     └─ PYBIND11_MODULE(...)
              │
              └─ Produces: _raymarching.pyd (Windows) or .so (Linux)
                 └─ Binary module loadable by Python

           Then: back to raymarching.py
                 _backend = loaded binary module
                 
Binary module loaded:
  └─ _backend contains C++ functions:
     ├─ near_far_from_aabb()
     ├─ composite_rays_train_forward()
     ├─ composite_rays_train_backward()
     ├─ march_rays_train()
     ├─ march_rays()
     ├─ composite_rays()
     ├─ compact_rays()
     └─ ... (all CUDA kernels wrapped)

raymarching/setup.py
  └─ Alternative compilation method (pip install -e .)
     └─ Uses same CUDAExtension + BuildExtension
     └─ Same output: _raymarching binary
```

---

## 5. CLASS HIERARCHY (Autograd Design Pattern)

```
PyTorch autograd.Function
  │
  └─ _near_far_from_aabb (Custom Operation 1)
     ├─ @staticmethod forward(ctx, rays_o, rays_d, aabb, min_near)
     └─ No backward (geometric operation)
  
  └─ _march_rays_train (Custom Operation 2)
     ├─ @staticmethod forward(ctx, ...) 
     │  └─ returns: xyzs, dirs, deltas, rays
     └─ No backward field needed (sampling op)
  
  └─ _composite_rays_train (Custom Operation 3)
     ├─ @staticmethod forward(ctx, sigmas, rgbs, deltas, rays)
     │  ├─ ctx.save_for_backward(...)
     │  └─ returns: weights_sum, depth, image
     │
     └─ @staticmethod backward(ctx, grad_weights_sum, grad_depth, grad_image)
        ├─ retrieve: sigmas, rgbs, deltas, rays from ctx.saved_tensors
        └─ return: grad_sigmas, grad_rgbs, None, None
  
  └─ ... (more operations)


USAGE PATTERN:

class _Operation(Function):
    @staticmethod
    def forward(ctx, *inputs):
        # Compute + optionally save for backward
        ctx.save_for_backward(tensors_needed_for_grad)
        ctx.metadata = simple_values
        return outputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        # Retrieve saved and compute gradients
        saved = ctx.saved_tensors
        return grad_inputs

Operation = _Operation.apply  # Export

# User uses:
output = Operation(input1, input2, ...)
# Internally: calls forward, tracks in autograd graph
```

---

## 6. MEMORY LAYOUT VISUALIZATION

```
TENSOR: rays_o = torch.randn(1024, 3).cuda()

┌─ BEFORE contiguous() ───────────────────┐
│ Shape: [1024, 3]                        │
│ Dtype: float32 (4 bytes per element)    │
│ Device: GPU (CUDA)                      │
│                                         │
│ Stored in GPU Global Memory:            │
│ [0.234, -0.512, 0.891, | 0.123, 0.456, │ (might not be linear!)
│                                         │
│ Stride: (3*4, 4) or irregular           │
│ (If transposed, stride could be complex)│
│                                         │
└─────────────────────────────────────────┘

┌─ AFTER contiguous() ────────────────────┐
│ Shape: [1024, 3] (unchanged)            │
│ Dtype: float32 (unchanged)              │
│ Device: GPU (unchanged)                 │
│                                         │
│ Stored in GPU Global Memory:            │
│ [0.234, -0.512, 0.891, 0.123, 0.456,.. │ ← LINEAR layout
│  ^row 0        row 1        row 2...    │
│                                         │
│ Stride: (12, 4)  ← Row stride = 3*4    │
│ (each row is 3 floats = 12 bytes apart) │
│                                         │
│ CUDA kernel accesses as:                │
│ rays_o[i] = base + i*12  (row access)  │
│ rays_o[i][j] = base + i*12 + j*4       │
│ (sequential memory reads: FAST!)        │
│                                         │
└─────────────────────────────────────────┘

┌─ CUDA Memory Access Pattern ────────────┐
│                                         │
│ Thread 0: reads rays_o[0] → address 0   │
│ Thread 1: reads rays_o[1] → address 12  │ ← Coalesced!
│ Thread 2: reads rays_o[2] → address 24  │ (one mem transaction)
│ ...                                     │
│ Thread 31: reads rays_o[31] → address   │
│                                         │
│ WARP (32 threads) aligned access:       │
│ All reads hit same cache line           │
│ Bandwidth utilization: >90%             │
│                                         │
└─────────────────────────────────────────┘
```

---

## 7. Compilation Timeline (First Run)

```
Time T0: User imports raymarching
  ↓
Time T0+10ms: __init__.py executes
  ↓
Time T0+20ms: raymarching.py executes "import _raymarching"
  ↓
Time T0+30ms: ImportError caught (first time)
  ↓
Time T0+40ms: backend.py executes
  ├─ torch.cpp_extension.load() called
  ├─ Checks for cl.exe on Windows (calls find_cl_path())
  ├─ Adds cl.exe to PATH if needed
  │
  └─ COMPILATION STARTED
    │
    ├─ nvcc compiles raymarching.cu → binary .obj files
    │  (Time: 5-10 seconds, depends on GPU driver)
    │
    ├─ cl.exe compiles bindings.cpp → .obj files
    │  (Time: 1-2 seconds)
    │
    ├─ Linker links .obj + CUDA libraries → _raymarching.pyd
    │  (Time: 1-2 seconds)
    │
    └─ COMPILATION DONE
      (Total: 10-15 seconds first time)
  
  │
  └─ Python loads _raymarching.pyd into memory
     ├─ Imports all functions: near_far_from_aabb, etc.
     └─ READY FOR USE

Time T0+15sec: _backend fully loaded, back to raymarching.py
  │
  └─ "from .backend import _backend" completes
     │
     └─ rest of raymarching.py executes (instant)

Time T0+16sec: User code can call near_far_from_aabb()
  └─ First CALL: FAST (<1ms computation)
     (Compilation happened once, reused thereafter)


SECOND RUN (After restart):
  └─ Binary already on disk (_raymarching.pyd)
     ├─ Try import _raymarching (precompiled)
     ├─ SUCCESS: loads from disk
     ├─ No recompilation
     └─ Time: ~50ms (just loading)
```

---

## 8. SEMANTIC DESIGN PATTERNS

### Pattern 1: **Wrapper + Backend Separation**

```
GOAL: Hide CUDA complexity from user, provide clean PyTorch API

┌─ Frontend (raymarching.py) ─┐
│ User-facing API             │
│ ├─ Type checking            │
│ ├─ Tensor shape validation  │
│ └─ Autograd integration     │
└─────────────────────────────┘
           ↓

┌─ Backend (backend.py + src/*) ─┐
│ Low-level implementation        │
│ ├─ C++ binding layer           │
│ ├─ CUDA kernel implementations │
│ └─ Raw computation            │
└────────────────────────────────┘

BENEFIT:
- User code portable (remove CUDA, add CPU backend)
- Easy to profile (frontend has overhead, backend doesn't)
- Separate concerns (Python logic vs GPU logic)
```

### Pattern 2: **`@staticmethod` for Functional Purity**

```
WHY USE @staticmethod?

Option A: Instance method
  class Renderer:
      def __init__(self, config):
          self.config = config
      
      def near_far(self, rays_o, rays_d):
          # Can access self.config
          return self.compute_near_far(...)
  
  renderer = Renderer(config)
  nears, fars = renderer.near_far(rays_o, rays_d)  ← Requires instance

Option B: Static method
  class _near_far(Function):
      @staticmethod
      def forward(ctx, rays_o, rays_d, aabb):
          # No access to self.state
          # Pure logic: input → output
          return nears, fars
  
  near_far = _near_far.apply
  nears, fars = near_far(rays_o, rays_d, aabb)  ← No instance needed

CHOSEN: Option B because:
- Pure function: input → output (testable, cacheable)
- No side effects (thread-safe in CUDA)
- Simpler API for users
- Custom Autograd expects stateless forward()
```

### Pattern 3: **Backward-less Operations**

```
SOME operations don't need backward:

❌ near_far_from_aabb:
   - Input: ray origins/directions (not learned)
   - Output: t_near, t_far (used for sampling, not loss)
   - Gradient: None (geometric operation)
   
✓ composite_rays_train:
   - Input: sigmas, rgbs (from neural networks, learned)
   - Output: image (fed to loss)
   - Gradient: Yes (backprop through rendering)
   
IMPLEMENTATION:

class _near_far_from_aabb(Function):
    @staticmethod
    def forward(ctx, rays_o, rays_d, ...):
        ...
        # ctx.save_for_backward() NOT CALLED
        return nears, fars
    
    # No backward method! (implicitly returns None)

class _composite_rays_train(Function):
    @staticmethod
    def forward(ctx, sigmas, rgbs, ...):
        ...
        ctx.save_for_backward(...)  ← Save for backprop
        return weights_sum, depth, image
    
    @staticmethod
    def backward(ctx, grad_weights_sum, ...):
        ...
        return grad_sigmas, grad_rgbs, None, None
```

---

**Conclusione**: Raymarching è un **elegante esempio di Python ↔ GPU bridging** che combina:
1. **Python frontend**: Clean API + Autograd
2. **C++ binding**: Type marshalling + thread launching
3. **CUDA kernel**: Parallel computation engine

Ogni livello ottimizzato per il suo ruolo! 🚀
