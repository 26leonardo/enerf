# 📊 FFMLP: Diagrammi Architettura e Flussi Dati

Rappresentazioni visive del flusso computazionale di FFMLP.

---

## 1. Architettura Complessiva (3-Layer Stack)

```
┌──────────────────────────────────────────────────────────────────────┐
│                           PYTHON LAYER                               │
│                         (ffmlp.py)                                    │
│                                                                      │
│  class FFMLP(nn.Module):                                             │
│    def forward(inputs):                                              │
│      return ffmlp_forward(inputs, weights, ...)                      │
│                                                                      │
│  class _ffmlp_forward(Function):  ← Custom autograd                 │
│    def forward(ctx, input, weights, ...):                            │
│      outputs = _backend.ffmlp_forward(...)  ← Chiama C++            │
│    def backward(ctx, grad):                                          │
│      grad_weights = _backend.ffmlp_backward(...)  ← Chiama C++      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                          C++ BINDING LAYER                            │
│                      (bindings.cpp)                                   │
│                                                                      │
│  PYBIND11_MODULE(_ffmlp, m) {                                        │
│    m.def("ffmlp_forward", &ffmlp_forward);   ← Export to Python     │
│    m.def("ffmlp_backward", &ffmlp_backward);                         │
│  }                                                                   │
│                                                                      │
│  void ffmlp_forward(...) {  ← C++ function                           │
│    // Extract tensor pointers                                        │
│    // Compute grid/block dimensions                                  │
│    kernel_mlp_fused<<<grid, block, shmem>>>(...); ← Launch kernel   │
│  }                                                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         CUDA KERNEL LAYER                            │
│                      (ffmlp.cu)                                      │
│                                                                      │
│  __global__ kernel_mlp_fused<<<N_blocks, threads, shmem>>>() {       │
│    // All MLP computation (forward + activations)                    │
│    // All operations fused in single kernel                          │
│    // Returns outputs to global memory                               │
│  }                                                                   │
│                                                                      │
│   __global__ kernel_mlp_fused_backward<<<...>>>() {                  │
│    // All backpropagation (backward activations + grad computation)  │
│    // Returns gradients                                              │
│  }                                                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Forward Pass Data Flow (Detailed)

```
INPUT:
   inputs [B, input_dim]
   weights [num_params] (flattened)
           │
           ↓
   ┌──────────────────────────────┐
   │  C++ ffmlp_forward()         │
   │  ├─ Validate tensor sanity   │
   │  ├─ Compute grid dimensions  │
   │  └─ Launch kernel            │
   └──────────────────────────────┘
           │
           ↓ (via <<<grid, threads, shmem>>>)
   
   ┌────────────────────────────────────────────────────────────┐
   │           CUDA DEVICE: kernel_mlp_fused                    │
   └────────────────────────────────────────────────────────────┘
           │
           ├─ Thread 0-31 (warp 0): Process batch[0:16]
           │                        Layer 0: in_dim → hidden_dim
           │                        Layer 1..n-1: hidden → hidden
           │                        Layer n: hidden → output_dim
           │
           ├─ Thread 32-63 (warp 1): Process batch[16:32]
           │
           ├─ ...(more warps for more batch items)
           │
           └─ Block 0-511: Process different batch chunks in parallel
   
           │
           ↓ (After all blocks complete)

OUTPUT:
   outputs [B, output_dim]
   forward_buffer [num_layers, B, hidden_dim] (if training)
```

---

## 3. Forward Kernel Data Flow (Internal)

```
┌─────────────────────────────────────┐
│ Global Memory (Input from Host)     │
│  inputs [B, input_dim]              │
│  weights [num_params]               │
└─────────────────────────────────────┘
              │
              ├─ Read by multiple blocks
              │
              ↓
┌─────────────────────────────────────────────────────────────┐
│ Shared Memory (per block, ~48KB)                            │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Layer 0 Activations (after input_layer)             │   │
│ │ [B_block×16, hidden_dim+SKEW] f16                  │   │
│ │                                                   │   │
│ │ ┌─────────────────────────────────────────────┐  │   │
│ │ │ COMPUTATION:                                │  │   │
│ │ │ Load from shmem: act [16,16]               │  │   │
│ │ │ Load from gmem: weights [16,16]            │  │   │
│ │ │ WMMA multiply: result = act @ weights     │  │   │
│ │ │ Activation: relu(result)                   │  │   │
│ │ │ Store to shmem                             │  │   │
│ │ └─────────────────────────────────────────────┘  │   │
│ │                                                   │   │
│ │ Layer 1 Activations (after hidden_layer)        │   │
│ │ [B_block×16, hidden_dim+SKEW] f16               │   │
│ │                                                   │   │
│ │ (repeat for all hidden layers)                  │   │
│ │                                                   │   │
│ │ Layer N Intermediate (post last layer)          │   │
│ │ [B_block×16, output_dim] f16                    │   │
│ └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
              │
              │ (After all layers computed in shared memory)
              ↓
┌─────────────────────────────────────────────────────────────┐
│ Global Memory (Output)                                      │
│  outputs [B, output_dim]      ← Final network output       │
│  forward_buffer [num_layers, B, hidden_dim]  ← For backward│
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Memory Layout: Weight Organization

```
Weights stored LINEARLY in global memory:

offset 0:
    ┌─────────────────────────────────────────────┐
    │ Layer 0 Weights: [hidden_dim, input_dim]    │
    │ Size: hidden_dim × input_dim × 2 bytes      │
    │ Example: 128 × 64 = 8192 f16 values        │
    └─────────────────────────────────────────────┘
     
offset hidden_dim×input_dim:
    ┌─────────────────────────────────────────────┐
    │ Layer 1 Weights: [hidden_dim, hidden_dim]   │
    │ Size: hidden_dim × hidden_dim × 2 bytes     │
    │ Example: 128 × 128 = 16384 f16 values      │
    └─────────────────────────────────────────────┘

    ... (repeat for num_layers-2 hidden layers)

offset ...:
    ┌─────────────────────────────────────────────┐
    │ Layer N Weights: [padded_output_dim, hid]   │
    │ Size: padded_output_dim × hidden_dim × 2    │
    │ Example: 16 × 128 = 2048 f16 values        │
    └─────────────────────────────────────────────┘

CUDA Access Pattern (Linear!):
    offset = layer_0_size + layer_1_size + ... + layer_k_size
    weights_ptr = global_weights + offset
    // Sequential memory access = cache hits!
```

---

## 5. Tensor Core (WMMA) Operation Diagram

```
┌───────────────────────────────────────────────────────────────┐
│ WMMA 16×16×16 Matrix Multiply (Single Operation)              │
│                                                               │
│ Input Fragments (distributed across 32 threads):             │
│                                                               │
│ matrix_a (Activations, row-major):                           │
│   Thread0   Thread1   Thread2   ...  Thread31                │
│    [0,0]     [0,8]    [0,16]         [0,216]                │
│    [1,0]     [1,8]    [1,16]         [1,216]                │
│    ...                                                        │
│   [15,0]    [15,8]   [15,16]        [15,216]                │
│                                                               │
│ matrix_b (Weights, col-major):                               │
│   Thread0   Thread1   Thread2   ...  Thread31                │
│    [0,0]     [8,0]   [16,0]         [216,0]                │
│    [0,1]     [8,1]   [16,1]         [216,1]                │
│    ...                                                        │
│   [0,15]     [8,15]  [16,15]        [216,15]                │
│                                                               │
│                          ╭─────────────────╮                 │
│                          │  TENSOR CORE    │                 │
│         ┌─────────────→  │    (hardware)    │  ←─────────┐   │
│         │                │ 256 mult-add ops │            │   │
│         │                │ in ~100 clocks   │            │   │
│         │                ╰─────────────────╯            │   │
│     matrix_a              Multiple TCs operate in        │   │
│    (16×16)                parallel (more lanes)     matrix_b │
│                                                      (16×16) │
│                                                               │
│ Output (accumulator, distributed):                          │
│   Thread0  [0,0-7]     Thread1  [0,8-15]                    │
│   Thread2  [1,0-7]     Thread3  [1,8-15]                    │
│   ...                                                        │
│   Thread30 [15,0-7]    Thread31 [15,8-15]                   │
│                                                               │
│ Result: 16×16 = 256 elements ← All computed in 100s ns!    │
└───────────────────────────────────────────────────────────────┘
```

---

## 6. Backward Pass Data Flow

```
BACKWARD INPUT:
   grad_output [B, output_dim]  ← From loss.backward()
   forward_buffer [num_layers, B, hidden_dim]  ← Saved from forward
   inputs [B, input_dim]        ← Saved from forward (if calc_grad_inputs)
   weights [num_params]         ← Network parameters
   
             │
             ↓
   ┌────────────────────────────────────────────────────────────┐
   │     CUDA DEVICE: kernel_mlp_fused_backward                 │
   │                                                            │
   │  Reverse iteration through layers:                         │
   │  ├─ Output Activation Backward (done pre-kernel in C++)   │
   │  │  grad_output = dLoss/d(sigmoid_input) * dactivation    │
   │  │                                                         │
   │  ├─ Layer N Backward:                                      │
   │  │  grad_hidden = grad_output @ weights_N^T              │
   │  │  Activation backward (use forward activation values)    │
   │  │                                                         │
   │  ├─ Layer N-1...1 Backward (loop):                         │
   │  │  grad_hidden_prev = grad_hidden @ weights_k^T          │
   │  │  Activation backward                                    │
   │  │                                                         │
   │  └─ Layer 0 Backward (OPTIONAL):                           │
   │     grad_input = grad_hidden @ weights_0^T               │
   │     (No activation backward for input)                     │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
             │
             ├─ All grad_hidden values stored in backward_buffer
             │  for weight gradient accumulation
             │
             ↓
   ┌────────────────────────────────────────────────────────────┐
   │     C++ REDUCTION PHASE (after kernel)                     │
   │                                                            │
   │  ACCUMULATE: grad_weights += grad_hidden ⊗ activations   │
   │  ├─ grad_weights[layer_0] = grad_hidden[0] ⊗ input      │
   │  ├─ grad_weights[layer_1] = grad_hidden[1] ⊗ hidden[0]   │
   │  ├─ ...                                                   │
   │  └─ grad_weights[layer_n] = grad_hidden[n] ⊗ hidden[n-1] │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
             │
             ↓

BACKWARD OUTPUT:
   grad_weights [num_params]  ← Weight gradients (ALWAYS computed)
   grad_inputs [B, input_dim] ← Input gradients (OPTIONAL)
```

---

## 7. Shared Memory Layout Detail

```
Shared Memory Organization (per block):

┌──────────────────────────────────────────────────────────────┐
│ SHARED MEMORY (48-96 KB per block)                           │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Batch chunk activations [N_ITERS*BLOCK_DIM_Z*16, WIDTH] │ │
│ │ float16, with SKEW=8 to avoid bank conflicts            │ │
│ │                                                          │ │
│ │ [16, WIDTH+SKEW] ← Batch sample 0                      │ │
│ │ [16, WIDTH+SKEW] ← Batch sample 1                      │ │
│ │ [16, WIDTH+SKEW] ← Batch sample 2                      │ │
│ │ ...                                                      │ │
│ │ [16, WIDTH+SKEW] ← Batch sample (N_ITERS*BLOCK_DIM_Z-1)│ │
│ │                                                          │ │
│ → All threads in block access different regions           │ │
│ → No warp conflicts, full memory bandwidth                │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Weight matrix (input layer, if input_dim != hidden_dim) │ │
│ │ [hidden_dim, input_dim+INPUT_SKEW]                      │ │
│ │ Loaded once at initialization                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Bank Conflict Prevention (SKEW = 8):

Without SKEW:
  shmem[col, :] = {shmem[col+0*WIDTH], shmem[col+1*WIDTH], ...}
  Example col=0: shmem[0], shmem[WIDTH], shmem[2*WIDTH], ...
  If WIDTH=128, all indices mod 32 (banks) = same bank!
  → 16-way bank conflict

With SKEW=8:
  stride_shmem = WIDTH + SKEW = 136
  shmem[col, :] = {shmem[col+0*136], shmem[col+1*136], ...}
  Example col=0: shmem[0], shmem[136], shmem[272], ...
  123 % 32 = 12-14 (different  banks!)
  → No conflict, full bandwidth
```

---

## 8. Thread Block Organization

```
Each Thread Block:
  
  Threads:
    Grid: 32 threads/warp × (hidden_dim/16) warps/block × {1,2} blocks
    Example: 32 × 8 × 2 = 512 threads/block
  
  Warp Assignment:
    Warp 0 (threads 0-31):    Process hidden_dim/16 = 8 results
    Warp 1 (threads 32-63):   Process next 8 results
    ...
    Total output: 8×16 = 128 rows computed by 1 block
  
  Batch Assignment (threadIdx.z):
    threadIdx.z=0: Batch items [0:16]
    threadIdx.z=1: Batch items [16:32]
    (if BLOCK_DIM_Z=2)

Block Parallelism Across GPU:
  ┌──────────────────────────────────────────┐
  │ Block (0,0)  ║ Block (1,0) ║ Block (2,0) │
  │ Batch[0:X]   ║ Batch[X:2X] ║ Batch[2X:3X]│
  │              ║             ║             │
  └──────────────────────────────────────────┘
  
  Multiple blocks = multiple batch chunks processed in parallel
  Example: 512 blocks × 256 threads/block = 131,072 threads active
  RTX 4060 Ti: ~2048 SMs × 64 threads/SM = 131,072 max occupancy
  → FULL OCCUPANCY!
```

---

## 9. Activation Function Forward/Backward Diagram

```
Forward (Example: ReLU):
  ┌──────────┐
  │  Input   │ [16×16 matrix in WMMA fragment]
  │ x ∈ ℝ    │
  └──────────┘
       │
       │ Element-wise: y = max(0, x)
       ↓
  ┌──────────┐
  │ Output   │ Sparse activation pattern
  │ y ∈ ℝ+   │ Some elements zero (inactive)
  └──────────┘

Backward (ReLU gradient):
  ┌─────────────────────┐
  │ grad_output         │ [16×16 matrix from next layer]
  │ dL/dy ∈ ℝ           │
  └─────────────────────┘
       │
       │ Element-wise: dL/dx = (x > 0) ? dL/dy : 0
       ↓
  ┌─────────────────────┐
  │ grad_input          │ Zeros where x was negative
  │ dL/dx ∈ ℝ≤0 ∪ ℝ+    │ Non-zeros where x was positive
  └─────────────────────┘

More complex: Sigmoid backward
  Forward: y = σ(x) = 1/(1+exp(-x))
  Backward: dL/dx = σ(x)(1-σ(x)) × dL/dy
                  = y(1-y) × dL/dy
                  Needs forward value y to compute gradient!
  → Why we save forward_buffer in context
```

---

## 10. Complete Training Iteration Timeline

```
TIMELINE (ms):

T0:       ffmlp_forward() invoked
          ├─ inputs [1024, 64] prepared
          └─ weights [num_params] ready

T0+0.1:   kernel_mlp_fused <<<512 blocks, 256 threads>>>
          ├─ Blocks launch & occupy SMs
          ├─ Shared memory allocated
          ├─ Layer 0: input → hidden (50 GPU clocks)
          ├─ Layer 1..3: hidden → hidden (each 50 GPU clocks)
          ├─ Layer 4: hidden → output (40 GPU clocks)
          └─ All threads wait at kernel barrier

T0+1.2:   kernel completes
          ├─ outputs [1024, 16] in global memory
          ├─ forward_buffer [4, 1024, 128] in global memory
          └─ Barrier synchronization

T0+1.3:   Loss computed (MSE etc.)
          └─ loss ∈ ℝ scalar

T0+1.5:   loss.backward() invoked
          ├─ grad_output [1024, 16] computed
          └─ Propagates to ffmlp_backward()

T0+1.6:   kernel_mlp_fused_backward <<<512 blocks, 256 threads>>>
          ├─ Backprop layer 4: hidden → grad
          ├─ Backprop layer 3..1: hidden → hidden (reverse)
          ├─ Compute grad_weights via reduction
          └─ Store grad_weights [num_params]

T0+3.0:   kernel_mlp_fused_backward completes
          ├─ grad_weights [num_params] ready in global memory
          └─ backward_buffer [4, 1024, 128] intermediate results

T0+3.1:   optimizer.step() 
          └─ weights -= learning_rate * grad_weights

T0+3.2:   Iteration complete
          Total time: ~3.2 ms

---

10k iterations = 3.2 ms × 10,000 = 32 seconds ≈ 0.5 minutes
For full training (100k iterations): ~5-10 minutes on RTX 4060 Ti
```

---

## 11. Comparison: FFMLP vs Standard PyTorch

```
PyTorch Linear Layers:
  
  forward() [B, input_dim]:
    ├─ layer0 = Linear(input_dim, hidden_dim)
    │  Time: 0.5 ms
    ├─ activation (ReLU)
    │  Time: 0.1 ms
    ├─ layer1 = Linear(hidden_dim, hidden_dim)
    │  Time: 0.7 ms
    ├─ ... (repeat for each hidden layer)
    └─ layer_n = Linear(hidden_dim, output_dim)
       Time: 0.2 ms
    
  TOTAL FORWARD: ~0.5 + 0.1 + 0.7 + ... = 2-4 ms
  Memory overhead: Kernel launch (50 µs each) × 2*num_layers
  Cache misses due to separate kernel calls
  

FFMLP Fused:
  
  kernel_mlp_fused <<<blocks, threads, shmem>>> [B, input_dim]:
    ├─ Layer 0 compute: 0.5 ms (fused with activation)
    ├─ Layer 1 compute: 0.5 ms (fused with activation)
    ├─ ... (all in same kernel)
    └─ Layer n compute: 0.2 ms
    
  TOTAL FORWARD: ~1.5 ms
  Single kernel launch (50 µs)
  Cache hits from shared memory reuse
  

PERFORMANCE RATIO:
  PyTorch: 2-4 ms
  FFMLP:   1.5 ms
  Speedup: 1.3-2.7x faster
  (+ Smaller memory footprint, better scaling)
```

---

This visualization-focused document complements ANALISI_FFMLP_COMPLETA.md by showing architectural patterns and data flows visually.
