# 📚 Analisi Approfondita: Cartella raymarching/

**Scopo**: Intersezione raggio-geometria accelerata su GPU CUDA
**Livelli**: C++ (binding) → CUDA (kernel) → PyTorch Autograd (forward/backward)

---

## 1. Struttura File e Interazione

```
raymarching/
├── __init__.py          ← Export pubblico (entry point)
├── backend.py           ← Compilazione C++/CUDA JIT
├── raymarching.py       ← Wrapper PyTorch + Autograd
├── setup.py             ← Installazione package
└── src/
    ├── raymarching.cu   ← Kernel CUDA (logica GPU)
    ├── bindings.cpp     ← Binding C++/Python
    └── pcg32.h          ← Random generator header
```

### **Flusso di Interazione (Esecuzione)**

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. IMPORT:  user code                                       │
│      ↓                                                        │
│      from raymarching import near_far_from_aabb              │
│      ↓                                                        │
│      __init__.py  (from .raymarching import *)               │
│      ↓                                                        │
│      raymarching.py  (Python wrapper + Autograd)             │
│      ↓                                                        │
│      ← Tenta import: import _raymarching as _backend         │
│      ├─ SE SUCCESS: usa compiled CUDA kernel                 │
│      └─ SE FAIL: da try/except → from .backend import       │
│          ↓                                                    │
│          backend.py  (JIT compile CUDA → _backend)           │
│                                                              │
│  2. COMPILATION (primo run):                                 │
│      backend.py richiama torch.utils.cpp_extension.load()    │
│      ↓                                                        │
│      trova raymarching.cu + bindings.cpp                     │
│      ↓                                                        │
│      Compila con nvcc (CUDA) + cl.exe/gcc (C++)              │
│      ↓                                                        │
│      Crea _raymarching modulo binario (.so/.pyd)             │
│      ↓                                                        │
│      Carica in memoria Python                                │
│                                                              │
│  3. RUNTIME (ogni call):                                     │
│      raymarching.py: _near_far_from_aabb.apply()             │
│      ↓                                                        │
│      Chiama: _backend.near_far_from_aabb(...)                │
│      ↓                                                        │
│      CUDA kernel eseguito su GPU                             │
│      ↓                                                        │
│      Ritorna risultati a Python                              │
│                                                              │
│  4. AUTOGRAD (backward):                                     │
│      PyTorch autograd system                                 │
│      ↓                                                        │
│      custom_fwd / custom_bwd decoratori                      │
│      ↓                                                        │
│      Salva tensori per backward pass                         │
│      ↓                                                        │
│      Chiama _backend CUDA per gradienti                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## a. **setup.py - Quando e come viene richiamato?**

Chiamato una volta da bash

### **Timeline reale:**

```
Setup.py viene richiamato SOLO una volta (installazione)
↓
Compila raymarching.cu + bindings.cpp
↓
Produce binary _raymarching.pyd (Windows) o _raymarching.so (Linux)
↓
Salva su disco (site-packages o ./)
↓
Codice Python lo carica con "import _raymarching"
↓
Da quel punto: usa il binario, NIENTE più ricompilazione
```

**Differenza tra setup.py e backend.py:**
- **setup.py**: Installazione manuale / packaging (python setup.py install)
- **backend.py**: JIT compilation automatica (al primo import)

---

## b. **`import _raymarching` - Cosa è e da dove esce?**

### **Cosa è `_raymarching`:**

```python
_raymarching è un MODULO BINARIO (compiled C++/CUDA)
```

**Non è codice Python**, è un `.pyd` (Windows) / `.so` (Linux) file:
- **Binario compilato** (machine code, non leggibile)
- **Contiene C++ functions** registrate con PYBIND11
- **Caricabile in Python** come un normale modulo

### **Da dove esce:**

```
CASO A: Precompilato (se installato con pip install -e .)
┌─────────────────────────────────────┐
│ setup.py compila una volta          │
│ Produce: _raymarching.pyd           │
│ Salva in: site-packages/raymarching │
│           o ./raymarching           │
└─────────────────────────────────────┘
                ↓
        import _raymarching
                ↓
        Python carica .pyd da disco
        ✓ Veloce (~50ms)

---

CASO B: JIT compilation (se backend.py richiamato)
┌─────────────────────────────────────┐
│ import _raymarching FALLISCE        │
│ (file non trovato)                  │
│           ↓                          │
│ backend.py eseguito                 │
│ torch.cpp_extension.load()          │
│           ↓                          │
│ Compila raymarching.cu + bindings   │
│ (nvcc + cl.exe)                     │
│           ↓                          │
│ Produce: _raymarching in memoria    │
│ Salva cache su disco                │
└─────────────────────────────────────┘
                ↓
        _raymarching pronto
        ✓ Lento primo run (10-15s), veloce dopo (cached)
```

### **Cosa contiene `_raymarching` (i dati):**

```python
# Dentro _raymarching.pyd c'è:

_raymarching.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)
_raymarching.march_rays_train(...)
_raymarching.composite_rays_train_forward(...)
_raymarching.composite_rays_train_backward(...)
_raymarching.march_rays(...)
_raymarching.composite_rays(...)
_raymarching.compact_rays(...)
# ... (tutte le CUDA kernel functions)
```

**Sono le CUDA kernels wrappate da PYBIND11**, callable da Python.

---

## c. **JIT (Just-In-Time Compilation) - Cosa è?**

### **Definizione:**

```
JIT = Compila il codice AL MOMENTO DEL BISOGNO (runtime)
      Non prima dell'esecuzione, ma DURANTE l'esecuzione
```

### **Flusso JIT in raymarching:**

```
┌────────────────────────────────────────────────────┐
│ T0: User code imports raymarching                  │
│                                                    │
│     from raymarching import near_far_from_aabb    │
│     ↓                                              │
│     raymarching.py eseguito                       │
│     ↓                                              │
│     "import _raymarching" ← Try carica precompilato│
│                                                    │
│                             ImportError!          │
│                             (non trovato)          │
│                                                    │
│     ← NIENTE PANICO!                              │
│     "from .backend import _backend"               │
│     ↓ (Fallback attivo - JIT START)               │
│                                                    │
│ T0+100ms: backend.py eseguito                     │
│                                                    │
│     torch.cpp_extension.load()                    │
│     ├─ Vede: raymarching.cu, bindings.cpp        │
│     ├─ Chiama: nvcc (CUDA compiler)              │
│     │   Compila raymarching.cu → binary .obj      │
│     ├─ Chiama: cl.exe (C++ compiler)             │
│     │   Compila bindings.cpp → binary .obj        │
│     ├─ Linker lega tutto                         │
│     └─ Output: _raymarching loaded in memoria    │
│                                                    │
│ T0+15sec: Compilazione finita!                   │
│           _backend = _raymarching                │
│           ✓ Pronto per uso                       │
│                                                    │
│ T0+16sec: User code può usare                     │
│           near_far_from_aabb() ← FUNCIONA!       │
│                                                    │
└────────────────────────────────────────────────────┘

```

---

## 1.5. **Architettura 3-Layer Dettagliata (con Code)**

```
LAYER 1: Python (raymarching.py)
┌─────────────────────────────────────────────┐
│ class _near_far_from_aabb(Function):       │
│     @staticmethod                          │
│     @custom_fwd(cast_inputs=torch.float32) │
│     def forward(ctx, rays_o, rays_d)       │
│         ↓                                   │
│         .contiguous().view(-1, 3)          │
│         ↓                                   │
│         extract GPU pointers               │
│         ↓                                   │
│     _backend.near_far_from_aabb()  ← Chiama layer 2
│                                    │
│                                    ↓
LAYER 2: C++ (bindings.cpp / raymarching.h)
│ void near_far_from_aabb(
│     at::Tensor rays_o,
│     at::Tensor rays_d,
│     ...
│ ) {
│     // Extract pointers
│     auto rays_o_ptr = rays_o.data_ptr<float>();
│     auto rays_d_ptr = rays_d.data_ptr<float>();
│     
│     // Calcola grid/block dimensions
│     int threads = 256;
│     int blocks = (N + 255) / 256;
│     
│     // Launch kernel → Layer 3
│     kernel_near_far_from_aabb<<<blocks, threads>>>(
│         rays_o_ptr, rays_d_ptr, ...
│     );
│ }
│                                    │
│                                    ↓
LAYER 3: CUDA (raymarching.cu)
│ __global__ void kernel_near_far_from_aabb(
│     const float * __restrict__ rays_o,
│     const float * __restrict__ rays_d,
│     ...
│ ) {
│     const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
│     if (n >= N) return;
│     
│     // Slab test per AABB intersection
│     const float ox = rays_o[n*3], oy = rays_o[n*3+1], oz = rays_o[n*3+2];
│     const float dx = rays_d[n*3], dy = rays_d[n*3+1], dz = rays_d[n*3+2];
│     
│     // Calculate near/far for X axis
│     float near = (aabb[0] - ox) / dx;
│     float far = (aabb[3] - ox) / dx;
│     if (near > far) swap(near, far);
│     
│     // ... repeat per Y, Z axis ...
│     
│     nears[n] = near;
│     fars[n] = far;
│     // Ogni thread calcola 1 ray
│ }
│
└─────────────────────────────────────────────┘
```

---

## 2. File Per File: Spiegazione Dettagliata

### **A) `__init__.py` (1 linea)**

```python
from .raymarching import *
```

**Cosa fa**: Export pubblico di ALL symbols da raymarching.py  
**Semantica Python**:
- `.raymarching`: import relativo (from current package)
- `import *`: import tutto da `__all__` (non definito qui, quindi imports tutto pubblico)
- **Effetto**: Quando utente fa `from raymarching import near_far_from_aabb`, trova la funzione qui

---

### **B) `backend.py` (JIT Compilation via Torch)**

```python
import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
```

**Riga 1-3**:
- `import os`: Per path manipulation
- `load`: Funzione torch che compila C++/CUDA on-the-fly
- `_src_path`: Trova directory corrente (dove è backend.py)

```python
nvcc_flags = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', ...
]
```

**Che cosa**: Flag per compilatore CUDA `nvcc`
- **`-O3`**: Massima ottimizzazione
- **`-std=c++14`**: Standard C++ 2014
- **`-U...`**: Undefine macro per supportare float16 operations

```python
if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']
```

**Cosa fa**: Flag per C++ compiler dipende da OS
- **`posix"** = Linux/Mac → usa gcc/clang
- **`"nt"** = Windows → usa Microsoft cl.exe (Visual Studio)
- **Flag differenti**: `/O2` (MSVC) vs `-O3` (GCC)

```python
elif os.name == "nt":
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(...), reverse=True)
            if paths:
                return paths[0]
    
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError(...)
        os.environ["PATH"] += ";" + cl_path
```

**Cosa fa**: Trova compilatore MSVC su Windows
- **`where cl.exe`**: Check se cl.exe è in PATH
- Se non trovato (`!= 0`), cerca manualmente in Visual Studio folders
- **Logica**: Prova 4 edizioni di VS (Enterprise → Community)
- **`glob.glob()`**: Regex path matching
- **`reverse=True`**: Preferisci versione più recente
- Se ancora non trovato: `RuntimeError` (exit)
- Altrimenti: aggiungi al PATH ambiente

**Perché necessario**: Alcuni sistemi non hanno cl.exe automatico in PATH anche se VS è installato.

```python
_backend = load(name='_raymarching',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'raymarching.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']
```

**Cosa fa**: JIT compilation
- **`name='_raymarching'`**: Nome modulo binario (import con `import _raymarching`)
- **`extra_cflags`**: Flag C++ runtime binding
- **`extra_cuda_cflags`**: Flag CUDA kernel
- **`sources`**: File da compilare (raymarching.cu + bindings.cpp)
- **Output**: `_backend` = compiled binary module
- **`__all__`**: Esporta solo `_backend`

**Nota**: Se compilation fallisce (es. CUDA non trovato), genera errore. Questo perché try/except in raymarching.py cattura e fa fallback.

---

### **C) `raymarching.py` (Wrapper PyTorch)**

#### **Parte 1: Import e Fallback**

```python
try:
    import _raymarching as _backend
except ImportError:
    from .backend import _backend
```

**Semantica**:
- **Try**: Importa modulo precompilato `_raymarching` (se setup.py già eseguito)
- **Except**: Se non trovato, esegui compilazione JIT da backend.py

**Perché**: Permette di usare sia precompilato (fast) che JIT (flexible)

#### **Parte 2: Utility Functions (non su GPU, semplici)**

Esempio: `near_far_from_aabb`

```python
class _near_far_from_aabb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        '''
        
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        
        N = rays_o.shape[0] # num rays
        
        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        
        _backend.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)
        
        return nears, fars

near_far_from_aabb = _near_far_from_aabb.apply
```

**Analisi Linea per Linea**:

1. **`class _near_far_from_aabb(Function):`**
   - **Classe con `_` prefix**: Convenzione Python per "private" (interno, non export diretto)
   - **Eredita da `Function`**: Classe PyTorch base per custom autograd operations
   - **Why Function?**: Ogni operazione CUDA deve definire forward + backward separatamente

2. **`@staticmethod`**
   - **Semantica**: Metodo senza accesso a `self` (classe stessa))
   - **When/Why**: Quando funzione non usa stato istanza, è più efficiente e chiaro

3. **`@custom_fwd(cast_inputs=torch.float32)`**
   - **Che cosa**: Decorator PyTorch per forward pass
   - **`cast_inputs=torch.float32`**: Automaticamente converte input a float32
   - **Perché**: Computazione stability (float16 può avere NaN, float32è safe). CUDA kernel è scritto per float32
   - **Quando usare**: Quando usi fp16 training (mixed precision) ma kernel è fp32

4. **`def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):`**
   - **`ctx`**: Autograd context object (salva dati per backward), **SEMPRE primo argomento**
   - **`rays_o`**: Ray origins, [N, 3] (N rays, 3D coordinates)
   - **`rays_d`**: Ray directions, [N, 3]
   - **`aabb`**: AABB bounds, [6] = (xmin, ymin, zmin, xmax, ymax, zmax)
   - **`min_near`**: Minimum near value (avoid numerical issues)

5. **`if not rays_o.is_cuda: rays_o = rays_o.cuda()`**
   - **Logica**: Se tensor è su CPU, muovilo a GPU
   - **Perché**: CUDA kernel gira su GPU, input deve essere GPU memory

6. **`rays_o = rays_o.contiguous().view(-1, 3)`**
   - **`contiguous()`**: Assicura layout memoria è C-contiguous (row-major). CUDA kernel assume questo.
   - **`view(-1, 3)`**: Reshape a [N, 3] se non è già. `-1` = deduce dimensione
   - **Perché**: CUDA code fa accesso memory lineare, non sparse

7. **`N = rays_o.shape[0]`**
   - Numero di rays = prima dimensione
   - Usato per allocare output tensori

8. **`nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)`**
   - **`torch.empty()`**: Alloca memoria non inizializzata (veloce)
   - **`dtype=rays_o.dtype`**: Mantieni precision (float32 or float64)
   - **`device=rays_o.device`**: Mantieni device (GPU/CPU)
   - **Perché `empty` e non `zeros`?**: CUDA kernel scriverà comunque i dati, non necessario inizializzare

9. **`fars = torch.empty(...)`**
   - Stesso buffer per farness values

10. **`_backend.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)`**
    - **Chiama CUDA kernel** (compilato in raymarching.cu)
    - **Input**: rays_o, rays_d, aabb, N, min_near (geometria)
    - **Output**: nears, fars (riscrivibili, passed by reference come C++)
    - **GPU execution**: Qui gira il vero calcolo su GPU

11. **`return nears, fars`**
    - Ritorna i due output tensori

12. **`near_far_from_aabb = _near_far_from_aabb.apply`**
    - **Semantica Python**: Assegna `.apply` metodo a funzione più semplice
    - **Perché**: `.apply` è il "entry point" di autograd (chiama forward automaticamente)
    - **Effetto**: Utente chiama `near_far_from_aabb(rays_o, rays_d, aabb)` che internamente chiama `_near_far_from_aabb.forward(ctx, ...)`

---

#### **Parte 3: Training Functions (con backward pass)**

Esempio: `_composite_rays_train`

```python
class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, rays):
        ''' composite rays' rgbs, according to the NeRF formula '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        
        M = sigmas.shape[0]
        N = rays.shape[0]
        
        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)
        
        _backend.composite_rays_train_forward(sigmas, rgbs, deltas, rays, M, N, weights_sum, depth, image)
        
        ctx.save_for_backward(sigmas, rgbs, deltas, rays, weights_sum, depth, image)
        ctx.dims = [M, N]
        
        return weights_sum, depth, image
```

**Differenze rispetto `_near_far_from_aabb`**:

1. **`ctx.save_for_backward(...)`**
   - **Cosa fa**: Salva tensori da usare nel backward pass
   - **Perché**: Backward deve ricalcolare gradienti, serve ricordare forward values
   - **Limitazione**: Salva reference, **non copia** (memoria efficiente)

2. **`ctx.dims = [M, N]`**
   - **Cosa fa**: Salva dimensioni come attributo (non tensor Python)
   - **Differenza**: `save_for_backward` è per tensori/autograd, `ctx.dims` è per metadata
   - **Perché servono**: Backward ha size N, M
   
```python
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_depth, grad_image):
        '''
        Backward pass: calcola gradienti rispetto input (sigmas, rgbs)
        '''
        
        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()
        
        sigmas, rgbs, deltas, rays, weights_sum, depth, image = ctx.saved_tensors
        M, N = ctx.dims
        
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        
        _backend.composite_rays_train_backward(grad_weights_sum, grad_image, sigmas, rgbs, deltas, rays, weights_sum, image, M, N, grad_sigmas, grad_rgbs)
        
        return grad_sigmas, grad_rgbs, None, None
```

**Analisi**:

1. **`@custom_bwd`**
   - Decorator per backward pass (analogo a custom_fwd)
   - Specifica come calcolare gradienti

2. **`def backward(ctx, grad_weights_sum, grad_depth, grad_image):`**
   - **`grad_*`**: Gradienti della loss rispetto output
   - **Numero argomenti**: Deve corrispondere a numero output da forward (3: weights_sum, depth, image)
   - **Logica**: Calcola gradienti rispetto input usando catena di derivate

3. **`sigmas, rgbs, deltas, rays, weights_sum, depth, image = ctx.saved_tensors`**
   - Recupera tensori salvati in forward

4. **`M, N = ctx.dims`**
   - Recupera dimensioni

5. **`grad_sigmas = torch.zeros_like(sigmas)`**
   - Alloca output gradient tensor (inizializzato a 0)
   - **`zeros_like`**: Stessa shape e dtype di sigmas, ma su GPU, tutti 0

6. **`_backend.composite_rays_train_backward(...)`**
   - Chiama CUDA kernel per calcolo gradienti
   - **Input**: grad_weights_sum, grad_image (chain rule), sigmas, rgbs, ... (forward values)
   - **Output**: grad_sigmas, grad_rgbs (gradient tensori da popolare)

7. **`return grad_sigmas, grad_rgbs, None, None`**
   - **4 return values**: Uno per ogni input a forward (sigmas, rgbs, deltas, rays)
   - **`None`**: Per deltas e rays (non richiedono gradiente, no backprop)
   - **Perché None?**: deltas è calcolato da ray marching (geometric, non learnable), rays è indice (discrete, no gradient)

---

### **D) `setup.py` (Installazione Package)**

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```

**Cosa fa**: Configurazione installazione package
- **`setup()`**: Función setuptools per packaging Python
- **`BuildExtension`**: Custom build command per estensioni C++
- **`CUDAExtension`**: Specifica compilazione CUDA

```python
setup(
    name='raymarching',
    ext_modules=[
        CUDAExtension(
            name='_raymarching',
            sources=[...],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**Come usarlo**:
```bash
pip install -e .                 # Installa in development mode (JIT)
python setup.py build_ext --inplace  # Compila locally
```
---

## 3. Logica di Esecuzione Passo-Passo

### **Scenario: User chiama `near_far_from_aabb`**

```python
from raymarching import near_far_from_aabb

rays_o = torch.randn(1024, 3).cuda()  # [1024, 3]
rays_d = torch.randn(1024, 3).cuda()  # [1024, 3]
aabb = torch.tensor([...]).cuda()      # [6]

nears, fars = near_far_from_aabb(rays_o, rays_d, aabb)  # Output: [1024] each
```

**Cosa succede**:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. IMPORT RESOLUTION                                            │
│    from raymarching import near_far_from_aabb                  │
│    ↓                                                            │
│    __init__.py: from .raymarching import *                     │
│    ↓                                                            │
│    raymarching.py: cerca import _raymarching                   │
│    ├─ Success? → Usa precompiled binary                        │
│    └─ Fail? → Executa backend.py → JIT compile               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. FUNCTION RESOLUTION                                          │
│    near_far_from_aabb = _near_far_from_aabb.apply             │
│    ↓                                                            │
│    Quando user chiama: near_far_from_aabb(rays_o, rays_d, ...)│
│    ↓                                                            │
│    In realtà chiama: _near_far_from_aabb.apply(rays_o, ...)   │
│    ↓                                                            │
│    PyTorch autograd chain: .apply() → forward() della classe  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. FORWARD EXECUTION                                            │
│    _near_far_from_aabb.forward(ctx, rays_o, rays_d, aabb)     │
│    ↓                                                            │
│    [1] Type check: is_cuda → move to GPU if needed            │
│    [2] Contiguity: .contiguous() → C-layout memory            │
│    [3] Reshape: .view(-1, 3) → ensure [N, 3]                 │
│    [4] Allocate: torch.empty() → output buffers                │
│    [5] CUDA launch: _backend.near_far_from_aabb(...)          │
│        ↓                                                        │
│        C++ binding code launces CUDA kernel                    │
│        ↓                                                        │
│        CUDA grid: 1024 threads → 1024 rays processed in //     │
│        ↓                                                        │
│        nears, fars buffers populated on GPU                    │
│    [6] Return tensors to Python                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. AUTOGRAD GRAPH (for training later)                         │
│    PyTorch builds computation graph:                            │
│    │                                                            │
│    near_far_from_aabb (forward function)                       │
│    ├─ inputs: rays_o [1024, 3], rays_d [1024, 3], aabb [6]   │
│    ├─ outputs: nears [1024], fars [1024]                     │
│    └─ gradients tracked (even though near_far doesn't use in  │
│       this case, but stored for potential backward)            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 5. LATER: BACKWARD (if loss depends on outputs)               │
│    (Usually near_far doesn't need backward, used only forward) │
│    But if it did lose.backward() would:                        │
│    ↓                                                            │
│    Backward graph traversal                                    │
│    ↓                                                            │
│    _composite_rays_train.backward() called (example with bwd)  │
│    ↓                                                            │
│    Retrieve saved tensors from ctx                             │
│    ↓                                                            │
│    _backend.composite_rays_train_backward() CUDA kernel        │
│    ↓                                                            │
│    Gradient tensors populated                                  │
│    ↓                                                            │
│    Return to optimizer                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Convenzioni di Nomenclatura

### **Prefisso `_` (Underscore)**

| Pattern | Significato | Esempio |
|---------|------------|---------|
| `_backend` | Private module (interno, usa via `apply`) | `import _raymarching as _backend` |
| `_ClassName` | Private class (usa via `.apply`) | `class _near_far_from_aabb(Function)` |
| `_src_path` | Private variable (locale a modulo) | `_src_path = os.path.dirname(...)` |

**Semantica**: Non per enforcement (Python non ha vero private), ma per **convenzione**: "questo è implementation detail, non parte del API pubblico"

### **Funzione vs Classe**

```python
# PATTERN:
class _OperationName(Function):  # ← Classe privata
    @staticmethod
    def forward(ctx, ...):
        ...
        _backend.cuda_kernel(...)  # ← Chiama C++/CUDA
        return outputs
    
    @staticmethod
    def backward(ctx, grad_outputs):
        ...
        return grad_inputs

OperationName = _OperationName.apply  # ← Export come funzione
```

**Perché questo pattern?**
- **Classe**: Necessaria per PyTorch `Function` (forward + backward separati)
- **`.apply`**: Semplifica API (user vede funzione, non classe)
- **`_` prefix**: Segnala non usare direttamente

### **Backend vs Frontend**

| Layer | File | Linguaggio | Cosa Fa |
|-------|------|-----------|---------|
| **Frontend** | raymarching.py | Python | API pubblica, Autograd bridge |
| **Backend** | backend.py | Python+C++ | Compilazione JIT |
| **Backend** | src/raymarching.cu | CUDA | Computation kernel |
| **Backend** | src/bindings.cpp | C++ | Bridge PyTorch ↔ CUDA |

---

## 5. Decoratori Spiegati

### **`@staticmethod`**

```python
class Example:
    def instance_method(self):      # Accede a self
        return self.value
    
    @staticmethod
    def static_method():             # Non accede a self
        return "constant"

# Uso:
obj = Example()
obj.instance_method()      # Necessita istanza
Example.static_method()    # Non necessita istanza
```

**Perché usato in raymarching.py?**
- Non serve accesso a stato istanza (self non usato)
- Sottolinea che metodo è "stateless function"
- Piccolo overhead di velocità (no self binding)

### **`@custom_fwd`**

```python
@custom_fwd(cast_inputs=torch.float32)
def forward(ctx, input_fp16):
    # Input viene castovato a float32
    # Computation fatto in float32
    # Mantiene stabilità numerica
    pass
```

**Scenario**: Mixed precision training (fp16)
- **Problema**: f16 ha range ristretto, sigma/exponential può divergere
- **Soluzione**: Cast a f32 per computation, mantieni f16 per memoria
- **Decoratore automatizza**: Cast prima di forward, recast a f16 dopo

### **`@custom_bwd`**

```python
@custom_bwd
def backward(ctx, grad_output):
    # Backward computation
    # Usa float precision specificata
    return grad_inputs
```

**Analogo a custom_fwd**: Configura precision per backward pass

---

## 6. Memory Layout e Contiguity

### **Pattern ricorrente**:

```python
input_tensor = input_tensor.contiguous().view(-1, 3)
```

**Perché due operazioni?**

1. **`.contiguous()`**
   - Certifica memoria layout C (row-major)
   - CUDA code assume stride lineare
   - Se tensor è già C-contiguous: noop (zero cost)
   - Se non contiguous: copia memory

2. **`.view(-1, 3)`**
   - Reshape logico (no copy se memoria permette)
   - `-1`: deduce dimension (N × 3 = total size)
   - Prepara shape attesa da CUDA kernel

**Perché necessario?**
```
Example: transpose può creare tensor non-contiguous
tensor = torch.randn(3, 1024)
tensor_t = tensor.T                    # [1024, 3] but not contiguous!
tensor_t[0, 0]  # Accesso è scatter, non lineare

CUDA kernel accede linearly, si aspetta stride=1 per inner dimension
Quindi: tensor_t.contiguous() "riorganizza" memoria
```

---

## 7. Context Object (`ctx`)**

**Ruolo di `ctx` in Autograd**:

```python
class MyFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = expensive_computation(input)
        
        # Salva per backward
        ctx.save_for_backward(input, output)
        ctx.some_constant = 42  # Salva metadata
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        constant = ctx.some_constant
        
        grad_input = compute_gradient(grad_output, input, output)
        return grad_input
```

**`ctx` è ponte tra forward/backward**:
- **Forward**: Save info per backward (`.save_for_backward()`)
- **Backward**: Retrieve info (`.saved_tensors`)
- **Metadata**: Non-tensor data (`.some_attr = value`)

**Memory efficiency**:
- `save_for_backward`: Reference (no copy)
- `ctx.attr`: Object reference (truly cheap)

---

## 8. Flusso Training vs Inference

### **Training: `_march_rays_train` + `_composite_rays_train`**

```python
# Forward
xyzs, dirs, deltas, rays = march_rays_train(...)
sigmas = density_net(xyzs)
rgbs = color_net(xyzs, dirs)

# Composite (backward enabled)
weights_sum, depth, image = composite_rays_train(sigmas, rgbs, deltas, rays)

# Loss
loss = criterion(image, gt_image)
loss.backward()  # ← Backprop attraverso composite_rays_train.backward()
optimizer.step()
```

**Flusso autograd**:
```
image ← composite_rays_train (saved tensors for backward)
             ↑
         sigmas, rgbs (require_grad=True)
             ↑
    density_net, color_net (nn.Module con .weight.grad)
             ↑
        Loss.backward() traverses graph
```

### **Inference: `_march_rays` + `_composite_rays` (no backward)**

```python
# No backward needed
with torch.no_grad():
    xyzs, dirs, deltas = march_rays(...)  # No ctx
    sigmas = density_net(xyzs)
    rgbs = color_net(xyzs, dirs)
    composite_rays(sigmas, rgbs, deltas, ..., weights_sum, depth, image)  # In-place
    # image already filled on GPU
```

**Differenza**:
- **`composite_rays_train`**: Return values (for loss computation)
- **`composite_rays`**: Void (modifies in-place), no autograd needed

---

## 9. Ottimizzazioni e Trade-off

### **Perché no Backward per `near_far_from_aabb`?**

```python
class _near_far_from_aabb(Function):
    def forward(...):
        ...
    
    # ← No backward defined!
```

**Ragione**: Calcola AABB intersection (geometric), non trainable
- Input (ray origine/direzione) non sono learnable
- Output (near/far) sono usati per sampling, non per loss
- Backward sarebbe inutile overhead

### **Perché CUDA per Composite?**

```
Naive Python:
  for i in range(M):
      alpha = 1 - exp(-sigma[i] * delta[i])
      weight = transmittance * alpha
      ...
  # M up to 2 milioni: 2M float operations
  # Time: ~1 second su CPU

CUDA Parallel:
  1024 threads in parallel
  Each thread handles 1-2000 points
  Time: 1-5 ms su GPU
  Speedup: 200-1000x
```

---

## 10. Recap: Come Tutto Funziona Insieme

```
USER CODE:
  from raymarching import near_far_from_aabb
  nears, fars = near_far_from_aabb(rays_o, rays_d, aabb)
  
  ↓ (Imports cascata)
  
  __init__.py → raymarching.py → backend.py → torch.cpp_extension.load()
  
  ↓ (First import: Compilation)
  
  nvcc + cl.exe compila:
    src/raymarching.cu (CUDA kernels)
    src/bindings.cpp (C++ ↔ Python bridge)
    → _raymarching.pyd (Windows binary module)
  
  ↓ (Carica modulo)
  
  import _raymarching as _backend (binary in memory)
  
  ↓ (Setup Autograd)
  
  _near_far_from_aabb class wraps CUDA function
  near_far_from_aabb = _near_far_from_aabb.apply (user-facing API)
  
  ↓ (User call)
  
  near_far_from_aabb(rays_o, rays_d, aabb)
  → __call__ → .apply() → forward() → _backend.near_far_from_aabb() → CUDA kernel
  
  ↓ (GPU Execution)
  
  CUDA grid: 4 blocks × 256 threads = 1024 threads total
  Each thread: ray-AABB slab test (3 iterations, 6 comparisons)
  Result: nears, fars buffers filled on GPU memory
  
  ↓ (Return to Python)
  
  PyTorch tensor [1024] returned (GPU memory pointer)
  Can be used in next operations without copying
```

---

**Conclusione**: Raymarching è un **3-layer stack**:
1. **Python layer**: User API + Autograd bridge
2. **C++ layer**: Type marshalling + grid computation
3. **CUDA layer**: Parallel computation kernel

Ogni layer ottimizzato per suo ruolo: Python for flexibility, C++ for interfacing, CUDA for speed.

---

# 📘 PARTE 2: CODICE SORGENTE - ANALISI DETTAGLIATA

## 1. `raymarching.h` - Header File (Function Signatures)

**Purpose**: _Dichiarazioni C++ di tutte le funzioni che verranno implementate in raymarching.cu e bindings.cpp_

```cpp
#pragma once
#include <stdint.h>
#include <torch/torch.h>

// Dichiarazioni delle funzioni (no implementation qui)
void near_far_from_aabb(at::Tensor rays_o, at::Tensor rays_d, at::Tensor aabb, 
                        const uint32_t N, const float min_near, 
                        at::Tensor nears, at::Tensor fars);
```

**Cosa significa ogni parte**:

`#pragma once`  
Serve a evitare il "loop delle inclusioni".   Immagina questa situazione:  
Hai un file Punto3D.h (definisce cos'è un punto).
Hai un file Forma.h che include Punto3D.h (perché una forma è fatta di punti).  
Nel tuo main.cpp, tu includi sia Punto3D.h che Forma.h.  
Senza #pragma once succederebbe questo:  
Il compilatore apre main.cpp.  
Legge Punto3D.h   
_"Ok, so cos'è un Punto3D"._  
Legge Forma.h, il quale gli dice di leggere di nuovo Punto3D.h.  
Rilegge Punto3D.h   
_ERRORE! Il compilatore urla: "Mi hai già detto cos'è un Punto3D! Non posso definirlo due volte!"._
Con #pragma once:
Il compilatore vede il comando in cima a Punto3D.h. La seconda volta che qualcuno prova a includerlo, il compilatore dice: "_Questo l'ho già letto, lo salto"_. Il codice compila correttamente senza errori di duplicazione.

| Elemento | Significato |
|----------|------------|
| `void` | Return type: niente ritornato (modifica argomenti in-place) |
| `at::Tensor` | Type PyTorch tensor (ATen library) |
| `const uint32_t N` | Numero di rays (constant, on-device) |
| `const float min_near` | Valore minimo per "near" (clamp numeico) |
| `rays_o, rays_d, ...` | Tensori di input/output (GPU memory) |

**Perché `at::Tensor` e non `float*`?**
- `at::Tensor`: Type-safe, checks device/dtype automaticamente
- `float*`: Raw pointer, pericoloso, può essere CPU o GPU memory

**Tutte le funzioni in raymarching.h**:

```
Utility functions:
├── near_far_from_aabb()     ← Ray-AABB intersection
├── polar_from_ray()         ← Ray to sphere coordinates
├── morton3D()               ← 3D spatial hash (index computation)
├── morton3D_invert()        ← Reverse spatial hash
└── packbits()               ← Grid to bitfield compression

Training functions:
├── march_rays_train()       ← Ray marching per training (variable steps)
├── composite_rays_train_forward()   ← Volume rendering forward
└── composite_rays_train_backward()  ← Gradient computation

Inference functions:
├── march_rays()             ← Ray marching per inference (fixed steps)
├── composite_rays()         ← Volume rendering inference
└── compact_rays()           ← Filter live rays (early termination)
```

---

## 2. `raymarching.cu` - CUDA Kernel Implementations

### **2.1 Parte 1: Include e Macro**

```cpp
#include <cuda.h>
#include <cuda_fp16.h>           // Half precision (float16) support
#include <cuda_runtime.h>        // CUDA runtime API
#include <ATen/cuda/CUDAContext.h>  // PyTorch CUDA context
#include <torch/torch.h>         // PyTorch types
#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>                // std::numeric_limits
#include "pcg32.h"               // Random number generator
```

**Cosa fa ogni header**:
- **`<cuda.h>`**: CUDA driver API
- **`<cuda_fp16.h>`**: Float16 operazioni (required per mixed precision)
- **`<torch/torch.h>`**: PyTorch C++ API (tensors, types)
- **`"pcg32.h"`**: PCG32 PRNG (permuted congruential generator) per random values

### **Macro di Debug/Check**:

```cpp
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int)
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || ...)
```

**Significato**:
- `#define` crea una macro. È fondamentalmente un comando di "cerca e sostituisci" che il compilatore esegue prima di leggere effettivamente il codice
- `TORCH_CHECK`: Se condizione falsa, lancia eccezione Python
- `#x`: Stringifica il nome della variabile (debug info)
- **Uso**: Validazione input prima di lancio kernel

### **2.2 Parte 2: Inline Device Functions (Helper)**

```cpp
inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }
```

**Cosa fa**:
- Definisce costanti matematiche come `__device__` functions
- `inline` il compilatore elimina il salto/chiamate alla funzione. Se usi SQRT3() in dieci punti diversi del tuo codice, il compilatore sostituisce fisicamente la chiamata con il valore `1.732...` in ognuno di quei dieci punti.
- `constexpr`: Compilatore computa al compile-time, _"Il valore di out di questa funzione non cambierà mai"_
- **Perché functions e non const?** Per garantire inline (no function call overhead)
- **Uso**: `SQRT3()` nel codice kernel

**Altre inline function helper**:

```cpp
template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}
// Uso: div_round_up(1024, 256) = 4 blocks
// Equivalente a: ceil(1024 / 256)

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}
// Clamp x tra min/max usando operazioni GPU-native

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent);  // Estrae esponente IEEE float
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}
// Calcola MIP level da coordinate 3D
// MIP = Mipmap level per hierarchical grid sampling
```

### **2.3 Morton Encoding (Spatial Hash)**
Morton Encoding (noto anche come Z-order curve). Il suo obiettivo è prendere tre coordinate (x, y, z) e "intrecciare" i loro bit per ottenere un unico numero.  
Il vantaggio? Punti vicini nello spazio 3D finiscono per avere indici 1D vicini, il che rende la memoria della GPU velocissima da leggere (cache locality **COALESCING** ).  
Queste sono `0x00010001u` costanti esadecimali (la u finale sta per unsigned), sono stati scelti con precisione matematica per la loro rappresentazione binaria.

```cpp
inline __host__ __device__ uint32_t __expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;  // Spread bits
    v = (v * 0x00000101u) & 0x0F00F00Fu;  // More spread
    v = (v * 0x00000011u) & 0xC30C30C3u;  // Even more
    v = (v * 0x00000005u) & 0x49249249u;  // Final spread
    return v;
}
// Espande 10-bit integer in 30-bit con spazi vuoti

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = __expand_bits(x);        // x bits: 0_0_0_0_...
    uint32_t yy = __expand_bits(y);        // y bits: shifted by 1
    uint32_t zz = __expand_bits(z);        // z bits: shifted by 2
    return xx | (yy << 1) | (zz << 2);     // Interleave: z y x z y x ...
}
// Morton code: Encodes 3D coordinates (x,y,z) in 1D index
// Proprietà: Punti vicini nello spazio → Indici vicini in 1D
// Utile per cache locality!
```

**Esempio Morton**:
```
x = 0b0101 (5)         xx = 0b01_00_01 (dopo __expand_bits)
y = 0b0011 (3)    →    yy = 0b00_01_01 << 1 = 0b00_10_10
z = 0b0110 (6)         zz = 0b01_00_10 << 2 = 0b0100_00_1000

morton3D = 0b0101_0011_0110 = richiede interleaving per spatial locality
```

### **2.4 Kernel Principale: `kernel_near_far_from_aabb`**

```cpp
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,     // [N, 3] origin
    const scalar_t * __restrict__ rays_d,     // [N, 3] direction
    const scalar_t * __restrict__ aabb,       // [6] bound: (xmin, ymin, zmin, xmax, ymax, zmax)
    const uint32_t N,                         // Numero di rays
    const float min_near,                     // Clamp minimo
    scalar_t * nears, scalar_t * fars        // Output [N] each
) {
    // STEP 1: Thread assignment
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;  // Se thread ID >= N, non fare niente
    
    // Ogni thread elabora 1 ray
    
    // STEP 2: Load ray origin/direction
    rays_o += n * 3;        // Punta al raggio n
    rays_d += n * 3;        // rays_o[0], rays_o[1], rays_o[2]
    
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];  // Origin
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];  // Direction
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;        // Reciprocals (avoid div in loop)
    
    // STEP 3: Slab test per asse X
    // Ray equation: P(t) = O + t*D
    // AABB bounds: [xmin, ...] to [..., xmax]
    // Intersection X-slab: xmin ≤ ox + t*dx ≤ xmax
    //                   → t1 = (xmin - ox) / dx
    //                   → t2 = (xmax - ox) / dx
    
    float near = (aabb[0] - ox) * rdx;  // t para xmin
    float far = (aabb[3] - ox) * rdx;   // t para xmax
    if (near > far) swapf(near, far);   // Assicura near < far
    
    // STEP 4-5: Slab test Y e Z (stessa logica)
    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);
    
    // Intersection = intersezione degli intervalli
    if (near > far_y || near_y > far) {  // No intersection nel piano XY
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }
    
    if (near_y > near) near = near_y;  // Expand near
    if (far_y < far) far = far_y;      // Shrink far
    
    // Z-slab test
    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);
    
    if (near > far_z || near_z > far) {  // No intersection
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }
    
    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;
    
    // STEP 6: Clamp minimo
    if (near < min_near) near = min_near;  // Non troppo vicino
    
    // STEP 7: Salva risultati
    nears[n] = near;
    fars[n] = far;
}
```

**Cosa fa** (passo-passo):
- **Init**: Ogni thread carica 1 raggio (origine + direzione)
- **Calcolo**: Intersezione raggio-AABB usando slab test
- **Slab test**: Per ogni asse (X, Y, Z), calcola intervallo di t
- **Intersezione**: Combina i 3 intervalli per trovare near/far finale
- **Output**: Salva near/far per compositing volometrico successivo

**Timing**: ~0.1 ms per 1024 rays su GPU

---

### **2.5 Kernel: `kernel_march_rays_train`**

Questo è il kernel più complesso (300+ linee).  
Non fa training nel senso che aggiorna parametri, salva solo metadata per futuro training.    
Fa:
1. **Stima steps**: Conta quanti sample points servono
2. **Marching adaptive**: Aumenta step dove densità alta, salta dove bassa
3. **Grid hierarchico**: MIP pyramid per velocità

```cpp
__global__ void kernel_march_rays_train(...) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;  // 1 thread = 1 ray
    
    // Load rayparameters
    const float ox = rays_o[n*3], oy = rays_o[n*3+1], oz = rays_o[n*3+2];
    const float dx = rays_d[n*3], dy = rays_d[n*3+1], dz = rays_d[n*3+2];
    
    // Temporal bounds [near, far]
    const float near = nears[n];
    const float far = fars[n];
    
    // Adaptive step sizes
    const float dt_min = 2 * SQRT3() / max_steps;  // Minimum step (grid diagonal / max_steps)
    const float dt_max = 2 * SQRT3() * (1 << (C-1)) / H;  // Maximum (highest cascade / H)
    
    // PASS 1: Count steps (binary search to find exact num_steps needed)
    float t = near;
    if (perturb) {
        pcg32 rng((uint64_t)n);
        t += dt_min * rng.next_float();  // Random jitter per anti-aliasing
    }
    
    uint32_t num_steps = 0;
    while (t < far && num_steps < max_steps) {
        float x = clamp(ox + t*dx, -bound, bound);
        float y = clamp(oy + t*dy, -bound, bound);
        float z = clamp(oz + t*dz, -bound, bound);
        
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);  // Variable step size
        
        // MIP level calculation
        int level = max(mip_from_pos(x,y,z,C), mip_from_dt(dt,H,C));  // [0, C-1]
        // MIP: More detailed grid (smaller cells) where ray is moving slowly
        
        // Grid position normalization
        float mip_bound = 1 << level;  // 2^level
        int nx = clamp(0.5 * (x / mip_bound + 1) * H, 0, H-1);  // Normalize to [0, H]
        int ny = clamp(0.5 * (y / mip_bound + 1) * H, 0, H-1);
        int nz = clamp(0.5 * (z / mip_bound + 1) * H, 0, H-1);
        
        // Query occupancy grid (bitfield packed)
        uint32_t index = level * H*H*H + morton3D(nx, ny, nz);
        bool occ = grid[index/8] & (1 << (index%8));  // 1 bit per voxel!
        
        if (occ) {
            num_steps++;  // Occupied: increment step counter
            t += dt;      // Small step
        } else {
            // Skip to next voxel boundary
            // Calculate which voxel boundary we'll hit first
            float tx = ... (complex formula);  // t para next voxel X
            float ty = ...;
            float tz = ...;
            t += max(0, min(tx, min(ty, tz)));  // Jump to boundary
        }
    }
    
    // PASS 2: Actually store xyz positions and directions
    // atomicAdd ensures no race conditions
    uint32_t point_index = atomicAdd(counter, num_steps);     // Get starting offset
    uint32_t ray_index = atomicAdd(counter+1, 1);             // Get ray batch index
    
    // Store ray metadata
    rays[ray_index * 3] = n;           // Ray ID
    rays[ray_index * 3 + 1] = point_index;   // Starting point offset
    rays[ray_index * 3 + 2] = num_steps;     // Numero di sample points
    
    // ... repeat loop from PASS 1, but now write xyz/dirs/deltas ...
    // (Same marching logic, but populates output arrays)
}
```

**Concetti chiave**:

| Concetto | Spiegazione |
|----------|------------|
| **MIP pyramid** | Griglia gerarchica: level 0 = alto dettaglio, level C-1 = basso |
| **Adaptive stepping** | `dt = t * dt_gamma` cresce con profondità (veloce lontano, lento vicino) |
| **Bitfield grid** | 1 bit per voxel = 8x compressione di memoria (grid occupancy) |
| **atomicAdd** | Coordinate accumulo senza race conditions (CUDA atomic op) |
| **Morton encoding** | Spatial hash per cache locality |
| **Perturb** | Random jitter per anti-aliasing in training |

---

### **2.6 Kernel: `kernel_composite_rays_train_forward`**

Volume rendering forward pass (volume integral):

```cpp
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,   // [M] opacity per point
    const scalar_t * __restrict__ rgbs,     // [M, 3] color per point
    const scalar_t * __restrict__ deltas,   // [M, 2] step size + cumulative depth
    const int * __restrict__ rays,          // [N, 3] (ray_id, offset, steps)
    const uint32_t M, const uint32_t N,
    scalar_t * weights_sum,                 // Output [N]
    scalar_t * depth,                       // Output [N]
    scalar_t * image                        // Output [N, 3]
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;
    
    // Load ray batch info
    uint32_t index = rays[n*3];         // Ray ID in original image
    uint32_t offset = rays[n*3+1];      // Offset into sample points array
    uint32_t num_steps = rays[n*3+2];   // Number of samples
    
    // Sanity checks
    if (num_steps == 0 || offset + num_steps >= M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index*3] = image[index*3+1] = image[index*3+2] = 0;
        return;
    }
    
    // Accumulate color and depth
    scalar_t T = 1.0f;      // Transmittance (1 to 0 as we accumulate)
    scalar_t r = 0, g = 0, b = 0;
    scalar_t ws = 0, d = 0;         // weights_sum, depth
    
    for (uint32_t step = 0; step < num_steps; step++) {
        // Volume rendering equation:
        // C = ∫ (sigma(t) * c(t) * exp(-∫sigma dt)) dt
        // Discretized: C_i ≈ (1 - exp(-sigma_i * dt_i)) * T_i * c_i
        
        const scalar_t alpha = 1.0f - __expf(-sigmas[offset + step] * deltas[offset*2 + step*2]);
        // alpha = opacity of this sample [0, 1]
        
        const scalar_t weight = alpha * T;      // Contribution of this sample
        
        // Accumulate color-weighted by transmittance
        r += weight * rgbs[(offset+step)*3];
        g += weight * rgbs[(offset+step)*3+1];
        b += weight * rgbs[(offset+step)*3+2];
        
        d += weight * deltas[offset*2+step*2+1];  // Depth integral
        ws += weight;                               // Weights sum (alpha channel)
        
        T *= 1.0f - alpha;  // Update transmittance for next sample
        // T *= (1 - alpha) because unoccluded area decreases
        // As alpha increases, T decreases (less light reaches beyond)
    }
    
    // Store results
    weights_sum[index] = ws;
    depth[index] = d;
    image[index*3] = r;
    image[index*3+1] = g;
    image[index*3+2] = b;
}
```

**Equazione volumetrica** (spiegazione):

$$C = \int_{t_{near}}^{t_{far}} T(t) \cdot \sigma(t) \cdot c(t) \, dt$$

Dove:
- $T(t) = \exp\left(-\int_0^t \sigma(s) \, ds\right)$ = transmittance (occlusion)
- $\sigma(t)$ = densità (opacity) nel punto t
- $c(t)$ = colore
- Discretizzato: $\alpha_i = 1 - \exp(-\sigma_i \Delta t_i)$

**Timeline**:
- Per 64k ray samples: ~1-2 ms su RTX 4060 Ti

---

### **2.7 Kernel: `kernel_composite_rays_train_backward`**

Calcola gradienti `grad_sigmas` e `grad_rgbs`:

```cpp
__global__ void kernel_composite_rays_train_backward(...) {
    // Same ray loading as forward
    uint32_t index = rays[n*3];
    uint32_t offset = rays[n*3+1];
    uint32_t num_steps = rays[n*3+2];
    
    // Backward accumulation
    // This is the inverse of forward pass
    
    // Forward stored:
    // r = ∑ weight_i * rgb_i
    // ws = ∑ weight_i where weight_i = alpha_i * T_i
    
    // Backward computes:
    // ∂loss/∂sigma_i from ∂loss/∂image and ∂loss/∂weights_sum
    
    scalar_t T = 1.0f;
    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2];
    scalar_t r = 0, g = 0, b = 0, ws = 0;
    
    for (uint32_t step = 0; step < num_steps; step++) {
        const scalar_t alpha = 1.0f - __expf(-sigmas[offset+step] * deltas[offset*2+step*2]);
        const scalar_t weight = alpha * T;
        
        r += weight * rgbs[(offset+step)*3];
        g += weight * rgbs[(offset+step)*3+1];
        b += weight * rgbs[(offset+step)*3+2];
        ws += weight;
        
        // Gradient w.r.t. rgb_i è semplice:
        grad_rgbs[(offset+step)*3] = grad_image[0] * weight;
        grad_rgbs[(offset+step)*3+1] = grad_image[1] * weight;
        grad_rgbs[(offset+step)*3+2] = grad_image[2] * weight;
        
        // Gradient w.r.t. sigma_i è complesso (chain rule):
        // ∂loss/∂sigma_i = delta_t_i * (
        //     grad_image * (T * rgb - (rgb_final - rgb_accum)) +
        //     grad_weights_sum * (T - (weights_final - weights_accum))
        // )
        
        grad_sigmas[offset+step] = deltas[offset*2+step*2] * (
            grad_image[0] * (T * rgbs[(offset+step)*3] - (r_final - r)) +
            grad_image[1] * (T * rgbs[(offset+step)*3+1] - (g_final - g)) +
            grad_image[2] * (T * rgbs[(offset+step)*3+2] - (b_final - b)) +
            grad_weights_sum[0] * (T - (weights_sum[0] - ws))
        );
        
        T *= 1.0f - alpha;
    }
}
```

**Perché così complesso?**

Forward: $C = \sum_i w_i \cdot \text{rgb}_i$ dove $w_i = \alpha_i T_i$

Backward: $\frac{\partial C}{\partial \sigma_i}$ deve considerare:
1. **Effetto diretto**: $\sigma_i$ afcetta $w_i$ direttamente
2. **Effetto indiretto**: $\sigma_i$ afcetta $T_{i+1}$ per campioni futuri

Questo è perché la formula è complessa con termini $(T - (w_{final} - w_{accum}))$

---

### **2.8 PCG32 Random Number Generator (`pcg32.h`)**

Usato per perturbazione anti-aliasing:

```cpp
struct pcg32 {
    uint64_t state, inc;
    
    pcg32(uint64_t initstate, uint64_t initseq = 1) {
        seed(initstate, initseq);
    }
    
    uint32_t next_uint() {
        uint64_t oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18) ^ oldstate) >> 27);
        uint32_t rot = (uint32_t)(oldstate >> 59);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1) & 31));
    }
    
    float next_float() {
        return (float)next_uint() * (1.0f / 4294967296.0f);  // [0, 1)
    }
};
```

**Perché PCG vs rand()?**
- **LCG (linear congruential)**: Cheap ma bassa qualità (pattern visualization)
- **PCG**: Cheap + buona qualità (remedies LCG weaknesses)
- **XORshift**: Buono ma lento
- **Mersenne Twister**: Molto buono ma overkill per CUDA

**Uso in raymarching**:
```cpp
pcg32 rng((uint64_t)n, (uint64_t)perturb);  // Seed con ray ID + SPP
t0 += dt_min * rng.next_float();            // Jitter starting time
```

**Output**: Ogni raggio ha diverso random jitter → anti-aliasing naturale

---

## 3. `bindings.cpp` - PyTorch C++ Binding

```cpp
#include <torch/extension.h>
#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Utils
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.def("polar_from_ray", &polar_from_ray, "polar_from_ray (CUDA)");
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    m.def("packbits", &packbits, "packbits (CUDA)");
    
    // Training
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("composite_rays_train_forward", &composite_rays_train_forward, "forward (CUDA)");
    m.def("composite_rays_train_backward", &composite_rays_train_backward, "backward (CUDA)");
    
    // Inference
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
    m.def("compact_rays", &compact_rays, "compact rays (CUDA)");
}
```

**Cosa fa**:

| Elemento | Spiegazione |
|----------|------------|
| `#include <torch/extension.h>` | Macro PyTorch per definire moduli |
| `PYBIND11_MODULE(...)` | Macro che crea modulo Python bindato a C++ |
| `m.def(...)` | Registra una funzione C++ per uso da Python |
| `&near_far_from_aabb` | Address-of della funzione C++ |
| `"near_far_from_aabb (CUDA)"` | Docstring visibile da Python |

**Flusso di compilazione**:

```
bindings.cpp (+ raymarching.h declarations)
    ↓
Compilatore C++ (cl.exe on Windows, gcc on Linux)
    ↓
Estrae declarations da raymarching.h
    ↓
Crea stub Python che chiama C++
    ↓
...compilation linking with .cu object files...
    ↓
_raymarching.pyd (Windows) o _raymarching.so (Linux)

Result: Quando user fa:
from _raymarching import near_far_from_aabb
│
└─→ Carica .pyd → Chiama C++ wrapper → Esegue CUDA kernel
```

---

## 4. Flusso di Compilazione Completo

```
T0: User scrives (pip install -e . OR import raymarching)
    │
    ├─ setup.py eseguito (if pip install)
    │
    ├─ backend.py eseguito (if import fallisce)
    │   │
    │   └─ torch.cpp_extension.load()
    │       │
    └───→ Trovati file sources:
           • raymarching.cu    (CUDA kernels)
           • bindings.cpp      (C++ wrapper)
           • raymarching.h     (declarations)
    
    ├─ Compilation Phase:
    │  ├─ nvcc (NVIDIA CUDA compiler)
    │  │  Input: raymarching.cu
    │  │  Output: raymarching.cu.o (object file)
    │  │  Flags: -O3 -std=c++14 [CUDA flags]
    │  │
    │  └─ cl.exe / gcc (C++ compiler)
    │     Input: bindings.cpp
    │     Output: bindings.cpp.o
    │     Flags: /O2 /std:c++17 [C++ flags]
    
    ├─ Linking Phase:
    │  Combina .cu.o + .cpp.o
    │  Aggiunge CUDA runtime library
    │  Output: _raymarching.pyd
    
    └─ Runtime (post-compilation):
       _raymarching modulo caricabile in Python
       near_far_from_aabb() disponibile
       GPU execution ready!

T0 + 10s: First import (JIT)
           CUDA kernel è pronto per esecuzione
           
T1: Subsequent imports
           Usa _raymarching.pyd cached (~50ms)
```

---

## 5. Memory Access Patterns

### **Coalesced Access** (veloce → L1 cache hit):

```cuda
// BUONO: Sequential access per thread
// Thread 0 reads: sigmas[0], Thread 1 reads: sigmas[1], etc.
const scalar_t alpha = 1.0f - __expf(-sigmas[n] * deltas[n*2]);

// Memory layout:
// sigmas:  [σ0, σ1, σ2, σ3, σ4, σ5, σ6, σ7, ...]
// Threads: [0,  1,  2,  3,  4,  5,  6,  7,  ...]
// Access pattern: Sequential (1 cacheline = 32 bytes = 8×float4)
// Cache hit rate: ~95%
```

### **Non-coalesced Access** (lento → global memory):

```cuda
// CATTIVO: Scattered access
const scalar_t rgb = rgbs[n * stride];  // stride > 1
// If stride = large, thread 0, 1, 2, 3 access non-contiguous memory
// Cache misses lead to 10-100x slowdown!
```

**Perché `.contiguous()` è obbligatorio**:
```python
rays_o = rays_o.contiguous().view(-1, 3)
```

Se tensor è transposed o tiene permutation diversa, memoria non è C-contiguous. `.contiguous()` riorganizza in memory per coalescing.

---

## 6. Profiling: CUDA vs Python Equivalent

### **kernel_near_far_from_aabb**:

```
Input: 1024 rays
CUDA kernel:
├─ Thread creation: negligibile
├─ AABB slab test: 1024 rays × 9 float ops = 9K ops
├─ Latency hiding: GPU pipeline ≈ 100 ns per inst
└─ Total: ~0.5 ms

Python equivalent:
import torch
for i in range(1024):
    ox, oy, oz = rays_o[i]
    dx, dy, dz = rays_d[i]
    ...
    
CPU latency:
├─ Loop overhead: 1024 × ~100 ns/iteration = 100 μs
├─ Memory load: 1024 × 3 floats × ~100 ns = 300 μs
├─ Computation: 1024 × 9 ops × ~5 ns = 45 μs (CPU has better ALU)
└─ Total: ~500 μs (CPU)

Speedup: 500 μs / 0.5 ms = ~1000x (GPU)
```

### **kernel_composite_rays_train (more realistic)**:

```
Input: 64k ray samples
CUDA kernel:
├─ Bandwidth-bound (memory reads exceed compute)
├─ Per-sample: 1 sigma + 3 RGB + 2 delta = 6 floats load + 2 float exp + mult
├─ Total ops: 64k × ~15 FP ops ≈ 1M FLOPs
├─ Memory: 64k × 6 floats × 4 bytes = 1.5 MB (fits in L2 cache!)
├─ Latency: ~2 ms (memory-bound due to exp)

Python NumPy:
alpha = 1 - np.exp(-sigma * delta)
weight = alpha * T
RGB = np.sum(weight * rgb)

NumPy (vectorized):
├─ Memory bandwidth: CPU ~100 GB/s vs GPU ~500 GB/s
├─ Compute: exp is expensive on CPU
└─ Total: ~50-100 ms (memory + compute not pipelined well)

Speedup: 50 ms / 2 ms = ~25x (GPU, realistic with overhead)
```

---

## 7. Data Types: Float16 vs Float32

### **Float16 (half precision)**:

```cpp
Memoria: 2 bytes per valore (vs 4 bytes float32)
Range: [±6×10^-8, ±65504]
Precision: ~3-4 decimal digits

CUDA support:
├─ Hardware: RTX series ha tensor cores per fp16
├─ Speed: 2x faster per watt vs float32
├─ Accuracy: Sufficient per network training

Mixed precision (come in E-NeRF):
├─ Forward: float32 (stability)
├─ Backward: float32 (gradient computation needs precision)
├─ Storage: float16 (memory efficiency)
├─ @custom_fwd(cast_inputs=torch.float32): Ensures stability
```

### **Float32 (single precision)**:

```cpp
Memoria: 4 bytes
Range: [±1.2×10^-38, ±3.4×10^38]
Precision: ~7 decimal digits

Situation:
├─ Default for scientific computing
├─ Required for grad computation (exp() can have NaN in float16)
├─ Network weights: float32 after gradient step
```

---

## Summary: Catena di Esecuzione Completa

```
┌──────────────────────────────────────────────────────────────────────┐
│ TRAINING ITERATION                                                   │
└──────────────────────────────────────────────────────────────────────┘

1. FORWARD PASS:
   ├─ near_far_from_aabb()
   │  └─ kernel_near_far_from_aabb() per 1024 rays (0.5 ms)
   │
   ├─ march_rays_train()
   │  └─ kernel_march_rays_train() per 1024 rays (adaptive steps, 2-10 ms)
   │     Output: 32k-64k sample points
   │
   ├─ Network forward (density_net, color_net)
   │  └─ PyTorch NN.Linear (FFMLP kernel) (1-5 ms)
   │     Output: sigmas [64k], rgbs [64k, 3]
   │
   └─ composite_rays_train_forward()
      └─ kernel_composite_rays_train_forward() (2 ms)
         Output: image [1024, 3], depth [1024], weights_sum [1024]

   Total forward: ~10-20 ms per batch

2. LOSS COMPUTATION:
   └─ L2 loss = norm(image - gt_image)^2
      + other regularizations
      Gradient computed by PyTorch autograd

3. BACKWARD PASS:
   ├─ loss.backward() triggers CUDA backward kernels
   │
   ├─ composite_rays_train_backward()
   │  └─ kernel_composite_rays_train_backward() (2 ms)
   │     Output: grad_sigmas [64k], grad_rgbs [64k, 3]
   │
   ├─ Network backward (density_net.backward, color_net.backward)
   │  └─ PyTorch autograd + CUDA (FFMLP backward kernel) (3-10 ms)
   │     Output: grad_xyz, grad_dirs (for network inputs)
   │     Update: density_net.weight.grad, color_net.weight.grad
   │
   └─ Ray marching backward (chain rule)
      └─ Auto-diff through near_far (negligible since no backward defined)

   Total backward: ~8-15 ms per batch

4. OPTIMIZER STEP:
   └─ optimizer.step() updates network weights
      density_net.weight -= lr * .weight.grad
      (Done on GPU, ~1 ms)

5. TOTAL TIME PER ITERATION:
   ├─ Forward: 10-20 ms
   ├─ Backward: 8-15 ms
   ├─ Optimizer: 1 ms
   └─ Total: ~20-40 ms per iteration
      
   10k iterations = 200-400 seconds ≈ 3-7 minutes on RTX 4060 Ti

MEMORY USAGE (RTX 4060 Ti, 8GB):
├─ Model weights: 1-2 GB
├─ Raymarching tensors: 2-3 GB (64k points × 3 floats × 4 bytes)
├─ Gradient buffers: 2-3 GB
└─ Total: ~5-8 GB (within RTX 4060 Ti limit with fp16 + gradient checkpointing)
```

---
