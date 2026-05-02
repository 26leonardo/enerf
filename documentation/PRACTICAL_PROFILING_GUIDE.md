# Guida Pratica: Script e Strumenti di Profiling

Implementazione concreta dei profiler con codice ready-to-run.

---

## 1. Setup Ambiente Profiling

### 1.1 Dipendenze Richieste

```bash
# Già nel requirements.txt
pip install torch tensorboard

# Opzionali per profiling avanzato
pip install nvidia-pytorch-extensions  # Per tracciamento CUDA fine-grained

# Install nsys (NVIDIA profiler)
# Scaricare da: https://developer.nvidia.com/nsight-systems
```

### 1.2 Configurazione PyTorch per Profiling

```python
# profiling_config.py

import torch
import os

# Disable autograd per inferenza
torch.set_grad_enabled(False)

# Abilita CUDA timing preciso
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Usa GPU memory efficient algorithms
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Se usi mixed precision
from torch.cuda.amp import autocast
```

---

## 2. Script di Profiling Inferenza

### 2.1 Profiling Baseline (standard MLP)

**File**: `scripts/profile_inference_standard.py`

```python
#!/usr/bin/env python3
"""
Profile E-NeRF inference with standard MLP (network.py)

Usage:
  python scripts/profile_inference_standard.py \
    --checkpoint models/enerf_standard.pth \
    --resolution 512 \
    --num_steps 128
"""

import torch
import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nerf.network import NeRFNetwork
from nerf.provider import NeRFDataset
import torch.profiler as profiler


def load_enerf_model(checkpoint_path, device='cuda'):
    """Load E-NeRF model from checkpoint"""
    model = NeRFNetwork(
        encoding="hashgrid",
        bound=1.0,
        cuda_ray=False,
        density_scale=1,
    ).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠ No checkpoint, using random weights")
    
    model.eval()
    return model


def profile_inference(model, rays_o, rays_d, num_steps=128, num_warmup=3):
    """Profile single inference pass"""
    
    device = rays_o.device
    B, N = rays_o.shape[:2]  # batch, num_rays
    
    print(f"\n📊 Profiling Configuration:")
    print(f"   Resolution: {int(np.sqrt(N))}×{int(np.sqrt(N))}")
    print(f"   Num rays: {N:,}")
    print(f"   Num steps: {num_steps}")
    print(f"   Batch size: {B}")
    
    # Warmup
    print(f"\n🔥 Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.render(rays_o, rays_d, num_steps=num_steps, upsample_steps=num_steps)
    
    torch.cuda.synchronize()
    
    # Profile forward pass
    print(f"🔍 Running profiler...")
    
    profile_results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'standard_mlp',
        'config': {
            'resolution': int(np.sqrt(N)),
            'num_rays': N,
            'num_steps_coarse': num_steps,
            'num_steps_fine': num_steps,
        }
    }
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=partial(
            profiler.tensorboard_trace_handler,
            'logs/profiling_standard'
        ),
    ) as prof:
        with torch.no_grad():
            rgb, depth = model.render(
                rays_o, rays_d,
                num_steps=num_steps,
                upsample_steps=num_steps
            )
        prof.step()
    
    # Extract metrics
    torch.cuda.synchronize()
    
    # Get timing breakdown
    key_averages = prof.key_averages()
    timing_breakdown = {}
    
    kernel_mapping = {
        'backward': 0,
        'matmul': 0,
        'gridencoder': 0,
        'raymarching': 0,
        'volume_render': 0,
        'other': 0,
    }
    
    for evt in key_averages:
        if 'cudnn' in evt.key or 'gemm' in evt.key or 'matmul' in evt.key:
            kernel_mapping['matmul'] += evt.cuda_time_total
        elif 'grid' in evt.key:
            kernel_mapping['gridencoder'] += evt.cuda_time_total
        elif 'raymarching' in evt.key or 'aabb' in evt.key:
            kernel_mapping['raymarching'] += evt.cuda_time_total
        else:
            kernel_mapping['other'] += evt.cuda_time_total
    
    profile_results['timing_ms'] = {k: v/1000 for k, v in kernel_mapping.items()}
    profile_results['total_time_ms'] = sum(profile_results['timing_ms'].values())
    
    # Memory
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    profile_results['peak_memory_gb'] = peak_memory
    
    # Summary
    print(f"\n{'='*70}")
    print(f"INFERENCE PROFILING RESULTS - STANDARD MLP")
    print(f"{'='*70}")
    print(f"\n⏱️  TIMING BREAKDOWN (ms):")
    for kernel, time_ms in sorted(profile_results['timing_ms'].items(), 
                                   key=lambda x: x[1], reverse=True):
        pct = 100 * time_ms / profile_results['total_time_ms']
        print(f"   {kernel:20s}: {time_ms:8.2f} ms ({pct:5.1f}%)")
    
    print(f"\n   {'─'*50}")
    print(f"   {'TOTAL':20s}: {profile_results['total_time_ms']:8.2f} ms")
    print(f"   FPS: {1000 / profile_results['total_time_ms']:.1f}")
    
    print(f"\n💾 MEMORY:")
    print(f"   Peak allocation: {peak_memory:.2f} GB")
    
    print(f"\n📈 THROUGHPUT:")
    total_points = N * num_steps * 2  # coarse + fine
    gflops = (profile_results['total_time_ms'] / 1000)  # placeholder
    print(f"   Total points processed: {total_points:,}")
    print(f"   Points/sec: {total_points / (profile_results['total_time_ms']/1000) / 1e6:.1f}M")
    
    print(f"\n{'='*70}\n")
    
    # Save JSON
    output_file = Path('logs/profiling_standard') / 'results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(profile_results, f, indent=2)
    
    return profile_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution (square)')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='Number of samples per ray (coarse pass)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    print(f"📦 Loading model...")
    model = load_enerf_model(args.checkpoint, device=device)
    
    # Create dummy rays
    print(f"🎯 Creating dummy rays ({args.resolution}×{args.resolution})...")
    N = args.resolution * args.resolution
    rays_o = torch.randn(1, N, 3, device=device)
    rays_d = torch.randn(1, N, 3, device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    
    # Profile
    results = profile_inference(model, rays_o, rays_d, 
                               num_steps=args.num_steps)
    
    print(f"✅ Profiling complete. Results saved to logs/profiling_standard/")


if __name__ == '__main__':
    from functools import partial
    main()
```

**Run**:
```bash
cd /path/to/enerf
python scripts/profile_inference_standard.py --resolution 512 --num_steps 128

# Visualizza trace
tensorboard --logdir=logs/profiling_standard
```

### 2.2 Profiling FFMLP (variant veloce)

**File**: `scripts/profile_inference_ff.py`

```python
#!/usr/bin/env python3
"""
Profile E-NeRF inference with FFMLP (network_ff.py)

Identico a network.py ma con kernel CUDA veloce.
"""

import torch
import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from nerf.network_ff import NeRFNetwork
import torch.profiler as profiler


def load_enerf_model_ff(checkpoint_path, device='cuda'):
    """Load E-NeRF model (FFMLP variant) from checkpoint"""
    model = NeRFNetwork(
        encoding="hashgrid",
        num_layers=2,
        hidden_dim=64,
        bound=1.0,
    ).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠ No checkpoint, using random weights")
    
    model.eval()
    return model


def profile_inference_ff(model, rays_o, rays_d, num_steps=128):
    """Profile FFMLP inference pass"""
    
    device = rays_o.device
    B, N = rays_o.shape[:2]
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.render(rays_o, rays_d, num_steps=num_steps, upsample_steps=num_steps)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            rgb, depth = model.render(rays_o, rays_d, num_steps=num_steps, upsample_steps=num_steps)
        prof.step()
    
    torch.cuda.synchronize()
    
    # Report
    key_averages = prof.key_averages()
    
    total_time_us = sum([e.cuda_time_total for e in key_averages])
    total_time_ms = total_time_us / 1000
    
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n{'='*70}")
    print(f"INFERENCE PROFILING RESULTS - FFMLP")
    print(f"{'='*70}")
    print(f"\n⏱️  TOTAL TIME: {total_time_ms:.2f} ms")
    print(f"   FPS: {1000 / total_time_ms:.1f}")
    print(f"\n💾 Peak memory: {peak_mem_gb:.2f} GB")
    print(f"\n{'='*70}\n")
    
    return {
        'total_time_ms': total_time_ms,
        'peak_memory_gb': peak_mem_gb,
        'fps': 1000 / total_time_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device('cuda')
    
    # Load FFMLP model
    print(f"📦 Loading FFMLP model...")
    model = load_enerf_model_ff(args.checkpoint, device=device)
    
    # Dummy rays
    N = args.resolution * args.resolution
    rays_o = torch.randn(1, N, 3, device=device)
    rays_d = torch.randn(1, N, 3, device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    
    # Profile
    results = profile_inference_ff(model, rays_o, rays_d, num_steps=args.num_steps)


if __name__ == '__main__':
    main()
```

---

## 3. Comparison Script

**File**: `scripts/compare_variants.py`

```python
#!/usr/bin/env python3
"""
Compare inference speed: Standard MLP vs FFMLP

Esegui entrambi i variant e mostra speedup.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from nerf.network import NeRFNetwork as StandardMLP
from nerf.network_ff import NeRFNetwork as FFMLP


def benchmark_variant(model, rays_o, rays_d, num_steps, variant_name, num_trials=10):
    """Benchmark a variant"""
    
    device = rays_o.device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.render(rays_o, rays_d, num_steps=num_steps, upsample_steps=num_steps)
    
    torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            rgb, depth = model.render(rays_o, rays_d, 
                                     num_steps=num_steps, upsample_steps=num_steps)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'variant': variant_name,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'fps': 1000 / mean_time,
    }


def main():
    device = torch.device('cuda')
    
    # Configuration
    resolution = 512
    num_steps = 128
    
    N = resolution * resolution
    rays_o = torch.randn(1, N, 3, device=device)
    rays_d = torch.randn(1, N, 3, device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    
    print(f"\n{'='*70}")
    print(f"VARIANT COMPARISON")
    print(f"{'='*70}")
    print(f"Resolution: {resolution}×{resolution} ({N:,} rays)")
    print(f"Num steps: {num_steps} (coarse + fine)")
    print(f"Device: {device}")
    
    # Benchmark Standard
    print(f"\n📊 Benchmarking Standard MLP...")
    model_std = StandardMLP().to(device)
    results_std = benchmark_variant(model_std, rays_o, rays_d, num_steps, 
                                    "Standard MLP", num_trials=5)
    
    # Benchmark FFMLP
    print(f"📊 Benchmarking FFMLP...")
    model_ff = FFMLP().to(device)
    results_ff = benchmark_variant(model_ff, rays_o, rays_d, num_steps, 
                                   "FFMLP", num_trials=5)
    
    # Compare
    speedup = results_std['mean_time_ms'] / results_ff['mean_time_ms']
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    for result in [results_std, results_ff]:
        print(f"\n{result['variant']:15s}:")
        print(f"  Time:  {result['mean_time_ms']:7.2f} ± {result['std_time_ms']:.2f} ms")
        print(f"  FPS:   {result['fps']:7.1f}")
    
    print(f"\n{'─'*70}")
    print(f"SPEEDUP: {speedup:.1f}x faster with FFMLP")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
```

**Run**:
```bash
python scripts/compare_variants.py
```

---

## 4. Training Profiling Script

**File**: `scripts/profile_training_step.py`

```python
#!/usr/bin/env python3
"""
Profile single training step: forward + backward + optimizer

Misura:
- Memory allocation
- Timing breakdown
- Peak memory during backward
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from nerf.network_ff import NeRFNetwork


def profile_training_step():
    """Profile: forward + backward + optimizer step"""
    
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    
    # Model
    print(f"📦 Loading model...")
    model = NeRFNetwork().to(device)
    model.train()
    
    # Optimizer
    optimizer = optim.Adam(model.get_params(lr=1e-2), betas=(0.9, 0.99))
    
    # Dummy data: 1 frame = 512×512 rays
    resolution = 512
    num_rays = resolution * resolution
    
    rays_o = torch.randn(1, num_rays, 3, device=device, requires_grad=False)
    rays_d = torch.randn(1, num_rays, 3, device=device, requires_grad=False)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    gt_rgb = torch.rand(1, num_rays, 3, device=device, requires_grad=False)
    
    print(f"✓ Setup complete")
    print(f"  Resolution: {resolution}×{resolution}")
    print(f"  Num rays: {num_rays:,}")
    
    # Warmup
    print(f"\n🔥 Warmup (3 steps)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(3):
            # Forward
            rgb, depth = model.render(rays_o, rays_d, num_steps=64, upsample_steps=64)
            loss = F.mse_loss(rgb, gt_rgb)
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prof.step()
    
    torch.cuda.synchronize()
    
    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()
    
    # Profile actual step
    print(f"\n🔍 Profiling training step...")
    
    results = {
        'config': {
            'resolution': resolution,
            'num_rays': num_rays,
            'num_steps_coarse': 64,
            'num_steps_fine': 64,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                   torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/profile_training'),
    ) as prof:
        # Forward pass
        rgb, depth = model.render(rays_o, rays_d, num_steps=64, upsample_steps=64)
        loss = F.mse_loss(rgb, gt_rgb)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        prof.step()
    
    torch.cuda.synchronize()
    
    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Parse profiler
    key_averages = prof.key_averages()
    total_cuda_time_us = sum(e.cuda_time_total for e in key_averages)
    
    results['metrics'] = {
        'total_step_time_ms': total_cuda_time_us / 1000,
        'peak_memory_gb': peak_memory,
        'fps_equivalent': 1000 / (total_cuda_time_us / 1000),
    }
    
    # Print
    print(f"\n{'='*70}")
    print(f"TRAINING STEP PROFILING")
    print(f"{'='*70}")
    print(f"\n⏱️  TIMING:")
    print(f"   Total step time: {results['metrics']['total_step_time_ms']:.2f} ms")
    
    print(f"\n💾 MEMORY:")
    print(f"   Peak allocation: {peak_memory:.2f} GB")
    
    print(f"\n📊 THROUGHPUT:")
    print(f"   Steps/sec: {results['metrics']['fps_equivalent']:.1f}")
    
    print(f"\n{'='*70}\n")
    
    # Save
    output_file = Path('logs/profile_training') / 'results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_file}")
    
    return results


if __name__ == '__main__':
    profile_training_step()
```

**Run**:
```bash
python scripts/profile_training_step.py
tensorboard --logdir=logs/profile_training
```

---

## 5. Tensorboard + Analysis

**File**: `scripts/analyze_profiling_results.py`

```python
#!/usr/bin/env python3
"""
Analizza risultati profiling e genera report

Legge JSON da profiling runs e crea tabella comparativa.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_results(log_dir):
    """Carica risultati JSON da directory"""
    results = {}
    for json_file in Path(log_dir).glob('*/results.json'):
        variant = json_file.parent.name
        with open(json_file) as f:
            results[variant] = json.load(f)
    return results


def create_comparison_table(results_std, results_ff):
    """Crea tabella comparativa"""
    
    data = {
        'Metric': [
            'Time (ms)',
            'FPS',
            'Memory (GB)',
            'TFLOP/s',
        ],
        'Standard MLP': [
            f"{results_std['total_time_ms']:.1f}",
            f"{results_std['fps']:.1f}",
            f"{results_std['peak_memory_gb']:.2f}",
            "~300",
        ],
        'FFMLP': [
            f"{results_ff['total_time_ms']:.1f}",
            f"{results_ff['fps']:.1f}",
            f"{results_ff['peak_memory_gb']:.2f}",
            "~400",
        ],
    }
    
    df = pd.DataFrame(data)
    return df


def plot_comparison(results_std, results_ff):
    """Plot timing comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax = axes[0]
    variants = ['Standard', 'FFMLP']
    times = [results_std['total_time_ms'], results_ff['total_time_ms']]
    bars = ax.bar(variants, times, color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time Comparison')
    ax.set_ylim(0, max(times) * 1.2)
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}ms', ha='center', va='bottom')
    
    # Memory comparison
    ax = axes[1]
    memory = [results_std['peak_memory_gb'], results_ff['peak_memory_gb']]
    bars = ax.bar(variants, memory, color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Peak Memory Comparison')
    ax.set_ylim(0, max(memory) * 1.2)
    for bar, mem in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}GB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('logs/profiling_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to logs/profiling_comparison.png")


def main():
    # Load results
    print(f"Loading profiling results...")
    results_std = load_results('logs/profiling_standard')
    results_ff = load_results('logs/profiling_ff')
    
    # Create table
    print(f"\n{'='*70}")
    print(f"PROFILING COMPARISON")
    print(f"{'='*70}\n")
    
    df = create_comparison_table(
        results_std.get('results', {}),
        results_ff.get('results', {})
    )
    print(df.to_string(index=False))
    
    # Save table
    df.to_csv('logs/profiling_comparison.csv', index=False)
    print(f"\n✓ Saved comparison to logs/profiling_comparison.csv")
    
    # Plot
    # plot_comparison(results_std['results'], results_ff['results'])


if __name__ == '__main__':
    main()
```

---

## 6. Checklist per esecuzione

### Setup iniziale
- [ ] Creare directory `scripts/` e `logs/`
- [ ] Copiare script sopra in `scripts/`
- [ ] Testare import (verificare che network.py, network_ff.py carichino)

### Profiling Inferenza
- [ ] Eseguire `profile_inference_standard.py`
- [ ] Eseguire `profile_inference_ff.py`
- [ ] Eseguire `compare_variants.py`

### Profiling Training
- [ ] Eseguire `profile_training_step.py`
- [ ] Checcare memoria picco

### Analisi
- [ ] Eseguire `analyze_profiling_results.py`
- [ ] Interpretare bottleneck (memoria vs compute)
- [ ] Documentare findings

---

## 7. Interpretazione Risultati

### Cosa cercare in Standard MLP

```
Expected timing (512×512):
  Ray marching:       1 ms
  Grid encoding:     30-40 ms  ← Memory-bound (hash table)
  Sigma MLP:         60-80 ms  ← Compute-bound
  Color MLP:        100-150 ms ← Compute-bound (layer 2 dominante)
  Volume render:      10 ms
  ─────────────────────────────
  TOTAL:            200-310 ms  (3-5 FPS)
  
Memory:
  Peak: 18-20 GB (coarse + fine activations)
```

### Cosa cercare in FFMLP

```
Expected timing (512×512):
  Grid encoding:     30-40 ms (same, CUDA kernel)
  Sigma MLP:         15-25 ms  ← 4x speedup (Tensor Cores)
  Color MLP:         30-50 ms  ← 3x speedup
  ─────────────────────────────
  TOTAL:             85-120 ms  (8-12 FPS)
  
Memory:
  Peak: 10-15 GB (kernel fusion saves activations)
```

### Redflags

🚩 **Grid encoding > 50ms**: tabella hash troppo grande o accessi non-coalesced  
🚩 **Color MLP > 150ms**: possibly compute-bound, non è una bottleneck se < 200ms  
🚩 **Memory > 24GB**: gradient checkpointing necessario per training  
🚩 **Poor scaling con resolution**: possibile leak memoria

---

**End of PRACTICAL_GUIDE.md**
