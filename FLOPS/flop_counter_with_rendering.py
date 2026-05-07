"""
FLOP Counter per NeRF con Ray Marching (renderer.py.run())

Questo conta FLOPs di:
✓ network.py (forward pass per ogni sample)
✓ renderer.py.run() - ray marching + compositing
✓ CUDA ray marching opzionale

NOTA: Aggiunge overhead del ray marching (coarse + fine stage)
"""

import torch
import numpy as np
from fvcore.nn import FlopCounterMode, flop_count_str
from functools import wraps


class FlopProfilerWithRendering:
    """
    Profile FLOPs includendo ray marching completo
    
    Architettura:
    1. Coarse stage: num_steps samples per raggio
    2. Upsample stage: upsample_steps campioni aggiuntivi
    3. Color: calcolo solo per punti con weights > 1e-4
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.flops_stats = {
            'network_flops': 0,
            'raymarching_flops': 0,
            'compositing_flops': 0,
            'total_flops': 0,
        }
    
    def count_flops_network_inference(self, num_samples, network_flops_per_sample=22432):
        """
        FLOPs del network per inferenza
        
        Args:
            num_samples: numero totale di campioni 3D
            network_flops_per_sample: FLOPs per singolo sample (default: encoder+sigma+color)
        
        Returns:
            flops: FLOPs totali del network
        """
        return num_samples * network_flops_per_sample
    
    def count_flops_density_network(self, num_samples, hidden_dim=64, geo_feat_dim=15, D_enc=32):
        """
        FLOPs solo della rete di densità (sigma network)
        
        sigma_net:
          Layer 1: Linear(D_enc -> hidden_dim)
          Layer 2: Linear(hidden_dim -> 1 + geo_feat_dim)
        """
        # Encoder hashgrid
        encoder_flops = 1000 + (num_samples * 5)  # approssimato
        
        # Layer 1: D_enc -> hidden_dim
        layer1_flops = 2 * num_samples * D_enc * hidden_dim
        
        # Layer 2: hidden_dim -> (1 + geo_feat_dim)
        layer2_flops = 2 * num_samples * hidden_dim * (1 + geo_feat_dim)
        
        total = encoder_flops + layer1_flops + layer2_flops
        return total
    
    def count_flops_color_network(self, num_samples, D_dir=16, geo_feat_dim=15, 
                                   hidden_dim=64, out_dim_color=1):
        """
        FLOPs della rete di colore
        
        color_net:
          Layer 1: Linear(D_dir + geo_feat_dim -> hidden_dim)
          Layer 2: Linear(hidden_dim -> hidden_dim)
          Layer 3: Linear(hidden_dim -> out_dim_color)
        """
        # Direction encoder (sphere harmonics)
        dir_encoder_flops = 500 + (num_samples * 2)
        
        # Layer 1
        layer1_flops = 2 * num_samples * (D_dir + geo_feat_dim) * hidden_dim
        
        # Layer 2
        layer2_flops = 2 * num_samples * hidden_dim * hidden_dim
        
        # Layer 3
        layer3_flops = 2 * num_samples * hidden_dim * out_dim_color
        
        total = dir_encoder_flops + layer1_flops + layer2_flops + layer3_flops
        return total
    
    def count_flops_raymarching(self, num_rays, num_steps=128, upsample_steps=128):
        """
        FLOPs del ray marching non-CUDA (renderer.py.run)
        
        Operazioni:
        1. Generate z_vals: linspace (num_rays × num_steps) ≈ 2K ops
        2. Generate xyzs: ray_o + ray_d * z = (num_rays × num_steps × 3) ≈ 20K ops
        3. Density queries: num_steps + upsample_steps (network.py)
        4. PDF sampling: searchsorted + gather (num_rays × upsample_steps) ≈ 10K ops
        5. Weight calculation: exp + cumprod (num_rays × (num_steps + upsample_steps)) ≈ 5K ops
        6. Depth/image compositing: weighted sum ≈ 3K ops
        """
        total_samples = num_rays * (num_steps + upsample_steps)
        
        # z_vals generation
        linspace_flops = num_rays * num_steps * 2
        
        # xyz generation: ray_o + ray_d * z
        xyz_gen_flops = num_rays * total_samples * 3
        
        # PDF sampling (searchsorted)
        pdf_flops = num_rays * upsample_steps * 20  # searchsorted is log(n) per sample
        
        # Weight calculation: 1 - exp(-delta * sigma)
        weight_flops = num_rays * total_samples * 5
        
        # Compositing: weighted sum
        composite_flops = num_rays * total_samples * 3
        
        total = linspace_flops + xyz_gen_flops + pdf_flops + weight_flops + composite_flops
        return total
    
    def count_flops_event_batch_with_rendering(self,
                                                 rays_evs_o1, rays_evs_d1,
                                                 rays_evs_o2, rays_evs_d2,
                                                 num_steps=128,
                                                 upsample_steps=128,
                                                 batch_size_evs=4096,
                                                 measure_mode='forward'):
        """
        FLOPs TOTALI: network + ray marching per event batch
        
        Per ogni coppia di eventi (2 raggi):
          - Coarse: num_steps samples × 2 network calls
          - Fine: upsample_steps samples × 2 network calls
          - Ray marching: operations su z_vals, xyz, pdf, compositing
        """
        
        M = batch_size_evs  # event samples
        num_rays = M * 2  # 2 raggi per evento (o1 e o2)
        
        print(f"[EVENT RENDERING] Event samples: {M}")
        print(f"[EVENT RENDERING] Total rays: {num_rays} (2 per event)")
        print(f"[EVENT RENDERING] Coarse steps: {num_steps}, Fine steps: {upsample_steps}")
        
        # STAGE 1: COARSE (num_steps)
        # Density network calls: 2 * M * num_steps samples
        density_samples_coarse = num_rays * num_steps
        flops_density_coarse = self.count_flops_density_network(density_samples_coarse)
        
        # STAGE 2: FINE (upsample_steps)
        density_samples_fine = num_rays * upsample_steps
        flops_density_fine = self.count_flops_density_network(density_samples_fine)
        
        # STAGE 3: COLOR (calcolato per ~50% dei samples, onde weights > 1e-4)
        color_samples = int((density_samples_coarse + density_samples_fine) * 0.5)
        flops_color = self.count_flops_color_network(color_samples)
        
        # STAGE 4: RAY MARCHING (PDF sampling, alpha compositing)
        flops_raymarching = self.count_flops_raymarching(num_rays, num_steps, upsample_steps)
        
        # TOTALE
        flops_network = flops_density_coarse + flops_density_fine + flops_color
        flops_total = flops_network + flops_raymarching
        
        # For training: include backward pass
        if measure_mode == 'full':
            flops_total *= 3  # forward + 2x backward
        
        self.flops_stats['network_flops'] = flops_network
        self.flops_stats['raymarching_flops'] = flops_raymarching
        self.flops_stats['total_flops'] = flops_total
        
        print(f"\n[FLOP BREAKDOWN]")
        print(f"  Density network (coarse + fine): {flop_count_str(flops_density_coarse + flops_density_fine)}")
        print(f"  Color network (masked): {flop_count_str(flops_color)}")
        print(f"  Ray marching overhead: {flop_count_str(flops_raymarching)}")
        print(f"  Total (batch): {flop_count_str(flops_total)}")
        
        return flops_total, {
            'density_coarse': flops_density_coarse,
            'density_fine': flops_density_fine,
            'color': flops_color,
            'raymarching': flops_raymarching,
            'total': flops_total,
        }
    
    def count_flops_cuda_raymarching(self, num_rays, num_steps=128, 
                                      use_density_grid=True):
        """
        FLOPs per CUDA ray marching (renderer.py.run_cuda)
        
        CUDA Ray Marching è molto più efficiente:
        - Usa density grid (8-bit bitfield) per early termination
        - Sampling adattivo (n_step varia dinamicamente)
        - Molto meno overhead di ray marching Python
        
        Approssimativamente:
        - 50-70% meno FLOPs rispetto a Python ray marching
        - Pero` conteggi meno network calls (early stopping)
        """
        
        # CUDA ray marching overhead (molto ridotto)
        # march_rays_train: grid traversal, bitfield check, sampling
        cuda_overhead = num_rays * num_steps * 2  # 2 ops per step invece di 20+
        
        # Network calls: meno grazie al culling con density grid
        # Assumiamo ~70% dei samples vengono elaborati
        network_calls = int(num_rays * num_steps * 0.7)
        flops_network = self.count_flops_density_network(network_calls) + \
                       self.count_flops_color_network(network_calls)
        
        total = flops_network + cuda_overhead
        
        print(f"\n[CUDA RAYMARCHING FLOPS]")
        print(f"  Network calls (with density grid culling): {flop_count_str(flops_network)}")
        print(f"  CUDA overhead: {flop_count_str(cuda_overhead)}")
        print(f"  Total: {flop_count_str(total)}")
        
        return total
    
    def get_summary(self):
        """Riassunto FLOPs"""
        return {
            'network_flops': flop_count_str(self.flops_stats['network_flops']),
            'raymarching_flops': flop_count_str(self.flops_stats['raymarching_flops']),
            'total_flops': flop_count_str(self.flops_stats['total_flops']),
            'network_flops_raw': self.flops_stats['network_flops'],
            'raymarching_flops_raw': self.flops_stats['raymarching_flops'],
            'total_flops_raw': self.flops_stats['total_flops'],
        }


class RenderingFlopsComparison:
    """
    Confronta FLOPs: network-only vs full rendering
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def compare_rendering_methods(self, batch_size_evs=4096, num_steps=128, 
                                   upsample_steps=128):
        """
        Confronta:
        1. Network.py only (quello che calcolavamo prima)
        2. Full rendering Python ray marching
        3. CUDA ray marching
        """
        
        profiler = FlopProfilerWithRendering(self.model, self.device)
        
        print("\n" + "="*70)
        print("COMPARING RENDERING METHODS: FLOP ANALYSIS")
        print("="*70)
        
        num_rays = batch_size_evs * 2  # 2 rays per event
        
        # METHOD 1: Network only (old calculation)
        network_samples = num_rays * (num_steps + upsample_steps)
        flops_network_only = network_samples * 22432  # ~22K FLOPs per sample
        
        print(f"\n1. NETWORK ONLY (network.py)")
        print(f"   Total samples: {network_samples:,}")
        print(f"   FLOPs: {flop_count_str(flops_network_only)}")
        print(f"   {flops_network_only:,.0f} FLOPs")
        
        # METHOD 2: Full Python ray marching
        flops_full_python, _ = profiler.count_flops_event_batch_with_rendering(
            rays_evs_o1=None,
            rays_evs_d1=None,
            rays_evs_o2=None,
            rays_evs_d2=None,
            num_steps=num_steps,
            upsample_steps=upsample_steps,
            batch_size_evs=batch_size_evs,
            measure_mode='forward'
        )
        
        print(f"\n2. FULL PYTHON RAY MARCHING (renderer.py.run)")
        print(f"   FLOPs: {flop_count_str(flops_full_python)}")
        
        summary = profiler.get_summary()
        print(f"   Network component: {summary['network_flops']}")
        print(f"   Ray marching overhead: {summary['raymarching_flops']}")
        
        # METHOD 3: CUDA ray marching
        flops_cuda = profiler.count_flops_cuda_raymarching(num_rays, num_steps)
        
        print(f"\n3. CUDA RAY MARCHING (renderer.py.run_cuda - OPTIMIZED)")
        print(f"   FLOPs: {flop_count_str(flops_cuda)}")
        
        # COMPARAZIONE
        overhead_python = (flops_full_python - flops_network_only) / flops_network_only * 100
        overhead_cuda = (flops_cuda - flops_network_only) / flops_network_only * 100
        speedup_cuda = flops_full_python / flops_cuda
        
        print(f"\n" + "="*70)
        print("OVERHEAD ANALYSIS")
        print("="*70)
        print(f"Network only FLOPs: {flop_count_str(flops_network_only)}")
        print(f"Python ray marching overhead: +{overhead_python:.1f}%")
        print(f"CUDA ray marching overhead: +{overhead_cuda:.1f}%")
        print(f"CUDA speedup vs Python: {speedup_cuda:.2f}x")


# Example: Integration in main training loop
def profile_training_with_rendering(model, opt, train_loader, device, num_profiles=3):
    """
    Profile FLOPs con rendering completo
    
    Uso:
        python main_nerf.py --config ... --profile_flops --use_rendering
    """
    
    profiler = FlopProfilerWithRendering(model, device=device)
    
    print("\n" + "="*70)
    print("PROFILING WITH FULL RENDERING (network.py + renderer.py)")
    print("="*70)
    
    model.eval()
    
    for i, data in enumerate(train_loader):
        if i >= num_profiles:
            break
        
        rays_evs_o1 = data['rays_evs_o1'].to(device)
        rays_evs_d1 = data['rays_evs_d1'].to(device)
        rays_evs_o2 = data['rays_evs_o2'].to(device)
        rays_evs_d2 = data['rays_evs_d2'].to(device)
        
        flops_total, flops_breakdown = profiler.count_flops_event_batch_with_rendering(
            rays_evs_o1=rays_evs_o1,
            rays_evs_d1=rays_evs_d1,
            rays_evs_o2=rays_evs_o2,
            rays_evs_d2=rays_evs_d2,
            num_steps=opt.num_steps,
            upsample_steps=opt.upsample_steps,
            batch_size_evs=opt.batch_size_evs,
            measure_mode='full'
        )
        
        print(f"\nBatch {i+1}/{num_profiles}: {flop_count_str(flops_total)}")
    
    return profiler.get_summary()


if __name__ == '__main__':
    # Quick test
    print("\n" + "="*70)
    print("FLOP COUNTER: Network + Rendering")
    print("="*70)
    print("\nConfiguration:")
    print("  - batch_size_evs = 4096")
    print("  - num_steps (coarse) = 128")
    print("  - upsample_steps (fine) = 128")
    print("  - out_dim_color = 1 (grayscale/events)")
    
    # Calcoli rapidi
    batch_size = 4096
    num_rays = batch_size * 2
    num_steps = 128
    upsample_steps = 128
    
    profiler = FlopProfilerWithRendering(None)
    
    # Confronto
    comparator = RenderingFlopsComparison(None)
    comparator.compare_rendering_methods(batch_size, num_steps, upsample_steps)
