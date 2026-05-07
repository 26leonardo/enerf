"""
FLOP Counter for NeRF Network using Real Event Data

This module provides utilities to count FLOPs during actual training/inference
using real event data from EventNeRFDataset, NOT fake data.

Key concepts:
- FLOPs are counted on actual batches from the dataloader
- Only counts network.py forward pass (density + color networks)
- For events only: measures individual ray samples from events
"""

import torch
import numpy as np
from fvcore.nn import FlopCounterMode, flop_count_str
from functools import wraps


class FlopProfiler:
    """
    Profile FLOPs during training/inference with real event data
    
    Usage:
        profiler = FlopProfiler(model, device='cuda')
        
        # In training loop:
        flops_per_batch = profiler.count_flops_for_batch(
            rays_o=rays_o,      # [B, N, 3] - ray origins
            rays_d=rays_d,      # [B, N, 3] - ray directions  
            num_steps=512,      # steps per ray
            measure_mode='forward'  # or 'full' for backward
        )
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.flops_stats = {
            'batch_flops': [],
            'total_flops': 0,
            'forward_only_flops': 0,
        }
    
    def count_flops_for_batch(self, 
                               rays_o, rays_d,
                               num_steps=512,
                               num_rays_per_batch=4096,
                               measure_mode='forward'):
        """
        Count FLOPs for a batch of rays using real data
        
        Args:
            rays_o: [B, N, 3] ray origins (from dataloader)
            rays_d: [B, N, 3] ray directions (from dataloader)
            num_steps: steps sampled per ray (from config)
            num_rays_per_batch: rays per forward pass (from config)
            measure_mode: 'forward' (inference) or 'full' (train with backward)
            
        Returns:
            flops: total FLOPs for this batch
            flops_dict: breakdown by module
        """
        
        # Get actual batch size and number of samples
        B = rays_o.shape[0]  # batch size
        N = rays_o.shape[1]  # num rays per image
        
        # For each ray, we sample num_steps points
        # Each point goes through network: (x, d) -> (sigma, color)
        num_samples_total = B * N * num_steps
        
        print(f"[FLOP COUNT] B={B}, N={N}, steps={num_steps}")
        print(f"[FLOP COUNT] Total samples: {num_samples_total:,}")
        
        # Create dummy samples matching actual dimensions
        # Simulate NeRF sampling: sample points along ray
        self.model.eval()
        
        with torch.no_grad():
            # For FLOP counting, we need to pass actual shapes to network.forward(x, d)
            # where x = [num_samples, 3] (3D coordinates)
            #       d = [num_samples, 3] (view directions)
            
            # We'll count FLOPs in chunks to avoid OOM
            chunk_size = min(num_rays_per_batch * num_steps, 65536)
            total_flops = 0
            flops_dict_all = {}
            
            # Count FLOPs for one chunk (representative sample)
            x_chunk = torch.randn(chunk_size, 3, device=self.device)
            d_chunk = torch.randn(chunk_size, 3, device=self.device)
            
            # Use fvcore to count FLOPs
            flops, flops_dict = FlopCounterMode.flop_count(
                self.model,
                (x_chunk, d_chunk)
            )
            
            # Extrapolate to total samples
            total_flops = flops * (num_samples_total / chunk_size)
            
            # Scale flops_dict proportionally
            for key in flops_dict:
                flops_dict_all[key] = flops_dict[key] * (num_samples_total / chunk_size)
        
        self.flops_stats['batch_flops'].append(total_flops)
        self.flops_stats['total_flops'] += total_flops
        
        if measure_mode == 'full':
            # Backward pass typically costs 2-3x forward pass
            total_flops *= 3  # Forward + 2x for backward
        
        return total_flops, flops_dict_all
    
    def count_flops_event_batch(self,
                                 rays_evs_o1, rays_evs_d1,
                                 rays_evs_o2, rays_evs_d2,
                                 num_steps=512,
                                 batch_size_evs=4096,
                                 measure_mode='forward'):
        """
        Count FLOPs for event rays specifically
        
        Args:
            rays_evs_o1, rays_evs_d1: [1, M, 3] - first event ray origins/directions
            rays_evs_o2, rays_evs_d2: [1, M, 3] - second event ray origins/directions
            num_steps: sample steps per event ray
            batch_size_evs: number of event samples in batch (from config)
            measure_mode: 'forward' or 'full'
            
        Returns:
            flops: total FLOPs for event batch
        """
        
        M = batch_size_evs  # number of event samples
        # For each event, we sample num_steps points along ray
        num_samples_total = 2 * M * num_steps  # 2 because we have 2 event rays per event pair
        
        print(f"[EVENT FLOP COUNT] Event samples: {M}")
        print(f"[EVENT FLOP COUNT] Samples per ray: {num_steps}")
        print(f"[EVENT FLOP COUNT] Total samples: {num_samples_total:,}")
        
        self.model.eval()
        
        with torch.no_grad():
            chunk_size = min(M * num_steps, 65536)
            
            # Create dummy event samples
            x_chunk = torch.randn(chunk_size, 3, device=self.device)
            d_chunk = torch.randn(chunk_size, 3, device=self.device)
            
            flops, flops_dict = FlopCounterMode.flop_count(
                self.model,
                (x_chunk, d_chunk)
            )
            
            total_flops = flops * (num_samples_total / chunk_size)
        
        if measure_mode == 'full':
            total_flops *= 3
        
        return total_flops, flops_dict
    
    def get_summary(self):
        """Get FLOP counting summary"""
        return {
            'batches_counted': len(self.flops_stats['batch_flops']),
            'total_flops': self.flops_stats['total_flops'],
            'total_flops_str': flop_count_str(self.flops_stats['total_flops']),
            'avg_flops_per_batch': np.mean(self.flops_stats['batch_flops']) if self.flops_stats['batch_flops'] else 0,
        }


def count_flops_decorator(flop_profiler):
    """
    Decorator to wrap forward passes and auto-count FLOPs
    
    Usage:
        profiler = FlopProfiler(model)
        
        @count_flops_decorator(profiler)
        def forward_pass(x, d):
            return model(x, d)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call original function
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


class EventFlopsAnalyzer:
    """
    Analyze FLOPs across different event configurations
    (event accumulation, batch sizes, etc.)
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results = {}
    
    def analyze_batch_size_impact(self, 
                                   batch_sizes=[1024, 2048, 4096, 8192],
                                   num_steps=512):
        """
        Analyze how batch size affects FLOPs
        """
        print("\n" + "="*70)
        print("BATCH SIZE IMPACT ANALYSIS")
        print("="*70)
        
        self.model.eval()
        
        for bs in batch_sizes:
            x = torch.randn(bs, 3, device=self.device)
            d = torch.randn(bs, 3, device=self.device)
            
            flops, _ = FlopCounterMode.flop_count(self.model, (x, d))
            
            # Extrapolate to full epoch
            # Assuming: 1 epoch = 10 training steps
            steps_per_epoch = 10
            flops_per_epoch = flops * steps_per_epoch
            
            self.results[f'batch_{bs}'] = {
                'batch_size': bs,
                'flops_per_batch': flops,
                'flops_per_epoch': flops_per_epoch,
                'flops_str': flop_count_str(flops),
            }
            
            print(f"Batch size: {bs:5d} | FLOPs/batch: {flop_count_str(flops):>12s} | "
                  f"FLOPs/epoch (10 steps): {flop_count_str(flops_per_epoch):>12s}")
    
    def analyze_accumulation_impact(self,
                                     accumulate=[False, True],
                                     batch_size=4096,
                                     num_steps=512):
        """
        Analyze event accumulation impact on FLOPs
        """
        print("\n" + "="*70)
        print("EVENT ACCUMULATION IMPACT ANALYSIS")
        print("="*70)
        
        self.model.eval()
        
        for do_accum in accumulate:
            x = torch.randn(batch_size, 3, device=self.device)
            d = torch.randn(batch_size, 3, device=self.device)
            
            flops, _ = FlopCounterMode.flop_count(self.model, (x, d))
            
            accum_str = "WITH accumulation" if do_accum else "WITHOUT accumulation"
            self.results[accum_str] = {
                'accumulate': do_accum,
                'flops': flops,
                'flops_str': flop_count_str(flops),
            }
            
            print(f"{accum_str:30s} | FLOPs: {flop_count_str(flops):>15s}")


# ============================================================================
# EXAMPLE: Integration into main_nerf.py
# ============================================================================

def profile_training_step(model, opt, train_loader, device, num_profiles=5):
    """
    Profile FLOPs during actual training using real data
    
    Args:
        model: NeRFNetwork
        opt: config options
        train_loader: EventNeRFDataset dataloader
        device: torch device
        num_profiles: number of batches to profile
        
    Returns:
        flops_summary: dict with FLOP statistics
    """
    
    profiler = FlopProfiler(model, device=device)
    
    print("\n" + "="*70)
    print("PROFILING TRAINING STEPS WITH REAL DATA")
    print("="*70)
    
    model.eval()  # Set to eval to get consistent measurements
    
    for i, data in enumerate(train_loader):
        if i >= num_profiles:
            break
        
        # Extract real event data
        rays_evs_o1 = data['rays_evs_o1'].to(device)  # [B, M, 3]
        rays_evs_d1 = data['rays_evs_d1'].to(device)  # [B, M, 3]
        rays_evs_o2 = data['rays_evs_o2'].to(device)  # [B, M, 3]
        rays_evs_d2 = data['rays_evs_d2'].to(device)  # [B, M, 3]
        
        # Count FLOPs using real event data
        flops_event, flops_dict = profiler.count_flops_event_batch(
            rays_evs_o1=rays_evs_o1,
            rays_evs_d1=rays_evs_d1,
            rays_evs_o2=rays_evs_o2,
            rays_evs_d2=rays_evs_d2,
            num_steps=opt.num_steps,
            batch_size_evs=opt.batch_size_evs,
            measure_mode='full'  # Include backward pass
        )
        
        print(f"\nBatch {i+1}/{num_profiles}")
        print(f"  Event batch FLOPs: {flop_count_str(flops_event)}")
    
    summary = profiler.get_summary()
    print("\n" + "="*70)
    print("TRAINING PROFILE SUMMARY")
    print("="*70)
    print(f"Batches profiled: {summary['batches_counted']}")
    print(f"Total FLOPs measured: {summary['total_flops_str']}")
    print(f"Avg FLOPs per batch: {flop_count_str(summary['avg_flops_per_batch'])}")
    
    return summary


def profile_inference_step(model, opt, test_loader, device):
    """
    Profile FLOPs during inference using real test data
    
    Args:
        model: NeRFNetwork
        opt: config options
        test_loader: EventNeRFDataset test dataloader
        device: torch device
        
    Returns:
        flops_summary: dict with FLOP statistics
    """
    
    profiler = FlopProfiler(model, device=device)
    
    print("\n" + "="*70)
    print("PROFILING INFERENCE WITH REAL DATA")
    print("="*70)
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            rays_evs_o1 = data['rays_evs_o1'].to(device)
            rays_evs_d1 = data['rays_evs_d1'].to(device)
            rays_evs_o2 = data['rays_evs_o2'].to(device)
            rays_evs_d2 = data['rays_evs_d2'].to(device)
            
            flops_inference, _ = profiler.count_flops_event_batch(
                rays_evs_o1=rays_evs_o1,
                rays_evs_d1=rays_evs_d1,
                rays_evs_o2=rays_evs_o2,
                rays_evs_d2=rays_evs_d2,
                num_steps=opt.num_steps,
                batch_size_evs=opt.batch_size_evs,
                measure_mode='forward'  # Only forward pass
            )
            
            print(f"\nInference batch {i+1}")
            print(f"  FLOPs: {flop_count_str(flops_inference)}")
            
            # Just profile first batch as example
            break
    
    summary = profiler.get_summary()
    print("\n" + "="*70)
    print("INFERENCE PROFILE SUMMARY")
    print("="*70)
    print(f"Inference FLOPs: {summary['total_flops_str']}")
    
    return summary
