#!/usr/bin/env python3
"""
Inference Latency Analysis for ReIDNet Model
============================================

Measure inference latency for the ReIDNet model.

Usage:
    python tools/latency.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
"""

import argparse
import time
import torch
import numpy as np
from mmcv import Config
from mmdet3d.models import build_model

def create_dummy_inputs(batch_size=1, num_points=1024, num_classes=20):
    """Create dummy inputs for the ReIDNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy point clouds
    sparse_1 = [torch.randn(num_points, 3).to(device) for _ in range(batch_size)]
    sparse_2 = [torch.randn(num_points, 3).to(device) for _ in range(batch_size)]
    
    # Create dummy labels and IDs
    label_1 = [torch.randint(0, num_classes, (1,)).to(device) for _ in range(batch_size)]
    label_2 = [torch.randint(0, num_classes, (1,)).to(device) for _ in range(batch_size)]
    id_1 = [torch.randint(0, 100, (1,)).to(device) for _ in range(batch_size)]
    id_2 = [torch.randint(0, 100, (1,)).to(device) for _ in range(batch_size)]
    
    return {
        'sparse_1': sparse_1,
        'sparse_2': sparse_2,
        'label_1': label_1,
        'label_2': label_2,
        'id_1': id_1,
        'id_2': id_2,
    }

def create_test_inputs(batch_size=1, num_points=1024, num_classes=20):
    """Create dummy inputs for the ReIDNet model (testing mode)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy point clouds
    sparse_1 = [torch.randn(num_points, 3).to(device) for _ in range(batch_size)]
    sparse_2 = [torch.randn(num_points, 3).to(device) for _ in range(batch_size)]
    
    # Create dummy labels and IDs
    label_1 = [torch.randint(0, num_classes, (1,)).to(device) for _ in range(batch_size)]
    label_2 = [torch.randint(0, num_classes, (1,)).to(device) for _ in range(batch_size)]
    id_1 = [torch.randint(0, 100, (1,)).to(device) for _ in range(batch_size)]
    id_2 = [torch.randint(0, num_classes, (1,)).to(device) for _ in range(batch_size)]
    
    # Create dummy size and visibility tensors (required for testing)
    size_1 = [torch.tensor([num_points]).to(device) for _ in range(batch_size)]
    size_2 = [torch.tensor([num_points]).to(device) for _ in range(batch_size)]
    vis_1 = [torch.ones(1).to(device) for _ in range(batch_size)]
    vis_2 = [torch.ones(1).to(device) for _ in range(batch_size)]
    
    return {
        'sparse_1': sparse_1,
        'sparse_2': sparse_2,
        'label_1': label_1,
        'label_2': label_2,
        'id_1': id_1,
        'id_2': id_2,
        'size_1': size_1,
        'size_2': size_2,
        'vis_1': vis_1,
        'vis_2': vis_2,
    }

def measure_latency(model, inputs, warmup=20, iters=100):
    """Measure inference latency."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(**inputs, return_loss=False)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure latency
        t0 = time.time()
        for _ in range(iters):
            _ = model(**inputs, return_loss=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        dt = time.time() - t0
    
    return (dt / iters) * 1000  # Convert to milliseconds

def main():
    parser = argparse.ArgumentParser(description='Measure ReIDNet Latency')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-points', type=int, default=1024, help='Number of points per cloud')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Number of measurement iterations')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = Config.fromfile(args.config)
    
    # Build model
    model = build_model(cfg.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"=== Inference Latency Analysis ===")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measurement iterations: {args.iters}")
    print()
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    latencies = []
    
    for bs in batch_sizes:
        print(f"Testing batch size {bs}...")
        inputs = create_test_inputs(bs, args.num_points)
        latency = measure_latency(model, inputs, args.warmup, args.iters)
        latencies.append(latency)
        print(f"  Latency: {latency:.2f} ms")
    
    print("\n=== Results Summary ===")
    print("Batch Size | Latency (ms) | Throughput (samples/sec)")
    print("-" * 50)
    for i, bs in enumerate(batch_sizes):
        throughput = bs / (latencies[i] / 1000)  # samples per second
        print(f"{bs:9d} | {latencies[i]:11.2f} | {throughput:20.2f}")
    
    # Statistical analysis
    print(f"\n=== Statistical Analysis ===")
    print(f"Best latency (batch=1): {latencies[0]:.2f} ms")
    print(f"Best throughput: {batch_sizes[0] / (latencies[0] / 1000):.2f} samples/sec")
    
    # Memory usage (optional)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        inputs_mem = create_dummy_inputs(1, args.num_points)
        _ = model(**inputs_mem)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak GPU memory: {peak_memory:.2f} MB")

if __name__ == '__main__':
    main()
