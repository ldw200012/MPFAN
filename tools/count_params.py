#!/usr/bin/env python3
"""
Parameter Count Analysis for ReIDNet Model
=========================================

Simple script to count parameters in the ReIDNet model.

Usage:
    python tools/count_params.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
"""

import argparse
import torch
from mmcv import Config
from mmdet3d.models import build_model

def count_params(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='Count ReIDNet Parameters')
    parser.add_argument('--config', required=True, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = Config.fromfile(args.config)
    
    # Build model
    model = build_model(cfg.model)
    
    # Count parameters
    total_params = count_params(model)
    
    print(f"=== Parameter Count Analysis ===")
    print(f"Config: {args.config}")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameters (M): {total_params / 1e6:.2f}M")
    
    # Detailed breakdown
    print("\n=== Parameter Breakdown ===")
    total_counted = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                print(f"{name}: {params:,} ({params / 1e6:.2f}M)")
                total_counted += params
    
    print(f"\nTotal counted: {total_counted:,} ({total_counted / 1e6:.2f}M)")
    print(f"Verification: {'✓' if total_counted == total_params else '✗'}")

if __name__ == '__main__':
    main()
