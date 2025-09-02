# ReIDNet Complexity Analysis Tools

This directory contains tools for performing comprehensive complexity analysis on the ReIDNet model for testing/inference.

## Overview

The complexity analysis includes three main components as specified in the instructions:

1. **A. Parameter Count** - Model size in millions of parameters
2. **B. FLOPs/MACs** - Computational complexity for fixed input (1024 points)
3. **C. Inference Latency** - Runtime performance in milliseconds

## Files

- `complexity_analysis.py` - Comprehensive analysis script (all three components)
- `count_params.py` - Simple parameter counting script
- `flops_thop.py` - FLOPs calculation using custom estimation
- `latency.py` - Inference latency measurement
- `requirements_analysis.txt` - Dependencies for analysis tools

## Installation

Install the required dependencies:

```bash
pip install -r tools/requirements_analysis.txt
```

## Usage

### 1. Comprehensive Analysis (All Components)

```bash
python tools/complexity_analysis.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```

Options:
- `--batch-size`: Batch size for analysis (default: 1)
- `--num-points`: Number of points per cloud (default: 1024)

### 2. Parameter Count Only

```bash
python tools/count_params.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```

### 3. FLOPs Analysis Only

```bash
python tools/flops_thop.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```

Options:
- `--batch-size`: Batch size (default: 1)
- `--num-points`: Number of points per cloud (default: 1024)

### 4. Latency Analysis Only

```bash
python tools/latency.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```

Options:
- `--batch-size`: Batch size (default: 1)
- `--num-points`: Number of points per cloud (default: 1024)
- `--warmup`: Number of warmup iterations (default: 20)
- `--iters`: Number of measurement iterations (default: 100)

## Example Output

### Parameter Count
```
=== Parameter Count Analysis ===
Config: configs_reid/reid_nuscenes_pts/base_mpfan.py
Total parameters: 1,234,567
Parameters (M): 1.23M
```

### FLOPs Analysis
```
=== FLOPs Analysis ===
Config: configs_reid/reid_nuscenes_pts/base_mpfan.py
Batch size: 1
Points per cloud: 1024

Params: 1,234,567
Params (M): 1.23M
Estimated MACs: 2,345,678,901
Estimated MACs (G): 2.35G
Estimated FLOPs (G): 4.69G
Note: FLOPs are estimated based on model architecture and parameters
```

### Latency Analysis
```
=== Inference Latency Analysis ===
Config: configs_reid/reid_nuscenes_pts/base_mpfan.py
Device: cuda:0
Warmup iterations: 20
Measurement iterations: 100

Testing batch size 1...
  Latency: 15.23 ms
Testing batch size 2...
  Latency: 28.45 ms
Testing batch size 4...
  Latency: 52.67 ms
Testing batch size 8...
  Latency: 98.89 ms

=== Results Summary ===
Batch Size | Latency (ms) | Throughput (samples/sec)
--------------------------------------------------
        1 |        15.23 |                 65.66
        2 |        28.45 |                 70.30
        4 |        52.67 |                 75.94
        8 |        98.89 |                 80.90
```

## Supported Configurations

The analysis tools work with all ReIDNet configurations:

- `configs_reid/reid_nuscenes_pts/base_mpfan.py`
- `configs_reid/reid_nuscenes_pts/base_pointnet.py`
- `configs_reid/reid_nuscenes_pts/base_dgcnn.py`
- `configs_reid/reid_nuscenes_pts/base_pointnext.py`
- `configs_reid/reid_nuscenes_pts/base_deepgcn.py`
- `configs_reid/reid_waymo_pts/base_*.py`

## Notes

1. **GPU Memory**: The analysis includes optional GPU memory measurement if CUDA is available.

2. **Warmup**: Latency measurements include warmup iterations to ensure consistent performance.

3. **Batch Size Scaling**: The comprehensive analysis tests multiple batch sizes to understand scaling behavior.

4. **FLOPs Calculation**: Uses custom estimation based on model architecture and parameters for reliable FLOPs/MACs calculation.

5. **Device Support**: Automatically detects and uses GPU if available, falls back to CPU.

## Troubleshooting

### CUDA Memory Issues
If you encounter CUDA memory issues, try:
- Reducing batch size
- Reducing number of points
- Using CPU instead of GPU

### Model Loading Issues
Ensure the config file path is correct and the model can be built successfully.

## Performance Tips

1. **For accurate latency measurements**: Use multiple iterations and warmup
2. **For memory analysis**: Monitor GPU memory usage during inference
3. **For scaling analysis**: Test different batch sizes and point counts
4. **For comparison**: Use consistent input sizes across different models
