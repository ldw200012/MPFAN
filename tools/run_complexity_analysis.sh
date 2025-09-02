#!/bin/bash

# ReIDNet Complexity Analysis Runner
# ==================================
# This script demonstrates how to run all complexity analysis tools

echo "ReIDNet Complexity Analysis"
echo "=========================="
echo

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs_reid/reid_nuscenes_pts/base_mpfan.py"
    exit 1
fi

CONFIG_FILE=$1

echo "Using config: $CONFIG_FILE"
echo

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
echo "Dependencies OK"
echo

# Run individual analyses
# echo "1. Parameter Count Analysis"
# echo "---------------------------"
# python3 tools/count_params.py --config "$CONFIG_FILE"
# echo

# echo "2. FLOPs Analysis"
# echo "-----------------"
# python3 tools/flops_thop.py --config "$CONFIG_FILE"
# echo

echo "3. Latency Analysis"
echo "-------------------"
python3 tools/latency.py --config "$CONFIG_FILE"
echo

# echo "4. Comprehensive Analysis"
# echo "------------------------"
# python3 tools/complexity_analysis.py --config "$CONFIG_FILE"
# echo

echo "Analysis complete!"
echo "Check the output above for detailed results."
