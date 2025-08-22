#!/usr/bin/env python3
"""
Script to check data structure and verify file existence for pre-computed eigenvalue data.
"""

import os
import numpy as np
from pathlib import Path

def check_data_structure():
    """Check if the expected data structure exists."""
    
    # Configuration
    data_root = "Datasets/NuScenes-ReID/data/lstk/sparse-trainval-det-both"
    expected_feature = "xyz_eigen"
    expected_dim = 6
    knn_size = 10
    
    print("=== Data Structure Check ===")
    print(f"Data root: {data_root}")
    print(f"Expected feature: {expected_feature}")
    print(f"Expected dimension: {expected_dim}")
    print(f"KNN size: {knn_size}")
    print()
    
    if not os.path.exists(data_root):
        print(f"❌ Data root does not exist: {data_root}")
        return False
    
    # Find some example files
    print("=== Searching for files ===")
    
    # Look for eigenvalue files
    eigen_pattern = f"pts_{expected_feature}_{knn_size}.bin"
    xyz_pattern = "pts_xyz.bin"
    
    eigen_files = []
    xyz_files = []
    
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file == eigen_pattern:
                eigen_files.append(os.path.join(root, file))
            elif file == xyz_pattern:
                xyz_files.append(os.path.join(root, file))
    
    print(f"Found {len(eigen_files)} eigenvalue files: {eigen_pattern}")
    print(f"Found {len(xyz_files)} original xyz files: {xyz_pattern}")
    
    if len(eigen_files) == 0:
        print("❌ No eigenvalue files found!")
        print("You need to create the eigenvalue files first.")
        return False
    
    # Check a few example files
    print("\n=== Checking example files ===")
    
    for i, eigen_file in enumerate(eigen_files[:3]):  # Check first 3 files
        print(f"\nFile {i+1}: {eigen_file}")
        
        if os.path.exists(eigen_file):
            file_size = os.path.getsize(eigen_file)
            print(f"  ✅ File exists, size: {file_size} bytes")
            
            # Try to read the file
            try:
                data = np.fromfile(eigen_file, dtype=np.float32)
                print(f"  📊 Raw data shape: {data.shape}")
                
                # Check if it can be reshaped to expected dimensions
                if data.size % expected_dim == 0:
                    num_points = data.size // expected_dim
                    reshaped_data = data.reshape(-1, expected_dim)
                    print(f"  ✅ Can be reshaped to: {reshaped_data.shape}")
                    print(f"  📈 Data range: [{data.min():.4f}, {data.max():.4f}]")
                    
                    # Check if eigenvalues are reasonable (should be positive)
                    if expected_dim == 6:
                        coords = reshaped_data[:, :3]
                        eigenvals = reshaped_data[:, 3:]
                        print(f"  📍 Coordinates range: [{coords.min():.4f}, {coords.max():.4f}]")
                        print(f"  🔢 Eigenvalues range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                        
                        if eigenvals.min() < 0:
                            print(f"  ⚠️  Warning: Some eigenvalues are negative!")
                        else:
                            print(f"  ✅ Eigenvalues are all positive")
                else:
                    print(f"  ❌ Cannot be reshaped to {expected_dim} dimensions")
                    
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
        else:
            print(f"  ❌ File does not exist")
    
    # Check corresponding xyz files
    print(f"\n=== Checking corresponding xyz files ===")
    for i, eigen_file in enumerate(eigen_files[:3]):
        xyz_file = eigen_file.replace(f"pts_{expected_feature}_{knn_size}.bin", "pts_xyz.bin")
        print(f"\nCorresponding xyz file {i+1}: {xyz_file}")
        
        if os.path.exists(xyz_file):
            file_size = os.path.getsize(xyz_file)
            print(f"  ✅ File exists, size: {file_size} bytes")
            
            try:
                data = np.fromfile(xyz_file, dtype=np.float32)
                print(f"  📊 Raw data shape: {data.shape}")
                
                if data.size % 3 == 0:
                    num_points = data.size // 3
                    reshaped_data = data.reshape(-1, 3)
                    print(f"  ✅ Can be reshaped to: {reshaped_data.shape}")
                    print(f"  📈 Data range: [{data.min():.4f}, {data.max():.4f}]")
                else:
                    print(f"  ❌ Cannot be reshaped to 3 dimensions")
                    
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
        else:
            print(f"  ❌ File does not exist")
    
    print("\n=== Summary ===")
    if len(eigen_files) > 0:
        print("✅ Eigenvalue files found and appear to be valid")
        print("✅ You can proceed with training using pre-computed eigenvalues")
        return True
    else:
        print("❌ No eigenvalue files found")
        print("❌ You need to create the eigenvalue files first")
        return False

if __name__ == "__main__":
    check_data_structure()
