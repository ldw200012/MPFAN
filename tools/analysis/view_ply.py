#!/usr/bin/env python3
"""
PLY Point Cloud Viewer

This script can visualize PLY files in two ways:
1. Automatically scan a parent folder for subfolders with the same number of PLY files and visualize them in subplots
2. Visualize specific PLY files provided as command line arguments

Features:
- Automatic subfolder detection and grouping by PLY file count
- Subplot visualization for easy comparison
- Image saving mode for batch processing
- Configurable subfolder ordering (pointnet, pointnext, dgcnn, deepgcn, pointtransformer, calmnet)

Usage examples:
    # View all PLY files from subfolders with the same file count in subplots
    python view_ply.py --folder /path/to/parent/folder
    
    # Save all visualizations as image files (no display)
    python view_ply.py --folder /path/to/parent/folder --save-images
    
    # Save images to custom output directory
    python view_ply.py --folder /path/to/parent/folder --save-images --output-dir /path/to/output
    
    # View specific PLY files
    python view_ply.py --ply file1.ply file2.ply
    
    # Auto-close visualizations after 3 seconds
    python view_ply.py --auto-close
"""

import argparse
import os
from typing import Tuple

import numpy as np


def read_ascii_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an ASCII PLY file with header format written by feature_heatmap.py
    and return (points_xyz [N,3], colors_rgb01 [N,3]).
    """
    with open(path, 'r') as f:
        header = []
        line = f.readline().strip()
        if line != 'ply':
            raise ValueError(f"{path} is not a PLY file (missing 'ply' magic)")
        header.append(line)

        num_vertices = None
        # Read header
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Unexpected EOF while reading PLY header')
            line = line.strip()
            header.append(line)
            if line.startswith('element vertex'):
                parts = line.split()
                num_vertices = int(parts[-1])
            if line == 'end_header':
                break

        if num_vertices is None:
            raise ValueError('PLY header missing vertex count')

        # Read vertex lines: x y z r g b
        points = np.zeros((num_vertices, 3), dtype=np.float32)
        colors = np.zeros((num_vertices, 3), dtype=np.float32)
        for i in range(num_vertices):
            line = f.readline()
            if not line:
                raise ValueError('Unexpected EOF while reading vertices')
            parts = line.strip().split()
            if len(parts) < 6:
                raise ValueError(f'Vertex line {i} malformed: {parts}')
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            points[i] = [x, y, z]
            colors[i] = [r, g, b]

        colors = np.clip(colors / 255.0, 0.0, 1.0)
        return points, colors


def visualize(points_xyz: np.ndarray, colors_rgb01: np.ndarray, title: str = None, point_size: float = 2.0, save_path: str = None):
    """Visualize single point cloud using Matplotlib and save to file."""
    import matplotlib.pyplot as plt  # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c=colors_rgb01, s=point_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    # Equal aspect ratio
    max_range = (points_xyz.max(axis=0) - points_xyz.min(axis=0)).max()
    centers = points_xyz.mean(axis=0)
    mins = centers - max_range / 2
    maxs = centers + max_range / 2
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    plt.tight_layout()
    
    if save_path:
        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()


def visualize_multiple_subplots(point_clouds: list, titles: list, point_size: float = 2.0, save_path: str = None):
    """Visualize multiple point clouds in subplots using Matplotlib and save to file."""
    import matplotlib.pyplot as plt  # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    n_plots = len(point_clouds)
    
    # Calculate subplot layout (3x3 grid for 7 plots)
    if n_plots <= 3:
        cols = n_plots
        rows = 1
    elif n_plots <= 6:
        cols = 3
        rows = 2
    else:
        cols = 3
        rows = 3
    
    fig = plt.figure(figsize=(6*cols, 5*rows))
    
    for i, ((points_xyz, colors_rgb01), title) in enumerate(zip(point_clouds, titles)):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Plot point cloud
        ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                  c=colors_rgb01, s=point_size, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=10)
        
        # Equal aspect ratio and consistent view
        max_range = (points_xyz.max(axis=0) - points_xyz.min(axis=0)).max()
        centers = points_xyz.mean(axis=0)
        mins = centers - max_range / 2
        maxs = centers + max_range / 2
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        
        # Set consistent view angle for all subplots
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved subplot visualization: {save_path}")
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='View PLY point cloud saved by feature_heatmap.py')
    parser.add_argument('--ply', nargs='*', help='Path(s) to specific PLY file(s) (optional)')
    parser.add_argument('--folder', default='/home/dongwooklee1201/morin/Research/Masters_Dissertation/LiDAR_Re-Identification/MPFAN/runs/pc_feature_viz_cls', 
                       help='Parent folder to scan for subfolders with PLY files')
    parser.add_argument('--point-size', type=float, default=25.0, help='Point size for visualization')
    parser.add_argument('--auto-close', action='store_true', help='Automatically close each visualization after 3 seconds')
    parser.add_argument('--save-images', action='store_true', help='Save visualizations as image files instead of displaying them')
    parser.add_argument('--output-dir', default='runs/visualizations', help='Output directory for saved image files')
    args = parser.parse_args()

    # If specific PLY files are provided, use those
    if args.ply:
        ply_files = args.ply
        for i, ply_path in enumerate(ply_files):
            if not os.path.isfile(ply_path):
                print(f"[warn] File not found: {ply_path}")
                continue
            
            print(f"\n[{i+1}/{len(ply_files)}] Visualizing: {os.path.basename(ply_path)}")
            
            try:
                pts, cols = read_ascii_ply(ply_path)
                print(f"  Points: {pts.shape[0]}, Colors: {cols.shape}")
            except Exception as e:
                print(f"[error] Failed to read {ply_path}: {e}")
                continue
            
            title = os.path.basename(ply_path)
            if args.save_images:
                # Generate save path for this visualization
                file_name_base = os.path.splitext(title)[0]
                # Try to infer class name from directory structure: .../<class>/<file>
                inferred_class = os.path.basename(os.path.dirname(ply_path))
                save_path = os.path.join(args.output_dir, inferred_class, "single_files", f"{file_name_base}.png")
                visualize(pts, cols, title=title, point_size=args.point_size, save_path=save_path)
            else:
                visualize(pts, cols, title=title, point_size=args.point_size)
                
                # If auto-close is enabled, wait a bit before closing
                if args.auto_close:
                    import time
                    time.sleep(3)
    
    # Single folder mode (enhanced to support model/class subfolders)
    else:
        # Scan the parent folder for subfolders and group them by PLY file count
        if not os.path.isdir(args.folder):
            print(f"[error] Folder not found: {args.folder}")
            return
        
        print(f"Scanning {args.folder} for model/class subfolders with PLY files...")

        # desired order of model folders
        desired_order = ['pointnet', 'pointnext', 'dgcnn', 'deepgcn', 'pointtransformer', 'calmnet']

        # Map: model -> class -> [ply paths]
        model_class_plys = {}
        models_found = []
        for model_name in os.listdir(args.folder):
            model_path = os.path.join(args.folder, model_name)
            if not os.path.isdir(model_path):
                continue
            models_found.append(model_name)
            class_map = {}
            for class_name in os.listdir(model_path):
                class_path = os.path.join(model_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                ply_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.ply')]
                ply_files.sort()
                if ply_files:
                    class_map[class_name] = ply_files
                    print(f"  Found: {model_name}/{class_name} with {len(ply_files)} PLY files")
            if class_map:
                model_class_plys[model_name] = class_map

        if not model_class_plys:
            print(f"[warn] No model/class subfolders with PLY files found in {args.folder}")
            return

        # Determine all classes present across models
        all_classes = set()
        for class_map in model_class_plys.values():
            all_classes.update(class_map.keys())

        # Sort models per desired order
        def model_sort_key(m):
            try:
                return desired_order.index(m.lower())
            except ValueError:
                return len(desired_order)

        # For each class, visualize across all models that have that class
        for class_name in sorted(all_classes):
            models_with_class = [m for m in model_class_plys.keys() if class_name in model_class_plys[m]]
            if len(models_with_class) < 2:
                print(f"\nSkipping class '{class_name}' (available in fewer than 2 models)")
                continue

            models_sorted = sorted(models_with_class, key=model_sort_key)
            min_files = min(len(model_class_plys[m][class_name]) for m in models_sorted)
            print(f"\nProcessing class '{class_name}' with {min_files} files across models: {', '.join(models_sorted)}")

            for file_idx in range(min_files):
                print(f"  [{file_idx+1}/{min_files}] Processing PLY files for class '{class_name}'...")

                point_clouds = []
                titles = []
                last_file_name = None

                for model_name in models_sorted:
                    ply_path = model_class_plys[model_name][class_name][file_idx]
                    file_name = os.path.basename(ply_path)
                    last_file_name = file_name
                    try:
                        pts, cols = read_ascii_ply(ply_path)
                        print(f"    {model_name}: {file_name} - Points: {pts.shape[0]}, Colors: {cols.shape}")
                        point_clouds.append((pts, cols))
                        titles.append(f"{model_name}\n{file_name}")
                    except Exception as e:
                        print(f"[error] Failed to read {ply_path}: {e}")
                        point_clouds.append((np.array([]), np.array([])))
                        titles.append(f"{model_name}\n{file_name} (ERROR)")

                if point_clouds:
                    try:
                        if args.save_images:
                            file_name_base = os.path.splitext(last_file_name or f"idx_{file_idx+1:03d}")[0]
                            # Save under output_dir/<class_name>/...
                            save_path = os.path.join(args.output_dir, class_name, f"group_models_{len(models_sorted)}", f"file_{file_idx+1:03d}_{file_name_base}.png")
                            visualize_multiple_subplots(point_clouds, titles, point_size=args.point_size, save_path=save_path)
                        else:
                            visualize_multiple_subplots(point_clouds, titles, point_size=args.point_size)
                            if args.auto_close:
                                import time
                                time.sleep(3)
                    except Exception as e:
                        print(f"[error] Failed to visualize subplots: {e}")


if __name__ == '__main__':
    main()


