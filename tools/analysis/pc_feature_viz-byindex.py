#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from mmengine.config import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def _to_device(v: Any, device: torch.device) -> Any:
    if hasattr(v, 'data'):
        v = v.data
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, list):
        out = []
        for item in v:
            if hasattr(item, 'data'):
                item = item.data
            out.append(item.to(device) if torch.is_tensor(item) else item)
        return out
    return v


def _build_inputs(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for k, v in sample.items():
        inputs[k] = _to_device(v, device)
    return inputs


def _norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x - x.min()
    den = (x.max() + eps)
    return (x / den) if den > 0 else np.zeros_like(x)


def _extract_xyz_from_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Extract XYZ coordinates from point cloud data.
    Handles both 3-channel (x,y,z) and 6-channel (x,y,z,intensity,elongation,timestamp) formats.
    """
    if points.shape[1] == 3:
        # Already in XYZ format
        return points
    elif points.shape[1] == 6:
        # Extract first 3 channels (x, y, z)
        return points[:, :3]
    else:
        # For other channel counts, assume first 3 are XYZ
        print(f"[warn] Unexpected point cloud channels: {points.shape[1]}, using first 3 as XYZ")
        return points[:, :3]


def _save_ply(points_xyz: np.ndarray, rgb01: np.ndarray, out_path: str):
    n = points_xyz.shape[0]
    rgb255 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(n):
            x, y, z = points_xyz[i]
            r, g, b = rgb255[i]
            f.write(f'{x} {y} {z} {int(r)} {int(g)} {int(b)}\n')


def _o3d_show(points_xyz: np.ndarray, rgb01: np.ndarray, title: str = None, point_size: float = 2.0):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb01.astype(np.float64))
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title or 'pc_feature_viz')
        vis.add_geometry(pcd)
        ro = vis.get_render_option()
        ro.point_size = float(point_size)
        vis.run()
        vis.destroy_window()
    except Exception:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c=rgb01, s=point_size)
        plt.show()


def _features_from_model(model: torch.nn.Module, inputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Try to get per-point features and xyz via siamese_forward when available
    with torch.no_grad():
        # Prefer keys used by ReIDNet
        if all(k in inputs for k in ['sparse_1', 'sparse_2']):
            s1, s2 = inputs['sparse_1'], inputs['sparse_2']
            if torch.is_tensor(s1) and torch.is_tensor(s2):
                if hasattr(model, 'siamese_forward'):
                    xyz1, xyz2, h1, h2 = model.siamese_forward(s1.unsqueeze(0), s2.unsqueeze(0))
                    # shapes: [1,N,3], [1,N,3], [1,C,N] or [1,N,C]
                    # Normalize to [N,C]
                    def _nc(t: torch.Tensor) -> torch.Tensor:
                        if t.ndim == 3 and t.shape[0] == 1:
                            # pick [C,N] or [N,C]
                            d1, d2 = t.shape[1], t.shape[2]
                            if d1 <= 16 and d2 > d1:
                                return t[0].permute(1, 0).contiguous()
                            return t[0].contiguous()
                        return t
                    h1_nc = _nc(h1)
                    xyz1_np = (xyz1[0].contiguous().cpu().numpy() if xyz1.ndim == 3 else xyz1.cpu().numpy())
                    
                    # Extract raw point cloud from sparse input
                    raw_xyz = s1.cpu().numpy()
                    return xyz1_np, h1_nc.cpu().numpy(), raw_xyz

        # Fallback: try model.backbone returning (xyz, feat)
        if 'sparse_1' in inputs:
            s = inputs['sparse_1']
            if torch.is_tensor(s):
                if hasattr(model, 'backbone'):
                    out = model.backbone(s.unsqueeze(0), getattr(model, 'numpoints', None))
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        xyz, feat = out[0], out[1]
                        if torch.is_tensor(xyz) and torch.is_tensor(feat):
                            xyz_np = (xyz[0].contiguous().cpu().numpy() if xyz.ndim == 3 else xyz.cpu().numpy())
                            # normalize feat to [N,C]
                            if feat.ndim == 3 and feat.shape[0] == 1:
                                d1, d2 = feat.shape[1], feat.shape[2]
                                if d1 <= 16 and d2 > d1:
                                    feat_nc = feat[0].permute(1, 0).contiguous()
                                else:
                                    feat_nc = feat[0].contiguous()
                            else:
                                feat_nc = feat
                            
                            # Extract raw point cloud from sparse input
                            raw_xyz = s.cpu().numpy()
                            return xyz_np, feat_nc.cpu().numpy(), raw_xyz

    raise RuntimeError('Unable to extract per-point features from the model with provided inputs')


def viz_magnitude(xyz: np.ndarray, feat_nc: np.ndarray, raw_xyz: np.ndarray, out_dir: str, tag: str, open3d: bool, point_size: float):
    # Handle overlapping points: for points that exist in both raw and feature sets,
    # prioritize the feature colors over gray colors
    
    # Extract XYZ coordinates from raw point cloud (handles 6-channel data)
    raw_xyz_xyz = _extract_xyz_from_point_cloud(raw_xyz)
    print(f"Raw point cloud shape: {raw_xyz.shape} -> XYZ shape: {raw_xyz_xyz.shape}")
    
    # Create a dictionary to track which raw points have corresponding feature points
    raw_to_feat_map = {}
    tolerance = 1e-6  # Small tolerance for coordinate matching
    
    # Find which raw points have corresponding feature points
    for i, feat_pt in enumerate(xyz):
        for j, raw_pt in enumerate(raw_xyz_xyz):
            if np.allclose(feat_pt, raw_pt, atol=tolerance):
                raw_to_feat_map[j] = i
                break
    
    # Create colors for raw points: gray for non-overlapping, feature colors for overlapping
    raw_rgb = np.full((raw_xyz.shape[0], 3), 0.5)  # Default gray color
    
    # Feature points with magnitude-based colors: Blue (low) → Green (middle) → Red (high)
    mag = np.linalg.norm(feat_nc.astype(np.float32), axis=1)
    s01 = _norm01(mag)
    
    # Blue (low) → Green (middle) → Red (high) color scheme
    # s01 = 0: Blue [0, 0, 1]
    # s01 = 0.5: Green [0, 1, 0] 
    # s01 = 1: Red [1, 0, 0]
    
    # Red channel: increases from 0 to 1
    r = s01
    
    # Green channel: peaks at middle (0.5), decreases at ends
    g = 4 * s01 * (1 - s01)  # Creates a bell curve peaking at 0.5
    
    # Blue channel: decreases from 1 to 0
    b = 1 - s01
    
    feat_rgb = np.stack([r, g, b], axis=1)
    
    # Update raw point colors where they overlap with feature points
    for raw_idx, feat_idx in raw_to_feat_map.items():
        raw_rgb[raw_idx] = feat_rgb[feat_idx]
    
    # Save the raw point cloud with updated colors (overlapping points now have feature colors)
    # Use XYZ coordinates for PLY file (3D visualization)
    combined_out_ply = osp.join(out_dir, f'combined_{tag}.ply')
    _save_ply(raw_xyz_xyz, raw_rgb, combined_out_ply)
    print(f'Saved combined: {combined_out_ply} (raw: {raw_xyz.shape[0]} points, overlapping: {len(raw_to_feat_map)} points)')
    
    if open3d:
        try:
            import open3d as o3d
            # Create point cloud with XYZ coordinates but updated colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_xyz_xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(raw_rgb.astype(np.float64))
            
            # Show combined point cloud
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f'combined_{tag}', width=1000, height=700)
            vis.add_geometry(pcd)
            ro = vis.get_render_option()
            ro.point_size = float(point_size)
            vis.run()
            vis.destroy_window()
            
        except Exception:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            
            # Plot combined point cloud
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Separate overlapping and non-overlapping points for better visualization
            overlapping_mask = np.zeros(len(raw_xyz), dtype=bool)
            for raw_idx in raw_to_feat_map.keys():
                overlapping_mask[raw_idx] = True
            
            # Plot non-overlapping raw points in gray (using XYZ coordinates)
            non_overlapping_xyz = raw_xyz_xyz[~overlapping_mask]
            if len(non_overlapping_xyz) > 0:
                ax.scatter(non_overlapping_xyz[:, 0], non_overlapping_xyz[:, 1], non_overlapping_xyz[:, 2], 
                          c='gray', s=point_size*0.5, alpha=0.6, label='Raw points (non-overlapping)')
            
            # Plot overlapping points with feature colors (using XYZ coordinates)
            overlapping_xyz = raw_xyz_xyz[overlapping_mask]
            overlapping_rgb = raw_rgb[overlapping_mask]
            if len(overlapping_xyz) > 0:
                ax.scatter(overlapping_xyz[:, 0], overlapping_xyz[:, 1], overlapping_xyz[:, 2], 
                          c=overlapping_rgb, s=point_size, label='Feature points (overlapping)')
            
            ax.set_title(f'Combined Point Cloud - {tag}\n(Gray: Raw only, Colored: Raw+Features overlap)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.show()


def viz_diff(xyz1: np.ndarray, feat1: np.ndarray, xyz2: np.ndarray, feat2: np.ndarray, raw_xyz1: np.ndarray, raw_xyz2: np.ndarray, out_dir: str, tag: str, open3d: bool, point_size: float):
    n = min(len(xyz1), len(xyz2), len(feat1), len(feat2))
    xyz = xyz1[:n]
    
    # Helper function to handle overlapping points
    def create_combined_colors(raw_xyz, feat_xyz, feat_rgb, default_color):
        raw_to_feat_map = {}
        tolerance = 1e-6
        
        # Extract XYZ coordinates from raw point cloud (handles 6-channel data)
        raw_xyz_xyz = _extract_xyz_from_point_cloud(raw_xyz)
        
        # Find which raw points have corresponding feature points
        for i, feat_pt in enumerate(feat_xyz):
            for j, raw_pt in enumerate(raw_xyz_xyz):
                if np.allclose(feat_pt, raw_pt, atol=tolerance):
                    raw_to_feat_map[j] = i
                    break
        
        # Create colors for raw points: default color for non-overlapping, feature colors for overlapping
        raw_rgb = np.full((raw_xyz.shape[0], 3), default_color)
        
        # Update raw point colors where they overlap with feature points
        for raw_idx, feat_idx in raw_to_feat_map.items():
            raw_rgb[raw_idx] = feat_rgb[feat_idx]
        
        return raw_rgb, len(raw_to_feat_map), raw_xyz_xyz
    
    # Create combined colors for sample 1: raw points (gray) + feature points (red)
    feat_rgb1 = np.full((n, 3), [1.0, 0.0, 0.0])  # Red for sample 1 features
    combined_rgb1, overlap1, raw_xyz1_xyz = create_combined_colors(raw_xyz1, xyz1[:n], feat_rgb1, 0.5)
    
    # Create combined colors for sample 2: raw points (gray) + feature points (blue)
    feat_rgb2 = np.full((n, 3), [0.0, 0.0, 1.0])  # Blue for sample 2 features
    combined_rgb2, overlap2, raw_xyz2_xyz = create_combined_colors(raw_xyz2, xyz2[:n], feat_rgb2, 0.5)
    
    # Save combined point clouds
    combined_out_ply1 = osp.join(out_dir, f'combined1_{tag}.ply')
    _save_ply(raw_xyz1, combined_rgb1, combined_out_ply1)
    print(f'Saved combined1: {combined_out_ply1} (raw: {raw_xyz1.shape[0]} points, overlapping: {overlap1} points)')
    
    combined_out_ply2 = osp.join(out_dir, f'combined2_{tag}.ply')
    _save_ply(raw_xyz2, combined_rgb2, combined_out_ply2)
    print(f'Saved combined2: {combined_out_ply2} (raw: {raw_xyz2.shape[0]} points, overlapping: {overlap2} points)')
    
    # Cosine distance per point
    f1 = feat1[:n].astype(np.float32)
    f2 = feat2[:n].astype(np.float32)
    f1n = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-6)
    f2n = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-6)
    cos = (f1n * f2n).sum(axis=1)
    dist = 1.0 - cos
    s01 = _norm01(dist)
    r = s01
    g = 1.0 - s01
    b = np.zeros_like(s01)
    diff_rgb = np.stack([r, g, b], axis=1)
    
    # Create combined colors for diff: raw points (gray) + diff points (colored)
    combined_rgb_diff, overlap_diff, raw_xyz1_xyz_diff = create_combined_colors(raw_xyz1, xyz, diff_rgb, 0.5)
    diff_out_ply = osp.join(out_dir, f'combined_diff_{tag}.ply')
    _save_ply(raw_xyz1_xyz_diff, combined_rgb_diff, diff_out_ply)
    print(f'Saved combined_diff: {diff_out_ply} (raw: {raw_xyz1.shape[0]} points, overlapping: {overlap_diff} points)')
    
    if open3d:
        try:
            import open3d as o3d
            # Show combined point clouds
            for i, (xyz_data, rgb_data, title) in enumerate([
                (raw_xyz1_xyz, combined_rgb1, f'combined1_{tag}'),
                (raw_xyz2_xyz, combined_rgb2, f'combined2_{tag}'),
                (raw_xyz1_xyz_diff, combined_rgb_diff, f'combined_diff_{tag}')
            ]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_data.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(rgb_data.astype(np.float64))
                
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=title, width=1000, height=700)
                vis.add_geometry(pcd)
                ro = vis.get_render_option()
                ro.point_size = float(point_size)
                vis.run()
                vis.destroy_window()
            
        except Exception:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            
            # Plot combined point clouds
            for i, (xyz_data, rgb_data, title) in enumerate([
                (raw_xyz1, combined_rgb1, f'Sample 1 - {tag} (Gray: Raw only, Red: Raw+Features overlap)'),
                (raw_xyz2, combined_rgb2, f'Sample 2 - {tag} (Gray: Raw only, Blue: Raw+Features overlap)'),
                (raw_xyz1, combined_rgb_diff, f'Difference - {tag} (Gray: Raw only, Colored: Raw+Diff overlap)')
            ]):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create mask for overlapping points
                overlapping_mask = np.zeros(len(xyz_data), dtype=bool)
                tolerance = 1e-6
                
                # For sample 1 and 2, check overlap with their respective feature points
                if i < 2:  # Sample 1 or 2
                    feat_xyz = xyz1[:n] if i == 0 else xyz2[:n]
                    for feat_pt in feat_xyz:
                        for j, raw_pt in enumerate(xyz_data):
                            if np.allclose(feat_pt, raw_pt, atol=tolerance):
                                overlapping_mask[j] = True
                                break
                else:  # Diff
                    for feat_pt in xyz:
                        for j, raw_pt in enumerate(xyz_data):
                            if np.allclose(feat_pt, raw_pt, atol=tolerance):
                                overlapping_mask[j] = True
                                break
                
                # Plot non-overlapping raw points in gray
                non_overlapping_xyz = xyz_data[~overlapping_mask]
                if len(non_overlapping_xyz) > 0:
                    ax.scatter(non_overlapping_xyz[:, 0], non_overlapping_xyz[:, 1], non_overlapping_xyz[:, 2], 
                              c='gray', s=point_size*0.5, alpha=0.6, label='Raw points (non-overlapping)')
                
                # Plot overlapping points with feature/diff colors
                overlapping_xyz = xyz_data[overlapping_mask]
                overlapping_rgb = rgb_data[overlapping_mask]
                if len(overlapping_xyz) > 0:
                    ax.scatter(overlapping_xyz[:, 0], overlapping_xyz[:, 1], overlapping_xyz[:, 2], 
                              c=overlapping_rgb, s=point_size, label='Feature/Diff points (overlapping)')
                
                ax.set_title(title)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
                plt.show()


def main():
    parser = argparse.ArgumentParser(description='Point cloud feature visualization toolkit')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--out-dir', default='runs/temp')
    parser.add_argument('--mode', default='magnitude', choices=['magnitude', 'diff'])
    parser.add_argument('--open3d', action='store_true')
    parser.add_argument('--point-size', type=float, default=13.0)
    parser.add_argument('--start', type=int, default=650)
    parser.add_argument('--end', type=int, default=750)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    model = build_model(cfg.model)
    if hasattr(model, 'init_weights'):
        try:
            model.init_weights()
        except Exception:
            pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if args.checkpoint and osp.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state, strict=False)

    start = max(0, args.start)
    end = min(len(dataset) - 1, args.end if args.end > 0 else start)

    for idx in range(start, end + 1):
        sample = dataset[idx]
        inputs = _build_inputs(sample if isinstance(sample, dict) else {}, device)
        tag = f'val_{idx}'

        if args.mode == 'magnitude':
            try:
                # print(model)
                # print(inputs)
                xyz, feat, raw_xyz = _features_from_model(model, inputs)
                feat = feat.T
                print(f"Raw point cloud shape: {raw_xyz.shape} (channels: {raw_xyz.shape[1]})")
                print(f"Feature point cloud shape: {xyz.shape}")
                print(f"Feature shape: {feat.shape}")
                print(f"Color scheme: Blue (low magnitude) → Green (middle) → Red (high magnitude)")
                viz_magnitude(xyz, feat, raw_xyz, args.out_dir, tag, args.open3d, args.point_size)
            except Exception as e:
                print(f'[warn] magnitude idx={idx} failed: {e}')
        elif args.mode == 'diff':
            # Use both sparse_1 and sparse_2 paths where available
            try:
                if all(k in inputs for k in ['sparse_1', 'sparse_2']):
                    s1, s2 = inputs['sparse_1'], inputs['sparse_2']
                    if torch.is_tensor(s1) and torch.is_tensor(s2) and hasattr(model, 'siamese_forward'):
                        with torch.no_grad():
                            xyz1, xyz2, h1, h2 = model.siamese_forward(s1.unsqueeze(0), s2.unsqueeze(0))
                        # Normalize features to [N,C]
                        def _nc(t: torch.Tensor) -> torch.Tensor:
                            if t.ndim == 3 and t.shape[0] == 1:
                                d1, d2 = t.shape[1], t.shape[2]
                                if d1 <= 16 and d2 > d1:
                                    return t[0].permute(1, 0).contiguous()
                                return t[0].contiguous()
                            return t
                        xyz1_np = (xyz1[0].contiguous().cpu().numpy() if xyz1.ndim == 3 else xyz1.cpu().numpy())
                        xyz2_np = (xyz2[0].contiguous().cpu().numpy() if xyz2.ndim == 3 else xyz2.cpu().numpy())
                        h1_np = _nc(h1).cpu().numpy()
                        h2_np = _nc(h2).cpu().numpy()
                        
                        # Extract raw point clouds from sparse inputs
                        raw_xyz1 = s1.cpu().numpy()
                        raw_xyz2 = s2.cpu().numpy()
                        
                        viz_diff(xyz1_np, h1_np, xyz2_np, h2_np, raw_xyz1, raw_xyz2, args.out_dir, tag, args.open3d, args.point_size)
                else:
                    print(f'[warn] diff mode requires sparse_1 and sparse_2 in sample idx={idx}')
            except Exception as e:
                print(f'[warn] diff idx={idx} failed: {e}')


if __name__ == '__main__':
    main()


