#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from typing import Callable, Dict, Tuple, Any, List

import numpy as np
import torch

from mmengine.config import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def _register_feature_hooks(model: torch.nn.Module, pick_fn: Callable[[str, torch.nn.Module], bool]):
    buffers: Dict[str, torch.Tensor] = {}

    def _hook(name, m, inp, out):
        if isinstance(out, (list, tuple)):
            out = out[0]
        if torch.is_tensor(out):
            try:
                # Keep only reasonably shaped feature maps (avoid scalars)
                if out.ndim >= 2 and out.numel() > 0:
                    buffers[name] = out.detach().cpu()
            except Exception:
                pass

    handles = []
    for name, m in model.named_modules():
        try:
            if pick_fn(name, m):
                handles.append(m.register_forward_hook(lambda m, i, o, n=name: _hook(n, m, i, o)))
        except Exception:
            continue
    return buffers, handles


def _normalize_to_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x - x.min()
    den = (x.max() + eps)
    if den == 0:
        return np.zeros_like(x)
    return x / den


def _features_to_rgb(features_nc: np.ndarray, method: str = "pca", channel: int = None) -> np.ndarray:
    # features_nc: [N, C]
    if features_nc.ndim != 2:
        raise ValueError(f"Expected 2D features [N, C], got {features_nc.shape}")
    if method == "channel":
        if channel is None:
            channel = 0
        channel = int(np.clip(channel, 0, features_nc.shape[1]-1))
        v = _normalize_to_01(features_nc[:, channel])
        return np.stack([v, v, v], axis=1)
    # PCA via SVD to 3D
    f = features_nc - features_nc.mean(axis=0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(f, full_matrices=False)
        proj = f @ vt[:3].T
    except np.linalg.LinAlgError:
        # fallback to first 3 channels
        c = min(3, f.shape[1])
        pad = 3 - c
        proj = np.concatenate([f[:, :c], np.zeros((f.shape[0], pad))], axis=1)
    proj = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-6)
    return np.clip(proj, 0.0, 1.0)


def _features_to_scalar01(features_nc: np.ndarray, method: str = "pca", channel: int = None) -> np.ndarray:
    # Produce a single importance scalar per point in [0,1]
    if features_nc.ndim != 2:
        raise ValueError(f"Expected 2D features [N, C], got {features_nc.shape}")
    if method == "channel":
        if channel is None:
            channel = 0
        channel = int(np.clip(channel, 0, features_nc.shape[1]-1))
        v = features_nc[:, channel]
        return _normalize_to_01(v.astype(np.float32))
    # For PCA mode: use L2 norm as a generic saliency proxy
    v = np.linalg.norm(features_nc.astype(np.float32), axis=1)
    return _normalize_to_01(v)


def _colormap(color01: np.ndarray, cmap: str = "red") -> np.ndarray:
    # Map scalar [N] to RGB [N,3] using simple built-in maps (no extra deps)
    color01 = np.clip(color01.reshape(-1, 1), 0.0, 1.0)
    if cmap == "red":
        # black -> red
        rgb = np.concatenate([color01, np.zeros_like(color01), np.zeros_like(color01)], axis=1)
    elif cmap == "jet":
        # simple 4-color jet-like
        c = color01
        r = np.clip(1.5*c - 0.5, 0, 1)
        g = np.clip(1.5 - 1.5*np.abs(c - 0.5), 0, 1)
        b = np.clip(1.5*(1 - c) - 0.5, 0, 1)
        rgb = np.concatenate([r, g, b], axis=1)
    elif cmap == "blue":
        rgb = np.concatenate([np.zeros_like(color01), np.zeros_like(color01), color01], axis=1)
    else:
        # grayscale
        rgb = np.concatenate([color01, color01, color01], axis=1)
    return np.clip(rgb, 0.0, 1.0)


def _save_ply(points_xyz: np.ndarray, rgb01: np.ndarray, out_path: str):
    n = points_xyz.shape[0]
    rgb255 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points_xyz[i]
            r, g, b = rgb255[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def _select_buffer(buffers: Dict[str, torch.Tensor], prefer_keys: List[str]) -> Tuple[str, torch.Tensor]:
    if not buffers:
        raise RuntimeError("No feature buffers captured. Try different --match or layer names.")
    # Prefer user-provided order and prefer 3D or 2D tensors
    def _best_tensor(items):
        # Rank: 3D first, then 2D, then others
        ranked = sorted(items, key=lambda kv: (0 if kv[1].ndim == 3 else (1 if kv[1].ndim == 2 else 2)))
        return ranked[0]

    for key in prefer_keys:
        matches = [(name, t) for name, t in buffers.items() if key and key in name]
        if matches:
            name, tensor = _best_tensor(matches)
            return name, tensor

    # No key match; pick overall best
    name, tensor = _best_tensor(list(buffers.items()))
    return name, tensor


def _infer_feat_nc(t: torch.Tensor) -> torch.Tensor:
    # Expect [B, C, N] or [B, N, C] with B==1 → return [N, C]
    if t.ndim == 2:
        # Assume [N, C], add batch
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Captured feature must be 3D, got {t.shape}")
    if t.shape[0] != 1:
        # take the first in batch
        t = t[:1]
    b, d1, d2 = t.shape
    # Heuristic: treat the larger dim as N when ambiguous
    if d1 <= 16 and d2 > d1:
        # [B, C, N]
        nc = t[0].permute(1, 0).contiguous()  # [N, C]
    elif d2 <= 16 and d1 > d2:
        # [B, N, C]
        nc = t[0].contiguous()  # [N, C]
    else:
        # choose interpretation with bigger N
        if d2 >= d1:
            nc = t[0].contiguous()
        else:
            nc = t[0].permute(1, 0).contiguous()
    return nc


def visualize_from_config(cfg_path: str,
                          checkpoint: str,
                          sample_index: int,
                          match_layers: List[str],
                          method: str,
                          channel: int,
                          out_dir: str,
                          open3d_view: bool,
                          overlay: bool,
                          alpha: float,
                          cmap: str):
    cfg = Config.fromfile(cfg_path)

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    sample_index = int(np.clip(sample_index, 0, len(dataset) - 1))

    data = dataset[sample_index]

    # Some datasets return lists; ensure tensors are on device later
    model = build_model(cfg.model)
    if hasattr(model, 'init_weights'):
        try:
            model.init_weights()
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if checkpoint and osp.isfile(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            print(f"[warn] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
        if len(missing) > 0:
            print(f"[warn] Missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")

    # Prepare input dict to match model signature (similar to training/testing code)
    inputs: Dict[str, Any] = {}
    for k, v in data.items() if isinstance(data, dict) else []:
        print(f"Processing key {k}: type={type(v)}, value={v}")
        
        # Handle DataContainer objects first
        if hasattr(v, 'data'):
            print(f"  Extracting data from DataContainer for {k}")
            v = v.data
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'data'):
            print(f"  Extracting data from list of DataContainers for {k}")
            v = [item.data if hasattr(item, 'data') else item for item in v]
        
        if torch.is_tensor(v):
            print(f"  {k} is tensor: {v.shape}")
            inputs[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
            print(f"  {k} is list of tensors: {[t.shape for t in v]}")
            inputs[k] = [t.to(device) for t in v]
        elif isinstance(v, list) and len(v) > 0:
            # Handle mixed lists (some tensors, some not)
            print(f"  {k} is mixed list: {[type(item) for item in v]}")
            inputs[k] = [item.to(device) if torch.is_tensor(item) else item for item in v]
        else:
            print(f"  {k} is other type: {type(v)}")
            inputs[k] = v

    # Fallback: some datasets may store points under keys like 'sparse_1'
    # We will try to infer points for visualization from common keys
    # print(inputs.keys())

    points_xyz = None
    for key in [
        'sparse_1', 'sparse_2'
    ]:
        if key in inputs:
            val = inputs[key]

            if isinstance(val, list):
                val = val[0]
            
            # Handle DataContainer objects
            if hasattr(val, 'data'):
                val = val.data
                print("Extracted data from DataContainer")
            
            if torch.is_tensor(val):
                print("val is tensor")
                arr = val.detach().cpu().float()

                print(arr.shape)

                if arr.ndim == 2 and arr.shape[1] >= 3:
                    points_xyz = arr[:, :3].numpy()
                    break

    if points_xyz is None:
        raise RuntimeError("Could not infer point coordinates from the sample. Please adapt the key search.")

    # Register hooks
    def pick_fn(name: str, module: torch.nn.Module) -> bool:
        return any((m in name) for m in match_layers) if match_layers else False

    buffers, handles = _register_feature_hooks(model, pick_fn)

    # Debug: print what we're about to pass to the model
    print("\nInputs being passed to model:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: tensor {v.shape}")
        elif isinstance(v, list):
            print(f"  {k}: list of {len(v)} items, types: {[type(item) for item in v]}")
        else:
            print(f"  {k}: {type(v)}")
    
    with torch.no_grad():
        _ = model(return_loss=False, **inputs)

    # Fallback: if nothing captured, try capturing from all modules
    if len(buffers) == 0:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        def pick_any(name: str, module: torch.nn.Module) -> bool:
            return True
        buffers, handles = _register_feature_hooks(model, pick_any)
        with torch.no_grad():
            _ = model(return_loss=False, **inputs)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    name, feat = _select_buffer(buffers, match_layers)
    feat_nc = _infer_feat_nc(feat)
    feat_nc_np = feat_nc.detach().cpu().numpy()

    if feat_nc_np.shape[0] != points_xyz.shape[0]:
        # Try to align by truncation/padding if minor mismatch
        n = min(feat_nc_np.shape[0], points_xyz.shape[0])
        feat_nc_np = feat_nc_np[:n]
        points_xyz = points_xyz[:n]

    if overlay:
        # Flat gray base + heat overlay
        gray = np.full((points_xyz.shape[0], 3), 0.7, dtype=np.float32)
        s01 = _features_to_scalar01(feat_nc_np, method=method, channel=channel)
        heat = _colormap(s01, cmap=cmap).astype(np.float32)
        a = float(np.clip(alpha, 0.0, 1.0))
        # Blend proportionally to saliency
        rgb = (1.0 - a*s01.reshape(-1,1))*gray + (a*s01.reshape(-1,1))*heat
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = _features_to_rgb(feat_nc_np, method=method, channel=channel)
    os.makedirs(out_dir, exist_ok=True)
    safe_name = name.replace('/', '_').replace('.', '_')
    out_ply = osp.join(out_dir, f"heatmap_val_{sample_index}_{safe_name}.ply")
    _save_ply(points_xyz, rgb, out_ply)
    print(f"Saved: {out_ply}")

    if open3d_view:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            print(f"[warn] Open3D view failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize per-point features as a heatmap (colored PLY)")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', default='', help='Optional checkpoint path')
    parser.add_argument('--match', nargs='*', default=['backbone', 'ED_DualReID', 'transformer', 'gcn', 'edge'], help='Layer name substrings to hook (ordered preference)')
    parser.add_argument('--method', default='pca', choices=['pca', 'channel'], help='Color mapping method')
    parser.add_argument('--channel', type=int, default=0, help='Channel index when using method=channel')
    parser.add_argument('--out-dir', default='runs/feature_heatmaps', help='Output directory')
    parser.add_argument('--open3d', action='store_true', help='Open interactive Open3D viewer')
    parser.add_argument('--overlay', action='store_true', help='Blend heat over flat gray base color')
    parser.add_argument('--alpha', type=float, default=0.8, help='Overlay intensity scale in [0,1]')
    parser.add_argument('--cmap', type=str, default='jet', choices=['red','jet','blue','gray'], help='Overlay colormap')

    args = parser.parse_args()

    # Determine dataset length to iterate 0..1000 safely
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    num_samples = len(dataset)
    end_index = min(1000, num_samples - 1)

    for idx in range(0, end_index + 1):
        try:
            visualize_from_config(
                cfg_path=args.config,
                checkpoint=args.checkpoint,
                sample_index=idx,
                match_layers=args.match,
                method=args.method,
                channel=args.channel,
                out_dir=args.out_dir,
                open3d_view=args.open3d,
                overlay=args.overlay,
                alpha=args.alpha,
                cmap=args.cmap,
            )
        except Exception as e:
            print(f"[warn] Skipping index {idx} due to error: {e}")


if __name__ == '__main__':
    main()


