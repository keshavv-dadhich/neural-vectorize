"""
Create training dataset from benchmark SVGs.

Uses the 77 high-quality SVGs from advanced_full benchmark.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re

from config import RASTER_DEGRADED


def parse_svg_to_control_points(svg_path: Path, max_paths: int = 10, max_points_per_path: int = 50):
    """
    Extract control points from SVG as tensors.
    
    Returns:
        points: (max_paths, max_points_per_path, 2) tensor
        masks: (max_paths, max_points_per_path) tensor (1 = valid, 0 = padding)
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Initialize tensors with zeros
    points = torch.zeros(max_paths, max_points_per_path, 2)
    masks = torch.zeros(max_paths, max_points_per_path)
    
    path_idx = 0
    for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
        if path_idx >= max_paths:
            break
            
        d = path_elem.get('d', '')
        
        # Extract coordinates - handle M and L commands
        coords = []
        parts = d.strip().split()
        i = 0
        while i < len(parts):
            if parts[i] in ['M', 'L']:
                # Next two parts should be coordinates
                if i + 1 < len(parts):
                    try:
                        # Try to parse x,y
                        coord_str = parts[i + 1]
                        if ',' in coord_str:
                            x, y = coord_str.split(',')
                        else:
                            # Space-separated
                            x = parts[i + 1]
                            y = parts[i + 2] if i + 2 < len(parts) else x
                            i += 1
                        
                        x = float(x)
                        y = float(y)
                        coords.append([x, y])
                    except:
                        pass
                i += 2
            else:
                i += 1
        
        if not coords:
            path_idx += 1
            continue
        
        # Fill in points for this path
        for point_idx, (x, y) in enumerate(coords):
            if point_idx >= max_points_per_path:
                break
            points[path_idx, point_idx, 0] = x
            points[path_idx, point_idx, 1] = y
            masks[path_idx, point_idx] = 1.0
        
        path_idx += 1
    
    return points, masks


def main():
    # Get all benchmark SVGs
    benchmark_dir = Path('baselines/advanced_full')
    svg_paths = sorted(benchmark_dir.glob('*.svg'))
    
    print(f"Creating training dataset from {len(svg_paths)} benchmark SVGs...")
    
    dataset = []
    failed = []
    
    for svg_path in tqdm(svg_paths, desc="Processing"):
        sample_id = svg_path.stem
        
        # Use all 10 augmented rasters per SVG
        for aug_idx in range(1, 11):
            try:
                # Load degraded raster
                raster_path = RASTER_DEGRADED / f"{sample_id}_{aug_idx:02d}.png"
                if not raster_path.exists():
                    continue
                
                raster = np.array(Image.open(raster_path).convert('L')) / 255.0
                raster_tensor = torch.from_numpy(raster).float().unsqueeze(0)  # (1, 256, 256)
                
                # Parse control points from SVG
                points, masks = parse_svg_to_control_points(svg_path)
                
                # Validate
                if masks.sum() < 2:
                    failed.append(f"{sample_id}_{aug_idx:02d}: too few valid points")
                    continue
                
                # Check for degenerate paths (all same point)
                valid_points = points[masks > 0.5]
                if len(valid_points) > 0:
                    point_std = valid_points.std().item()
                    if point_std < 0.01:
                        failed.append(f"{sample_id}_{aug_idx:02d}: degenerate (std={point_std:.4f})")
                        continue
                
                # Add to dataset
                dataset.append({
                    'raster': raster_tensor,
                    'points': points,
                    'masks': masks,
                    'id': f"{sample_id}_{aug_idx:02d}"
                })
            
            except Exception as e:
                failed.append(f"{sample_id}_{aug_idx:02d}: {str(e)}")
                continue
    
    print(f"\nDataset created: {len(dataset)} samples")
    if failed:
        print(f"Failed: {len(failed)} samples")
        print(f"First 5 failures: {failed[:5]}")
    
    # Save
    output_path = Path('data/training_dataset.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Saved to: {output_path}")
    
    # Statistics
    if dataset:
        avg_paths = np.mean([d['masks'].sum(dim=1).gt(0).sum().item() for d in dataset])
        avg_points = np.mean([d['masks'].sum().item() for d in dataset])
        print(f"\nDataset statistics:")
        print(f"  Average paths per sample: {avg_paths:.1f}")
        print(f"  Average points per sample: {avg_points:.1f}")
        
        # Check first sample
        sample = dataset[0]
        valid_points = sample['points'][sample['masks'] > 0.5]
        print(f"  First sample valid points: {len(valid_points)}")
        print(f"  First sample point range: [{valid_points.min():.3f}, {valid_points.max():.3f}]")


if __name__ == '__main__':
    main()
