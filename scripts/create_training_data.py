"""
Generate training dataset for neural initialization.

Creates pairs of (degraded raster, oracle SVG control points) for training.
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

from config import RASTER_DEGRADED, TRAIN_IDS


def parse_svg_to_control_points(svg_path: Path, max_paths: int = 10, max_points_per_path: int = 50):
    """
    Extract control points from SVG as tensors.
    
    Args:
        svg_path: Path to SVG file
        max_paths: Maximum number of paths
        max_points_per_path: Maximum points per path
        
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
        
        # Extract coordinates
        coords = []
        tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
        
        for token in tokens:
            match = re.search(r'([\d.]+),([\d.]+)', token)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                coords.append([x, y])
        
        if not coords:
            continue
        
        # Fill in points and masks
        num_points = min(len(coords), max_points_per_path)
        for i in range(num_points):
            points[path_idx, i] = torch.tensor(coords[i])
            masks[path_idx, i] = 1.0
        
        path_idx += 1
    
    return points, masks


def create_training_dataset(
    output_path: Path = Path('data/training_dataset.pkl'),
    num_samples: int = None,
    verbose: bool = True
):
    """
    Create training dataset from oracle outputs.
    
    Args:
        output_path: Where to save dataset
        num_samples: Number of samples to process (None = all)
        verbose: Print progress
    """
    # Load training IDs
    train_ids = TRAIN_IDS.read_text().strip().split('\n')
    if num_samples:
        train_ids = train_ids[:num_samples]
    
    dataset = []
    
    print(f"Creating training dataset from {len(train_ids)} samples...")
    print(f"Looking for oracle outputs in: baselines/advanced_full/")
    
    for train_id in tqdm(train_ids, desc="Processing", disable=not verbose):
        # For each training ID, we have 10 augmented variants
        for aug_idx in range(1, 11):
            try:
                # Load degraded raster
                raster_path = RASTER_DEGRADED / f"{train_id}_{aug_idx:02d}.png"
                if not raster_path.exists():
                    continue
                
                raster = np.array(Image.open(raster_path).convert('L')) / 255.0
                raster_tensor = torch.from_numpy(raster).float().unsqueeze(0)  # (1, 256, 256)
                
                # Load oracle SVG (use full benchmark results - they are high quality!)
                oracle_svg_path = Path('baselines/advanced_full') / f"{train_id}.svg"
                if not oracle_svg_path.exists():
                    continue
                
                # Parse control points
                points, masks = parse_svg_to_control_points(oracle_svg_path)
                
                # Add to dataset
                dataset.append({
                    'id': f"{train_id}_{aug_idx:02d}",
                    'raster': raster_tensor,
                    'points': points,
                    'masks': masks
                })
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {train_id}_{aug_idx:02d}: {str(e)}")
                continue
    
    print(f"\nDataset created: {len(dataset)} samples")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Saved to: {output_path}")
    
    # Statistics
    if dataset:
        avg_paths = sum((d['masks'].sum(dim=1) > 0).sum().item() for d in dataset) / len(dataset)
        avg_points = sum(d['masks'].sum().item() for d in dataset) / len(dataset)
        print(f"\nDataset statistics:")
        print(f"  Average paths per sample: {avg_paths:.1f}")
        print(f"  Average points per sample: {avg_points:.1f}")
    
    return dataset


def test_dataset_loading():
    """Test loading and inspecting the dataset."""
    dataset_path = Path('data/training_dataset.pkl')
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    print("Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Inspect first sample
    if dataset:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  ID: {sample['id']}")
        print(f"  Raster shape: {sample['raster'].shape}")
        print(f"  Points shape: {sample['points'].shape}")
        print(f"  Masks shape: {sample['masks'].shape}")
        print(f"  Valid paths: {(sample['masks'].sum(dim=1) > 0).sum().item()}")
        print(f"  Total valid points: {sample['masks'].sum().item()}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None, help='Number of samples (None = all)')
    parser.add_argument('--test', action='store_true', help='Test loading dataset')
    args = parser.parse_args()
    
    if args.test:
        test_dataset_loading()
    else:
        create_training_dataset(num_samples=args.samples)
