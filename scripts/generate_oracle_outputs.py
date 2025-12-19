"""
Generate oracle outputs for all training samples.

Runs optimize_v3.py on all training samples to create oracle SVGs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import json

from config import RASTER_PERFECT, TRAIN_IDS, BASELINES
from vectorizers.optimize_v3 import AdvancedVectorizer


def generate_oracle_outputs(
    output_dir: Path = Path('baselines/oracle_training'),
    init_dir: Path = None,
    num_samples: int = None,
    num_steps: int = 150,
    verbose: bool = True
):
    """
    Generate oracle SVG outputs for training samples.
    
    Args:
        output_dir: Where to save oracle SVGs
        init_dir: Directory with initialization SVGs (defaults to Potrace)
        num_samples: Number of samples to process (None = all)
        num_steps: Optimization steps
        verbose: Print progress
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default to Potrace initialization
    if init_dir is None:
        init_dir = BASELINES / 'potrace'
    
    # Load training IDs
    train_ids = TRAIN_IDS.read_text().strip().split('\n')
    if num_samples:
        train_ids = train_ids[:num_samples]
    
    print(f"Generating oracle outputs for {len(train_ids)} samples...")
    print(f"Initialization from: {init_dir}")
    print(f"Steps per sample: {num_steps}")
    print(f"Output directory: {output_dir}")
    
    # Initialize vectorizer
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,
        lambda_curvature=0.1,
        lambda_intersection=0.3,
        lambda_complexity=0.005
    )
    
    results = []
    failed = []
    
    for train_id in tqdm(train_ids, desc="Processing", disable=not verbose):
        try:
            # Load perfect raster as target
            raster_path = RASTER_PERFECT / f"{train_id}_base.png"
            
            if not raster_path.exists():
                if verbose:
                    print(f"Warning: {raster_path} not found, skipping...")
                failed.append(train_id)
                continue
            
            # Load initialization SVG
            init_svg_path = init_dir / f"{train_id}.svg"
            if not init_svg_path.exists():
                if verbose:
                    print(f"Warning: {init_svg_path} not found, skipping...")
                failed.append(train_id)
                continue
            
            # Load target raster
            target = np.array(Image.open(raster_path).convert('L')) / 255.0
            
            # Optimize
            output_svg_path = output_dir / f"{train_id}.svg"
            vectorizer.optimize(
                init_svg_path,
                target,
                output_svg_path,
                num_steps=num_steps,
                learning_rate=0.01,
                verbose=False
            )
            
            # For tracking, we don't have direct access to the final loss
            # So just record success
            results.append({
                'id': train_id,
                'status': 'success'
            })
            
        except Exception as e:
            if verbose:
                import traceback
                print(f"Error processing {train_id}: {str(e)}")
                traceback.print_exc()
            failed.append(train_id)
            continue
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'num_samples': len(results),
            'num_failed': len(failed),
            'failed_ids': failed,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Oracle generation complete!")
    print(f"   Successful: {len(results)}/{len(train_ids)}")
    print(f"   Failed: {len(failed)}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='baselines/oracle_training')
    parser.add_argument('--init', type=str, default=None, help='Init SVG directory (default: Potrace)')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples (None = all)')
    parser.add_argument('--steps', type=int, default=150)
    args = parser.parse_args()
    
    init_dir = Path(args.init) if args.init else None
    
    generate_oracle_outputs(
        output_dir=Path(args.output),
        init_dir=init_dir,
        num_samples=args.samples,
        num_steps=args.steps
    )
