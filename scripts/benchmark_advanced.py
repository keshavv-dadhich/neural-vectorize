"""
Pilot benchmark with advanced loss functions.

This tests the complete pipeline:
1. Edge initialization
2. Optimization with multi-term loss
3. Segment reduction (aggressive merge)

Tests on 5 samples to validate improvements before full-scale run.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from typing import List, Dict
import torch
import torch.nn.functional as F

from config import RASTER_PERFECT, RASTER_DEGRADED, TEST_IDS, BASELINES
from vectorizers.initialize import EdgeInitializer
from vectorizers.optimize_v3 import AdvancedVectorizer
from vectorizers.merge import aggressive_simplify_svg
from utils.metrics import compute_l2, compute_ssim


def run_pilot_benchmark(
    num_samples: int = 5,
    output_dir: Path = Path('baselines/advanced_pilot'),
    verbose: bool = True
) -> Dict:
    """
    Run pilot benchmark with advanced loss.
    
    Args:
        num_samples: Number of test samples
        output_dir: Where to save results
        verbose: Print progress
        
    Returns:
        Results dictionary
    """
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test samples
    test_ids = TEST_IDS.read_text().strip().split('\n')[:num_samples]
    
    # Initialize components
    initializer = EdgeInitializer(
        canny_low=50,
        canny_high=150,
        epsilon_factor=0.002,
        min_contour_length=20
    )
    
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,        # CRITICAL for visual coherence
        lambda_curvature=0.1,   # Smooth strokes
        lambda_intersection=0.3,# Vector sanity
        lambda_complexity=0.005 # Segment count
    )
    
    results = []
    
    print(f"{'='*60}")
    print(f"PILOT BENCHMARK: Advanced Loss Functions")
    print(f"Testing {num_samples} samples")
    print(f"{'='*60}\n")
    
    for idx, test_id in enumerate(tqdm(test_ids, desc="Processing samples")):
        result = {'id': test_id}
        
        try:
            # Paths
            raster_perfect_path = RASTER_PERFECT / f"{test_id}_base.png"
            raster_degraded_path = RASTER_DEGRADED / f"{test_id}_base.png"
            init_svg_path = output_dir / f"{test_id}_init.svg"
            opt_svg_path = output_dir / f"{test_id}_opt.svg"
            final_svg_path = output_dir / f"{test_id}.svg"
            
            # Load images
            target_perfect = np.array(Image.open(raster_perfect_path).convert('L')) / 255.0
            input_degraded = np.array(Image.open(raster_degraded_path).convert('L')) / 255.0
            
            # Step 1: Edge initialization
            if verbose and idx == 0:
                print(f"\nSample {idx+1}/{num_samples}: {test_id}")
                print(f"  [1/3] Edge initialization...")
            
            initializer.initialize(
                image_path=raster_degraded_path,
                output_svg_path=init_svg_path
            )
            
            # Step 2: Optimization with advanced loss
            if verbose and idx == 0:
                print(f"  [2/3] Optimization with multi-term loss...")
            
            vectorizer.optimize(
                svg_path=init_svg_path,
                target_image=input_degraded,
                output_path=opt_svg_path,
                num_steps=150,
                learning_rate=0.01,
                verbose=False
            )
            
            # Step 3: Aggressive merge
            if verbose and idx == 0:
                print(f"  [3/3] Segment reduction...")
            
            aggressive_simplify_svg(
                input_svg=opt_svg_path,
                output_svg=final_svg_path,
                angle_threshold=10.0,
                epsilon=0.005,
                verbose=False
            )
            
            # Compute metrics
            # Rasterize final SVG
            from utils.svg_utils import svg_to_raster
            predicted_raster = svg_to_raster(final_svg_path, size=256)
            
            # Metrics
            l2_loss = compute_l2(predicted_raster, target_perfect)
            ssim_score = compute_ssim(predicted_raster, target_perfect)
            
            # Count segments
            svg_text = final_svg_path.read_text()
            num_paths = svg_text.count('<path')
            num_segments = svg_text.count(' L') + svg_text.count(' C')
            
            # File size
            file_size_kb = final_svg_path.stat().st_size / 1024
            
            result.update({
                'l2': float(l2_loss),
                'ssim': float(ssim_score),
                'num_paths': num_paths,
                'num_segments': num_segments,
                'file_size_kb': float(file_size_kb),
                'status': 'success'
            })
            
            if verbose and idx == 0:
                print(f"  ✅ L2={l2_loss:.4f}, Segments={num_segments}, Size={file_size_kb:.2f}KB")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            if verbose:
                print(f"  ❌ Error: {str(e)}")
        
        results.append(result)
    
    # Aggregate statistics
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        stats = {
            'num_samples': len(test_ids),
            'num_successful': len(successful),
            'avg_l2': float(np.mean([r['l2'] for r in successful])),
            'avg_ssim': float(np.mean([r['ssim'] for r in successful])),
            'avg_segments': float(np.mean([r['num_segments'] for r in successful])),
            'avg_file_size_kb': float(np.mean([r['file_size_kb'] for r in successful])),
            'results': results
        }
    else:
        stats = {
            'num_samples': len(test_ids),
            'num_successful': 0,
            'results': results
        }
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"PILOT BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(test_ids)}")
    if successful:
        print(f"Average L2: {stats['avg_l2']:.4f}")
        print(f"Average SSIM: {stats['avg_ssim']:.4f}")
        print(f"Average Segments: {stats['avg_segments']:.1f}")
        print(f"Average File Size: {stats['avg_file_size_kb']:.2f} KB")
    print(f"\nResults saved to: {results_path}")
    
    return stats


if __name__ == '__main__':
    run_pilot_benchmark(
        num_samples=5,
        verbose=True
    )
