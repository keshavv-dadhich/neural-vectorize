"""
Quick pilot test: Run advanced optimizer on 5 samples.

Simplified version without external dependencies.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F

from config import RASTER_DEGRADED, TEST_IDS
from vectorizers.initialize import EdgeInitializer  
from vectorizers.optimize_v3 import AdvancedVectorizer


def main():
    """Run quick pilot test."""
    output_dir = Path('baselines/advanced_pilot')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get 5 test samples
    test_ids = TEST_IDS.read_text().strip().split('\n')[:5]
    
    print(f"{'='*60}")
    print(f"PILOT TEST: Advanced Loss Functions")
    print(f"Testing {len(test_ids)} samples")
    print(f"{'='*60}\n")
    
    # Initialize components
    initializer = EdgeInitializer()
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,
        lambda_curvature=0.1,
        lambda_intersection=0.3,
        lambda_complexity=0.005
    )
    
    results = []
    
    for idx, test_id in enumerate(test_ids):
        print(f"\n[{idx+1}/{len(test_ids)}] Processing: {test_id}")
        
        try:
            # Paths
            raster_path = RASTER_DEGRADED / f"{test_id}_01.png"  # Use first augmented variant
            init_svg = output_dir / f"{test_id}_init.svg"
            final_svg = output_dir / f"{test_id}.svg"
            
            # Load image
            target = np.array(Image.open(raster_path).convert('L')) / 255.0
            
            # Step 1: Initialize
            print(f"  [1/2] Edge initialization...")
            initializer.initialize(raster_path, init_svg)
            
            # Step 2: Optimize
            print(f"  [2/2] Optimization with advanced loss...")
            vectorizer.optimize(
                svg_path=init_svg,
                target_image=target,
                output_path=final_svg,
                num_steps=150,
                learning_rate=0.01,
                verbose=False
            )
            
            # Count segments
            svg_text = final_svg.read_text()
            num_segments = svg_text.count(' L') + svg_text.count(' C')
            
            print(f"  ✅ Complete: {num_segments} segments")
            
            results.append({
                'id': test_id,
                'segments': num_segments,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'id': test_id,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results
    results_file = output_dir / 'quick_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    successful = [r for r in results if r['status'] == 'success']
    print(f"\n{'='*60}")
    print(f"PILOT TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(test_ids)}")
    if successful:
        avg_segs = np.mean([r['segments'] for r in successful])
        print(f"Average segments: {avg_segs:.1f}")
    print(f"\nResults: {results_file}")


if __name__ == '__main__':
    main()
