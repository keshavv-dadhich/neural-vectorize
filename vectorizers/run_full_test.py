"""
Run optimization vectorizer on ALL 77 test samples efficiently.

This is Priority 2: Scale from 5 pilot samples to full test set.

Optimizations for speed:
1. Fewer optimization steps (100 instead of 200)
2. Parallel processing where possible
3. Checkpoint/resume capability
4. Progress tracking
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from vectorizer import OptimizationVectorizer
from config import *
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


def run_full_test_set(
    num_samples: int = 77,  # All test samples
    num_steps: int = 100,   # Reduced from 200 for speed
    resume: bool = True,    # Skip already processed
    output_dir: Path = None
):
    """
    Run optimization vectorizer on all test samples.
    
    Args:
        num_samples: Number of test samples to process (max 77)
        num_steps: Optimization steps per sample
        resume: Skip already-processed samples
        output_dir: Where to save results
    """
    if output_dir is None:
        output_dir = BASELINES / "optimization_full"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test IDs
    test_ids = TEST_IDS.read_text().strip().split('\n')[:num_samples]
    
    print(f"{'='*70}")
    print(f"RUNNING OPTIMIZATION VECTORIZER ON {len(test_ids)} TEST SAMPLES")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Optimization steps: {num_steps}")
    print(f"Resume mode: {resume}")
    print(f"{'='*70}\n")
    
    # Initialize vectorizer (shared across samples)
    vectorizer = OptimizationVectorizer(
        canny_low=50,
        canny_high=150,
        epsilon_factor=0.002,
        learning_rate=0.01,
        num_steps=num_steps  # Faster
    )
    
    # Track progress
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_ids),
        'num_steps': num_steps,
        'samples': {}
    }
    
    # Process each sample
    skipped = 0
    processed = 0
    failed = 0
    
    pbar = tqdm(test_ids, desc="Vectorizing")
    for test_id in pbar:
        output_svg = output_dir / f"{test_id}.svg"
        
        # Skip if already exists (resume mode)
        if resume and output_svg.exists():
            skipped += 1
            pbar.set_postfix({'processed': processed, 'skipped': skipped, 'failed': failed})
            continue
        
        try:
            # Input raster
            raster_path = RASTER_PERFECT / f"{test_id}_base.png"
            
            if not raster_path.exists():
                failed += 1
                results['samples'][test_id] = {'status': 'missing_input'}
                continue
            
            # Vectorize
            svg_string = vectorizer.vectorize(
                raster_path=raster_path,
                output_path=output_svg,
                verbose=False  # Quiet mode for batch processing
            )
            
            processed += 1
            results['samples'][test_id] = {
                'status': 'success',
                'svg_path': str(output_svg.relative_to(BASELINES))
            }
            
        except Exception as e:
            failed += 1
            results['samples'][test_id] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"\n❌ Failed on {test_id}: {e}")
        
        pbar.set_postfix({'processed': processed, 'skipped': skipped, 'failed': failed})
    
    # Save results manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Processed: {processed}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total: {len(test_ids)}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"{'='*70}\n")
    
    return results


def estimate_time(num_samples: int = 77, seconds_per_sample: float = 110):
    """Estimate total processing time."""
    total_seconds = num_samples * seconds_per_sample
    hours = total_seconds / 3600
    
    print(f"Time estimate for {num_samples} samples:")
    print(f"  @ {seconds_per_sample}s per sample")
    print(f"  = {total_seconds:.0f} seconds")
    print(f"  = {total_seconds/60:.1f} minutes")
    print(f"  = {hours:.2f} hours")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimization vectorizer on full test set')
    parser.add_argument('--num-samples', type=int, default=77, help='Number of samples to process')
    parser.add_argument('--num-steps', type=int, default=100, help='Optimization steps per sample')
    parser.add_argument('--no-resume', action='store_true', help='Re-process existing samples')
    parser.add_argument('--estimate', action='store_true', help='Just show time estimate')
    
    args = parser.parse_args()
    
    if args.estimate:
        estimate_time(args.num_samples)
    else:
        results = run_full_test_set(
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            resume=not args.no_resume
        )
        
        print("✅ Run complete. Use scripts/benchmark.py to evaluate results.")
