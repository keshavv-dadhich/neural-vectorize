"""
Scale optimization to all 77 test samples efficiently.

Optimizations:
1. Reduced optimization steps (150 instead of 200)
2. Aggressive simplification after optimization
3. Checkpoint/resume capability
4. Parallel-ready (can run multiple instances)
5. Progress tracking with ETA
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import time

# Local imports
from initialize import EdgeInitializer
from optimize import DifferentiableVectorizer
from merge import aggressive_simplify_svg

# Config
from config import *


def vectorize_single_sample(
    test_id: str,
    output_dir: Path,
    num_steps: int = 150,  # Reduced from 200
    verbose: bool = False
) -> dict:
    """
    Vectorize one sample with full pipeline.
    
    Returns:
        Result dict with stats
    """
    result = {'test_id': test_id, 'status': 'unknown'}
    
    try:
        start_time = time.time()
        
        # Paths
        raster_path = RASTER_PERFECT / f"{test_id}_base.png"
        init_svg = output_dir / f"{test_id}_init.svg"
        opt_svg = output_dir / f"{test_id}_opt.svg"
        final_svg = output_dir / f"{test_id}.svg"
        
        if not raster_path.exists():
            result['status'] = 'missing_input'
            return result
        
        # Phase A: Initialize from edges
        initializer = EdgeInitializer(
            canny_low=50,
            canny_high=150,
            epsilon_factor=0.002,
            min_contour_length=20
        )
        
        initializer.initialize_from_raster(
            raster_path,
            init_svg,
            stroke_width=STROKE_WIDTH
        )
        
        # Phase B: Optimize
        optimizer = DifferentiableVectorizer(image_size=256)
        optimizer.optimize(
            init_svg,
            raster_path,
            opt_svg,
            num_steps=num_steps,
            verbose=verbose
        )
        
        # Phase C: Aggressive simplification
        merge_stats = aggressive_simplify_svg(
            opt_svg,
            final_svg,
            angle_threshold=5.0,
            epsilon=0.008,
            target_reduction=0.5,
            verbose=verbose
        )
        
        # Clean up intermediates
        if init_svg.exists():
            init_svg.unlink()
        if opt_svg.exists():
            opt_svg.unlink()
        
        elapsed = time.time() - start_time
        
        result.update({
            'status': 'success',
            'elapsed_seconds': elapsed,
            'segments_before_merge': merge_stats['original_segments'],
            'segments_after_merge': merge_stats['final_segments'],
            'reduction_rate': merge_stats['reduction_rate']
        })
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        if verbose:
            import traceback
            traceback.print_exc()
    
    return result


def run_full_optimization_benchmark(
    num_samples: int = 77,
    num_steps: int = 150,
    output_dir: Path = None,
    resume: bool = True,
    verbose: bool = False
) -> dict:
    """
    Run optimization on all test samples.
    
    Args:
        num_samples: Number of test samples (max 77)
        num_steps: Optimization steps per sample
        output_dir: Output directory
        resume: Skip existing files
        verbose: Print detailed progress
        
    Returns:
        Summary statistics
    """
    if output_dir is None:
        output_dir = BASELINES / "optimization_full"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test IDs
    test_ids = TEST_IDS.read_text().strip().split('\n')[:num_samples]
    
    print(f"{'='*70}")
    print(f"SCALING OPTIMIZATION TO {len(test_ids)} TEST SAMPLES")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Steps: {num_steps} (reduced for speed)")
    print(f"Resume: {resume}")
    print(f"{'='*70}\n")
    
    # Track results
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_ids),
        'num_steps': num_steps,
        'samples': {}
    }
    
    # Process samples
    processed = 0
    skipped = 0
    failed = 0
    total_time = 0
    
    pbar = tqdm(test_ids, desc="Vectorizing", unit="sample")
    for test_id in pbar:
        final_svg = output_dir / f"{test_id}.svg"
        
        # Resume: skip existing
        if resume and final_svg.exists():
            skipped += 1
            manifest['samples'][test_id] = {'status': 'skipped'}
            pbar.set_postfix({
                'done': processed,
                'skip': skipped,
                'fail': failed,
                'avg': f"{total_time/(processed+1):.1f}s" if processed > 0 else "N/A"
            })
            continue
        
        # Vectorize
        result = vectorize_single_sample(
            test_id,
            output_dir,
            num_steps=num_steps,
            verbose=verbose
        )
        
        manifest['samples'][test_id] = result
        
        if result['status'] == 'success':
            processed += 1
            total_time += result['elapsed_seconds']
        else:
            failed += 1
        
        pbar.set_postfix({
            'done': processed,
            'skip': skipped,
            'fail': failed,
            'avg': f"{total_time/(processed+1):.1f}s" if processed > 0 else "N/A"
        })
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Compute summary statistics
    successful = [r for r in manifest['samples'].values() if r['status'] == 'success']
    
    if successful:
        avg_segments = np.mean([r['segments_after_merge'] for r in successful])
        avg_reduction = np.mean([r['reduction_rate'] for r in successful])
        avg_time = np.mean([r['elapsed_seconds'] for r in successful])
    else:
        avg_segments = 0
        avg_reduction = 0
        avg_time = 0
    
    summary = {
        'total_samples': len(test_ids),
        'processed': processed,
        'skipped': skipped,
        'failed': failed,
        'avg_segments_after_merge': avg_segments,
        'avg_reduction_rate': avg_reduction,
        'avg_time_per_sample': avg_time,
        'total_time': total_time
    }
    
    manifest['summary'] = summary
    
    # Re-save with summary
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"Processed: {processed}/{len(test_ids)}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"\nAverage segments (after merge): {avg_segments:.1f}")
    print(f"Average reduction rate: {avg_reduction*100:.1f}%")
    print(f"Average time per sample: {avg_time:.1f}s")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"\nManifest: {manifest_path}")
    print(f"{'='*70}\n")
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=77, help='Number of samples')
    parser.add_argument('--steps', type=int, default=150, help='Optimization steps')
    parser.add_argument('--no-resume', action='store_true', help='Re-process existing')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    summary = run_full_optimization_benchmark(
        num_samples=args.samples,
        num_steps=args.steps,
        resume=not args.no_resume,
        verbose=args.verbose
    )
    
    print("✅ Full optimization complete!")
    print(f"   Run benchmark to evaluate: python3 scripts/benchmark.py")
