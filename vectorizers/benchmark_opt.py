"""
Benchmark: Optimization-based Vectorizer vs Potrace

Runs our new vectorizer on test set and compares against Potrace baseline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import json
from tqdm import tqdm

from config import *
from vectorizers.run_vectorizer import vectorize
from scripts.benchmark import rasterize_svg_simple, compute_l2_error, compute_ssim, count_svg_primitives


def run_optimization_vectorizer_on_test_set(
    max_samples: int = 10,  # Start with small subset
    verbose: bool = True
):
    """
    Run our optimization vectorizer on test set.
    
    Args:
        max_samples: Maximum number of test samples to process
        verbose: Print progress
    """
    # Load test IDs
    with open(TEST_IDS) as f:
        test_ids = [line.strip() for line in f]
    
    # Limit for testing
    test_ids = test_ids[:max_samples]
    
    output_dir = BASELINES / 'optimization'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RUNNING OPTIMIZATION VECTORIZER ON {len(test_ids)} TEST SAMPLES")
        print(f"{'='*70}\n")
    
    for icon_id in tqdm(test_ids, desc="Vectorizing"):
        # Input: perfect raster
        raster_path = RASTER_PERFECT / f"{icon_id}_base.png"
        
        if not raster_path.exists():
            continue
        
        # Output SVG
        output_svg = output_dir / f"{icon_id}.svg"
        
        try:
            vectorize(
                raster_path,
                output_svg,
                num_opt_steps=200,  # Moderate for speed
                simplify_after=True,
                verbose=False  # Suppress per-sample output
            )
        except Exception as e:
            if verbose:
                print(f"Error on {icon_id}: {e}")
            continue
    
    print(f"\n✅ Vectorization complete: {output_dir}")
    return output_dir


def benchmark_optimization_vs_potrace(
    optimization_dir: Path,
    verbose: bool = True
):
    """
    Compare optimization vectorizer to Potrace on test set.
    
    Args:
        optimization_dir: Directory with our optimized SVGs
        verbose: Print detailed results
    """
    potrace_dir = BASELINES / 'potrace'
    
    # Collect results
    results = {
        'optimization': {'l2': [], 'ssim': [], 'segments': [], 'size_kb': []},
        'potrace': {'l2': [], 'ssim': [], 'segments': [], 'size_kb': []},
        'ground_truth': {'l2': [], 'ssim': [], 'segments': [], 'size_kb': []}
    }
    
    # Load test IDs
    with open(TEST_IDS) as f:
        test_ids = [line.strip() for line in f]
    
    if verbose:
        print(f"\n{'='*70}")
        print("BENCHMARKING: Optimization vs Potrace vs Ground Truth")
        print(f"{'='*70}\n")
    
    for icon_id in tqdm(test_ids, desc="Benchmarking"):
        # Paths
        raster_path = RASTER_PERFECT / f"{icon_id}_base.png"
        gt_svg = SVG_CLEAN / f"{icon_id}.svg"
        opt_svg = optimization_dir / f"{icon_id}.svg"
        pot_svg = potrace_dir / f"{icon_id}.svg"
        
        # Check all exist
        if not all(p.exists() for p in [raster_path, gt_svg, pot_svg]):
            continue
        
        if not opt_svg.exists():
            continue  # Skip if optimization failed
        
        # Load target
        target = np.array(Image.open(raster_path).convert('L')) / 255.0
        
        # Benchmark each method
        for method, svg_path in [
            ('ground_truth', gt_svg),
            ('potrace', pot_svg),
            ('optimization', opt_svg)
        ]:
            # Rasterize SVG
            rendered = rasterize_svg_simple(svg_path, size=256)
            
            # Compute metrics
            l2 = compute_l2_error(target, rendered)
            ssim = compute_ssim(target, rendered)
            
            # Count segments
            primitives = count_svg_primitives(svg_path)
            segments = primitives['segments']  # Fixed: was 'total_segments'
            
            size_kb = svg_path.stat().st_size / 1024
            
            results[method]['l2'].append(l2)
            results[method]['ssim'].append(ssim)
            results[method]['segments'].append(segments)
            results[method]['size_kb'].append(size_kb)
    
    # Compute averages
    summary = {}
    for method in results:
        if results[method]['l2']:
            summary[method] = {
                'l2_mean': np.mean(results[method]['l2']),
                'l2_std': np.std(results[method]['l2']),
                'ssim_mean': np.mean(results[method]['ssim']),
                'ssim_std': np.std(results[method]['ssim']),
                'segments_mean': np.mean(results[method]['segments']),
                'segments_std': np.std(results[method]['segments']),
                'size_kb_mean': np.mean(results[method]['size_kb']),
                'size_kb_std': np.std(results[method]['size_kb']),
                'num_samples': len(results[method]['l2'])
            }
    
    # Save results
    output_file = OUTPUTS / 'metrics' / 'optimization_benchmark.json'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'detailed': results
        }, f, indent=2)
    
    # Print summary table
    if verbose:
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"\n{'Method':<20} {'L2 Error↓':<12} {'SSIM↑':<10} {'Segments↓':<12} {'Size(KB)↓':<10}")
        print(f"{'-'*70}")
        
        for method in ['ground_truth', 'potrace', 'optimization']:
            if method in summary:
                s = summary[method]
                print(f"{method:<20} "
                      f"{s['l2_mean']:>6.3f}±{s['l2_std']:.3f}  "
                      f"{s['ssim_mean']:>6.3f}  "
                      f"{s['segments_mean']:>7.1f}  "
                      f"{s['size_kb_mean']:>6.2f}")
        
        print(f"\n{summary['optimization']['num_samples']} test samples")
        print(f"\n✅ Results saved to: {output_file}")
        print(f"{'='*70}\n")
    
    return summary


def main():
    """Run complete benchmark."""
    # Step 1: Run optimization vectorizer
    opt_dir = run_optimization_vectorizer_on_test_set(
        max_samples=5,  # Just 5 for demonstration
        verbose=True
    )
    
    # Step 2: Benchmark
    summary = benchmark_optimization_vs_potrace(opt_dir, verbose=True)
    
    # Step 3: Analysis
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if 'optimization' in summary and 'potrace' in summary:
        opt = summary['optimization']
        pot = summary['potrace']
        
        l2_improvement = 100 * (pot['l2_mean'] - opt['l2_mean']) / pot['l2_mean']
        seg_improvement = 100 * (pot['segments_mean'] - opt['segments_mean']) / pot['segments_mean']
        
        print(f"\nOptimization vs Potrace:")
        print(f"  L2 Error: {l2_improvement:+.1f}% {'(BETTER)' if l2_improvement > 0 else '(WORSE)'}")
        print(f"  Segments: {seg_improvement:+.1f}% {'(SIMPLER)' if seg_improvement > 0 else '(MORE COMPLEX)'}")
        
        if l2_improvement > 0 and seg_improvement < 0:
            print(f"\n✅ SUCCESS: Better reconstruction with more segments")
            print(f"   (Trading complexity for accuracy - expected for gradient-based method)")
        elif l2_improvement > 0:
            print(f"\n🎯 EXCELLENT: Better reconstruction AND simpler!")
        
        print("\n" + "="*70)


if __name__ == '__main__':
    main()
