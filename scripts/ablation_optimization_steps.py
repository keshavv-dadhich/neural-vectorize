"""
Quick ablation: Compare optimization steps (no neural model needed).

Tests different step counts with edge initialization:
- 30 steps (fast)
- 75 steps (medium)
- 150 steps (full, baseline)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
import json
import time
import io
from tqdm import tqdm
import argparse
from skimage.metrics import structural_similarity as ssim_metric

from vectorizers.optimize_v3 import AdvancedVectorizer
from vectorizers.initialize import EdgeInitializer
from config import RASTER_DEGRADED

try:
    from utils import svg_utils
    svg_to_raster = svg_utils.svg_to_raster
except ImportError:
    def svg_to_raster(svg_path, size=256):
        """Simple SVG to raster using cairosvg."""
        import cairosvg
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
        img = Image.open(io.BytesIO(png_data)).convert('L')
        return np.array(img) / 255.0

def compute_l2(pred, target):
    """Compute L2 error."""
    return float(np.mean((pred - target) ** 2))

def compute_ssim(pred, target):
    """Compute SSIM."""
    return float(ssim_metric(pred, target, data_range=1.0))

def count_svg_segments(svg_path):
    """Count number of segments in SVG."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(svg_path)
    root = tree.getroot()
    # Count path elements and polyline elements
    paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    polylines = root.findall('.//{http://www.w3.org/2000/svg}polyline')
    return len(paths) + len(polylines)


def evaluate_steps(
    sample_ids: list,
    steps: int,
    output_dir: Path = None,
    verbose: bool = True
):
    """Evaluate edge initialization with N optimization steps."""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create initializer and vectorizer once
    initializer = EdgeInitializer(
        canny_low=50,
        canny_high=150,
        epsilon_factor=0.002,
        min_contour_length=20
    )
    
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,
        lambda_curvature=0.1,
        lambda_intersection=0.3,
        lambda_complexity=0.005
    )
    
    results = []
    
    desc = f"Edge init + {steps} steps"
    for sample_id in tqdm(sample_ids, desc=desc, disable=not verbose):
        try:
            # Load degraded raster
            raster_path = RASTER_DEGRADED / f"{sample_id}_01.png"
            if not raster_path.exists():
                continue
            
            raster = np.array(Image.open(raster_path).convert('L')) / 255.0
            
            # Temporary paths for initialization
            init_svg = output_dir / f"{sample_id}_init.svg" if output_dir else Path(f"/tmp/{sample_id}_init.svg")
            opt_svg = output_dir / f"{sample_id}.svg" if output_dir else Path(f"/tmp/{sample_id}.svg")
            
            start_time = time.time()
            
            # Edge initialization
            initializer.initialize_from_raster(
                image_path=raster_path,
                output_svg_path=init_svg
            )
            
            # Optimize with specified steps
            vectorizer.optimize(
                svg_path=init_svg,
                target_image=raster,
                output_path=opt_svg,
                num_steps=steps,
                learning_rate=0.01,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            
            # Render optimized SVG back to raster for metrics
            predicted = svg_to_raster(opt_svg, size=256)
            
            # Compute metrics
            l2 = compute_l2(predicted, raster)
            ssim = compute_ssim(predicted, raster)
            segments = count_svg_segments(opt_svg)
            
            result = {
                'id': sample_id,
                'steps': steps,
                'l2': l2,
                'ssim': ssim,
                'segments': segments,
                'time': elapsed
            }
            results.append(result)
        
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    return results


def compute_statistics(results: list):
    """Compute mean and std of metrics."""
    if not results:
        return {}
    
    l2_values = [r['l2'] for r in results]
    ssim_values = [r['ssim'] for r in results]
    seg_values = [r['segments'] for r in results]
    time_values = [r['time'] for r in results]
    
    return {
        'l2_mean': np.mean(l2_values),
        'l2_std': np.std(l2_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'segments_mean': np.mean(seg_values),
        'segments_std': np.std(seg_values),
        'time_mean': np.mean(time_values),
        'time_std': np.std(time_values),
        'num_samples': len(results)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20, help='Number of test samples')
    parser.add_argument('--output', type=str, default='baselines/ablation_steps', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ABLATION STUDY: Optimization Step Count")
    print("=" * 60)
    
    # Get test sample IDs
    test_ids_path = Path('data/test_ids.txt')
    if test_ids_path.exists():
        with open(test_ids_path) as f:
            test_ids = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: use all degraded rasters
        test_ids = sorted([
            p.stem.rsplit('_', 1)[0]
            for p in RASTER_DEGRADED.glob('*_01.png')
        ])
    
    test_ids = test_ids[:args.samples]
    print(f"\nTesting on {len(test_ids)} samples\n")
    
    # Test configurations
    step_counts = [30, 75, 150]
    
    all_results = {}
    
    for steps in step_counts:
        print(f"\n{'=' * 60}")
        print(f"Testing: {steps} optimization steps")
        print(f"{'=' * 60}\n")
        
        output_dir = Path(args.output) / f"steps_{steps}"
        
        results = evaluate_steps(
            sample_ids=test_ids,
            steps=steps,
            output_dir=output_dir,
            verbose=True
        )
        
        stats = compute_statistics(results)
        all_results[f'steps_{steps}'] = {
            'steps': steps,
            'results': results,
            'stats': stats
        }
        
        print(f"\n📊 Results:")
        if stats:
            print(f"   L2: {stats['l2_mean']:.4f} ± {stats['l2_std']:.4f}")
        else:
            print("   No results (all samples failed)")
            continue
        print(f"   SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
        print(f"   Segments: {stats['segments_mean']:.1f} ± {stats['segments_std']:.1f}")
        print(f"   Time: {stats['time_mean']:.1f}s ± {stats['time_std']:.1f}s")
    
    # Save results
    output_path = Path(args.output) / 'ablation_steps_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Steps':<10} {'L2 Error':<18} {'SSIM':<18} {'Segments':<18} {'Time (s)':<15}")
    print("-" * 80)
    
    for steps in step_counts:
        key = f'steps_{steps}'
        if key in all_results and all_results[key]['stats']:
            stats = all_results[key]['stats']
            print(f"{steps:<10} "
                  f"{stats['l2_mean']:.4f} ± {stats['l2_std']:.3f}    "
                  f"{stats['ssim_mean']:.4f} ± {stats['ssim_std']:.3f}    "
                  f"{stats['segments_mean']:.1f} ± {stats['segments_std']:.1f}       "
                  f"{stats['time_mean']:.1f} ± {stats['time_std']:.1f}")
        else:
            print(f"{steps:<10} No results (all samples failed)")
    
    print("\n" + "=" * 80)
    
    # Compute quality vs speed tradeoff
    if 'steps_30' in all_results and 'steps_150' in all_results:
        fast_l2 = all_results['steps_30']['stats']['l2_mean']
        full_l2 = all_results['steps_150']['stats']['l2_mean']
        fast_time = all_results['steps_30']['stats']['time_mean']
        full_time = all_results['steps_150']['stats']['time_mean']
        
        l2_gap = ((fast_l2 - full_l2) / full_l2) * 100
        speedup = full_time / fast_time if fast_time > 0 else 0
        
        print(f"\n📊 30 vs 150 Steps:")
        print(f"   L2 gap: {l2_gap:+.1f}% (30 steps has {abs(l2_gap):.1f}% {'worse' if l2_gap > 0 else 'better'} L2)")
        print(f"   Speedup: {speedup:.2f}x faster with 30 steps")
        print(f"   Time saved: {full_time - fast_time:.1f}s per sample")


if __name__ == '__main__':
    main()
