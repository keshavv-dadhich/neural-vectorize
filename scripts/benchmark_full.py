"""
Comprehensive benchmark of advanced optimizer on full test set.

Compares:
- Potrace baseline
- Old optimization (basic L2)
- New optimization (multi-term loss with edge alignment)
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import json
import time

import config
from vectorizers.optimize_v3 import AdvancedVectorizer

# Import metrics utilities
try:
    from utils import metrics, svg_utils
    compute_l2 = metrics.compute_l2
    compute_ssim = metrics.compute_ssim
    svg_to_raster = svg_utils.svg_to_raster
except ImportError:
    # Fallback: define inline
    from skimage.metrics import structural_similarity as ssim_metric
    import xml.etree.ElementTree as ET
    import re
    
    def compute_l2(pred, target):
        return np.mean((pred - target) ** 2)
    
    def compute_ssim(pred, target):
        return ssim_metric(pred, target, data_range=1.0)
    
    def svg_to_raster(svg_path, size=256):
        """Simple SVG to raster conversion."""
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)
        
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
            d = path_elem.get('d', '')
            coords = []
            tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
            for token in tokens:
                match = re.search(r'([\d.]+),([\d.]+)', token)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    coords.append((x, y))
            
            if len(coords) >= 2:
                for i in range(len(coords) - 1):
                    draw.line([coords[i], coords[i+1]], fill=0, width=2)
        
        return np.array(img) / 255.0

RASTER_PERFECT = config.RASTER_PERFECT
TEST_IDS = config.TEST_IDS
BASELINES = config.BASELINES


def run_full_benchmark(
    output_dir: Path = Path('baselines/advanced_full'),
    num_samples: int = None
):
    """Run advanced optimizer on full test set."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test IDs
    test_ids = TEST_IDS.read_text().strip().split('\n')
    if num_samples:
        test_ids = test_ids[:num_samples]
    
    print(f"Running advanced optimization on {len(test_ids)} test samples...")
    print(f"Output directory: {output_dir}")
    
    # Initialize vectorizer
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,        # CRITICAL innovation
        lambda_curvature=0.1,
        lambda_intersection=0.3,
        lambda_complexity=0.005
    )
    
    results = []
    failed = []
    
    start_time = time.time()
    
    for test_id in tqdm(test_ids, desc="Processing"):
        try:
            # Load target
            raster_path = RASTER_PERFECT / f"{test_id}_base.png"
            if not raster_path.exists():
                failed.append(test_id)
                continue
            
            target = np.array(Image.open(raster_path).convert('L')) / 255.0
            
            # Use optimization_full SVG as initialization (basic L2 optimized)
            init_svg_path = BASELINES / 'optimization_full' / f"{test_id}.svg"
            if not init_svg_path.exists():
                # Fallback to Potrace
                init_svg_path = BASELINES / 'potrace' / f"{test_id}.svg"
                if not init_svg_path.exists():
                    failed.append(test_id)
                    continue
            
            # Optimize
            output_svg_path = output_dir / f"{test_id}.svg"
            sample_start = time.time()
            
            vectorizer.optimize(
                init_svg_path,
                target,
                output_svg_path,
                num_steps=150,
                learning_rate=0.01,
                verbose=False
            )
            
            sample_time = time.time() - sample_start
            
            # Compute metrics
            predicted = svg_to_raster(output_svg_path, size=256)
            l2 = compute_l2(predicted, target)
            ssim = compute_ssim(predicted, target)
            
            # Count segments
            svg_text = output_svg_path.read_text()
            num_segments = svg_text.count('L')
            
            # File size
            file_size = output_svg_path.stat().st_size
            
            results.append({
                'id': test_id,
                'l2': float(l2),
                'ssim': float(ssim),
                'segments': num_segments,
                'file_size': file_size,
                'time': sample_time
            })
            
        except Exception as e:
            print(f"Error processing {test_id}: {str(e)}")
            failed.append(test_id)
            continue
    
    total_time = time.time() - start_time
    
    # Compute aggregates
    if results:
        avg_l2 = np.mean([r['l2'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_segments = np.mean([r['segments'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        
        summary = {
            'num_samples': len(results),
            'num_failed': len(failed),
            'failed_ids': failed,
            'metrics': {
                'l2_mean': float(avg_l2),
                'l2_std': float(np.std([r['l2'] for r in results])),
                'ssim_mean': float(avg_ssim),
                'ssim_std': float(np.std([r['ssim'] for r in results])),
                'segments_mean': float(avg_segments),
                'segments_std': float(np.std([r['segments'] for r in results])),
                'time_per_sample': float(avg_time),
                'total_time': float(total_time)
            },
            'results': results
        }
        
        # Save
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Benchmark complete!")
        print(f"   Successful: {len(results)}/{len(test_ids)}")
        print(f"   Failed: {len(failed)}")
        print(f"\nMetrics:")
        print(f"   L2: {avg_l2:.4f} ± {np.std([r['l2'] for r in results]):.4f}")
        print(f"   SSIM: {avg_ssim:.4f} ± {np.std([r['ssim'] for r in results]):.4f}")
        print(f"   Segments: {avg_segments:.1f} ± {np.std([r['segments'] for r in results]):.1f}")
        print(f"   Time per sample: {avg_time:.1f}s")
        print(f"   Total time: {total_time/60:.1f} minutes")
        
        return summary
    else:
        print("No successful results!")
        return None


def compare_with_baselines(advanced_results_path: Path):
    """Compare advanced results with Potrace and basic optimization."""
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    
    # Load advanced results
    with open(advanced_results_path, 'r') as f:
        advanced = json.load(f)
    
    # Load Potrace results (if available)
    potrace_results_path = BASELINES / 'potrace' / 'results.json'
    if potrace_results_path.exists():
        with open(potrace_results_path, 'r') as f:
            potrace = json.load(f)
    else:
        potrace = None
    
    # Load basic optimization results (if available)
    basic_results_path = BASELINES / 'optimization_full' / 'results.json'
    if basic_results_path.exists():
        with open(basic_results_path, 'r') as f:
            basic = json.load(f)
    else:
        basic = None
    
    # Print comparison table
    print(f"\n{'Method':<25} {'L2':<12} {'SSIM':<12} {'Segments':<12}")
    print("-" * 70)
    
    if potrace:
        print(f"{'Potrace':<25} {potrace['metrics']['l2_mean']:<12.4f} "
              f"{potrace['metrics'].get('ssim_mean', 0):<12.4f} "
              f"{potrace['metrics'].get('segments_mean', 0):<12.1f}")
    
    if basic:
        print(f"{'Basic Optimization':<25} {basic['metrics']['l2_mean']:<12.4f} "
              f"{basic['metrics'].get('ssim_mean', 0):<12.4f} "
              f"{basic['metrics'].get('segments_mean', 0):<12.1f}")
    
    print(f"{'Advanced (Edge-Aware)':<25} {advanced['metrics']['l2_mean']:<12.4f} "
          f"{advanced['metrics']['ssim_mean']:<12.4f} "
          f"{advanced['metrics']['segments_mean']:<12.1f}")
    
    # Compute improvements
    if potrace:
        l2_improvement = (1 - advanced['metrics']['l2_mean'] / potrace['metrics']['l2_mean']) * 100
        print(f"\n✅ L2 improvement over Potrace: {l2_improvement:.1f}%")
    
    if basic:
        l2_improvement = (1 - advanced['metrics']['l2_mean'] / basic['metrics']['l2_mean']) * 100
        segment_reduction = (1 - advanced['metrics']['segments_mean'] / basic['metrics'].get('segments_mean', 1)) * 100
        print(f"✅ L2 improvement over basic: {l2_improvement:.1f}%")
        print(f"✅ Segment reduction: {segment_reduction:.1f}%")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='baselines/advanced_full')
    parser.add_argument('--samples', type=int, default=None, help='Limit to N samples')
    parser.add_argument('--compare', action='store_true', help='Compare with baselines')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    if not args.compare:
        # Run benchmark
        run_full_benchmark(
            output_dir=output_path,
            num_samples=args.samples
        )
    
    # Compare with baselines
    results_path = output_path / 'results.json'
    if results_path.exists():
        compare_with_baselines(results_path)
