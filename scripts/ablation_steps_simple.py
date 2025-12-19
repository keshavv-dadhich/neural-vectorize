"""
Quick ablation: Compare optimization step counts using existing baselines.

Tests: 30 vs 75 vs 150 steps
Uses: existing optimization_full SVGs as initialization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw
import json
import time
from tqdm import tqdm
import argparse
from skimage.metrics import structural_similarity as ssim_metric
import xml.etree.ElementTree as ET
import re

from vectorizers.optimize_v3 import AdvancedVectorizer
from config import RASTER_PERFECT, BASELINES, TEST_IDS


def svg_to_raster(svg_path, size=256):
    """Convert SVG to raster image using simple line rendering."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    
    for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path_elem.get('d', '')
        coords = []
        tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
        
        for token in tokens:
            match = re.search(r'([\d.]+),([\d.]+)', token)
            if match:
                x = float(match.group(1)) * size
                y = float(match.group(2)) * size
                coords.append((x, y))
        
        if len(coords) >= 2:
            for i in range(len(coords) - 1):
                draw.line([coords[i], coords[i+1]], fill=0, width=2)
    
    return np.array(img) / 255.0


def compute_l2(pred, target):
    """Compute L2 error."""
    return float(np.mean((pred - target) ** 2))


def compute_ssim(pred, target):
    """Compute SSIM."""
    return float(ssim_metric(pred, target, data_range=1.0))


def count_svg_segments(svg_path):
    """Count L commands in SVG."""
    return svg_path.read_text().count('L')


def evaluate_steps(sample_ids: list, steps: int, output_dir: Path = None):
    """Evaluate optimization with N steps."""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,
        lambda_curvature=0.1,
        lambda_intersection=0.3,
        lambda_complexity=0.005
    )
    
    results = []
    desc = f"{steps} steps"
    
    for sample_id in tqdm(sample_ids, desc=desc):
        try:
            # Load target
            raster_path = RASTER_PERFECT / f"{sample_id}_base.png"
            if not raster_path.exists():
                continue
            
            target = np.array(Image.open(raster_path).convert('L')) / 255.0
            
            # Use existing optimization_full SVG as initialization
            init_svg = BASELINES / 'optimization_full' / f"{sample_id}.svg"
            if not init_svg.exists():
                # Fallback to potrace
                init_svg = BASELINES / 'potrace' / f"{sample_id}.svg"
                if not init_svg.exists():
                    continue
            
            # Output path
            output_svg = output_dir / f"{sample_id}.svg" if output_dir else Path(f"/tmp/{sample_id}_{steps}.svg")
            
            start_time = time.time()
            
            # Optimize
            vectorizer.optimize(
                svg_path=init_svg,
                target_image=target,
                output_path=output_svg,
                num_steps=steps,
                learning_rate=0.01,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            
            # Compute metrics
            predicted = svg_to_raster(output_svg, size=256)
            l2 = compute_l2(predicted, target)
            ssim = compute_ssim(predicted, target)
            segments = count_svg_segments(output_svg)
            
            results.append({
                'id': sample_id,
                'steps': steps,
                'l2': l2,
                'ssim': ssim,
                'segments': segments,
                'time': elapsed
            })
        
        except Exception as e:
            print(f"\nError processing {sample_id}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation: optimization step count')
    parser.add_argument('--samples', type=int, default=15, help='Number of samples to test')
    args = parser.parse_args()
    
    # Get test samples
    test_ids = TEST_IDS.read_text().strip().split('\n')[:args.samples]
    
    print("=" * 60)
    print("ABLATION STUDY: Optimization Step Count")
    print("=" * 60)
    print(f"\nTesting on {len(test_ids)} samples\n")
    
    # Test different step counts
    all_results = {}
    for steps in [30, 75, 150]:
        print(f"\n{'='*60}")
        print(f"Testing: {steps} optimization steps")
        print(f"{'='*60}\n")
        
        output_dir = BASELINES.parent / 'baselines' / f'ablation_steps_{steps}'
        results = evaluate_steps(test_ids, steps, output_dir)
        
        all_results[steps] = results
        
        if results:
            l2_mean = np.mean([r['l2'] for r in results])
            l2_std = np.std([r['l2'] for r in results])
            ssim_mean = np.mean([r['ssim'] for r in results])
            time_mean = np.mean([r['time'] for r in results])
            
            print(f"\n📊 Results:")
            print(f"   L2: {l2_mean:.4f} ± {l2_std:.4f}")
            print(f"   SSIM: {ssim_mean:.4f}")
            print(f"   Time: {time_mean:.1f}s per sample")
        else:
            print(f"\n📊 Results:")
            print(f"   No results (all samples failed)")
    
    # Save results
    results_dir = BASELINES.parent / 'baselines' / 'ablation_steps'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'ablation_steps_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    print(f"{'Steps':<10} {'L2':<15} {'SSIM':<15} {'Time/Sample':<15}")
    print(f"{'-'*60}")
    for steps in [30, 75, 150]:
        results = all_results.get(steps, [])
        if results:
            l2_mean = np.mean([r['l2'] for r in results])
            ssim_mean = np.mean([r['ssim'] for r in results])
            time_mean = np.mean([r['time'] for r in results])
            print(f"{steps:<10} {l2_mean:<15.4f} {ssim_mean:<15.4f} {time_mean:<15.1f}s")
        else:
            print(f"{steps:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")


if __name__ == '__main__':
    main()
