#!/usr/bin/env python3
"""
CRITICAL EXPERIMENT 3: Complexity Scaling Study

Question: How does performance scale with SVG complexity?

Protocol:
- Bin test icons by path count: Low (<30), Medium (30-70), High (>70)
- Measure: L2, time, failure rate for each complexity bin
- Steps: 30

Expected:
- Time scales linearly with complexity
- Quality degrades gracefully for complex icons
- Demonstrates robustness across difficulty levels

This is crucial for graphics reviewers who care about scalability.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorizers.optimize_v3 import DiffVectorizer
from vectorizers.losses import compute_multi_term_loss
from utils.metrics import compute_l2_error, compute_ssim

def count_svg_paths(svg_path):
    """Count number of paths in SVG file."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Handle namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        paths = root.findall('.//svg:path', ns)
        
        # If no namespace, try without
        if not paths:
            paths = root.findall('.//path')
        
        return len(paths)
    except Exception as e:
        print(f"⚠️  Error counting paths in {svg_path}: {e}")
        return 0

def classify_complexity(path_count):
    """Classify icon complexity by path count."""
    if path_count < 30:
        return 'low'
    elif path_count < 70:
        return 'medium'
    else:
        return 'high'

def run_single_sample(sample_name, num_steps=30):
    """Run optimization and return results with complexity info."""
    
    # Paths
    raster_path = f"data_processed/raster_degraded/{sample_name}"
    init_path = f"baselines/neural_init/{sample_name.replace('.png', '.svg')}"
    ground_truth_path = f"data_processed/svg_clean/{sample_name.replace('.png', '.svg')}"
    output_path = f"analysis/ablation_complexity/{sample_name.replace('.png', '.svg')}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Count paths in ground truth (original complexity)
    if not os.path.exists(ground_truth_path):
        print(f"⚠️  Skipping {sample_name}: Ground truth not found")
        return None
    
    path_count = count_svg_paths(ground_truth_path)
    complexity = classify_complexity(path_count)
    
    # Load raster
    raster = np.array(Image.open(raster_path).convert('L'))
    
    if not os.path.exists(init_path):
        print(f"⚠️  Skipping {sample_name}: Neural init not found")
        return None
    
    # Initialize vectorizer
    vectorizer = DiffVectorizer(svg_path=init_path)
    
    # Run optimization
    start_time = time.time()
    
    losses = []
    for step in range(num_steps):
        step_start = time.time()
        
        rendered = vectorizer.render()
        loss_dict = compute_multi_term_loss(rendered, raster, vectorizer.get_paths())
        total_loss = loss_dict['total']
        
        vectorizer.compute_gradients(loss_dict)
        vectorizer.step()
        
        step_time = time.time() - step_start
        losses.append({
            'loss': float(total_loss),
            'time': float(step_time)
        })
        
        if step % 10 == 0:
            print(f"  Step {step:3d}/{num_steps}: Loss = {total_loss:.4f}, Time = {step_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Save final output
    vectorizer.save(output_path)
    
    # Compute final metrics
    final_rendered = vectorizer.render()
    l2_error = compute_l2_error(final_rendered, raster)
    ssim_score = compute_ssim(final_rendered, raster)
    
    # Check for failure (threshold: L2 > 0.35)
    failed = l2_error > 0.35
    
    # Analyze time per step across optimization
    times_per_step = [l['time'] for l in losses]
    
    results = {
        'sample': sample_name,
        'complexity': {
            'path_count': path_count,
            'category': complexity
        },
        'quality': {
            'l2_error': float(l2_error),
            'ssim': float(ssim_score),
            'failed': failed
        },
        'performance': {
            'total_time': float(total_time),
            'time_per_step_mean': float(np.mean(times_per_step)),
            'time_per_step_std': float(np.std(times_per_step)),
            'time_per_step_first10': float(np.mean(times_per_step[:10])),
            'time_per_step_last10': float(np.mean(times_per_step[-10:]))
        },
        'optimization': {
            'initial_loss': float(losses[0]['loss']),
            'final_loss': float(losses[-1]['loss']),
            'loss_reduction_pct': float((losses[0]['loss'] - losses[-1]['loss']) / losses[0]['loss'] * 100)
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze complexity scaling')
    parser.add_argument('--samples', type=int, default=30, 
                       help='Number of samples (will bin by complexity)')
    parser.add_argument('--steps', type=int, default=30, help='Optimization steps')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CRITICAL EXPERIMENT 3: Complexity Scaling Study")
    print("="*70)
    print(f"Complexity bins: Low (<30 paths), Medium (30-70), High (>70)")
    print(f"Steps: {args.steps}")
    print(f"Target samples: {args.samples}")
    print("="*70 + "\n")
    
    # Get test samples
    raster_dir = Path("data_processed/raster_degraded")
    all_samples = sorted([f.name for f in raster_dir.glob("*.png")])[:args.samples]
    
    print(f"Processing {len(all_samples)} samples...\n")
    
    # Run all samples
    all_results = []
    
    for i, sample_name in enumerate(all_samples, 1):
        print(f"\n[{i}/{len(all_samples)}] Processing: {sample_name}")
        print("-" * 70)
        
        results = run_single_sample(sample_name, args.steps)
        if results:
            all_results.append(results)
            print(f"  ✓ Complexity: {results['complexity']['category']} ({results['complexity']['path_count']} paths)")
            print(f"    L2: {results['quality']['l2_error']:.4f}, Time: {results['performance']['total_time']:.1f}s")
    
    # Bin by complexity
    binned_results = defaultdict(list)
    for r in all_results:
        binned_results[r['complexity']['category']].append(r)
    
    # Compute statistics per bin
    stats_by_complexity = {}
    
    for complexity in ['low', 'medium', 'high']:
        results_list = binned_results[complexity]
        
        if results_list:
            path_counts = [r['complexity']['path_count'] for r in results_list]
            l2_errors = [r['quality']['l2_error'] for r in results_list]
            times = [r['performance']['total_time'] for r in results_list]
            failures = [r['quality']['failed'] for r in results_list]
            
            stats_by_complexity[complexity] = {
                'count': len(results_list),
                'path_count': {
                    'mean': float(np.mean(path_counts)),
                    'std': float(np.std(path_counts)),
                    'min': int(np.min(path_counts)),
                    'max': int(np.max(path_counts))
                },
                'l2_error': {
                    'mean': float(np.mean(l2_errors)),
                    'std': float(np.std(l2_errors)),
                    'median': float(np.median(l2_errors))
                },
                'time': {
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times)),
                    'median': float(np.median(times))
                },
                'failure_rate': float(sum(failures) / len(failures) * 100),
                'per_sample': results_list
            }
    
    # Analyze scaling trends
    scaling_analysis = {}
    
    if 'low' in stats_by_complexity and 'high' in stats_by_complexity:
        low_time = stats_by_complexity['low']['time']['mean']
        high_time = stats_by_complexity['high']['time']['mean']
        low_paths = stats_by_complexity['low']['path_count']['mean']
        high_paths = stats_by_complexity['high']['path_count']['mean']
        
        time_increase_factor = high_time / low_time if low_time > 0 else 0
        complexity_increase_factor = high_paths / low_paths if low_paths > 0 else 0
        
        scaling_analysis['time_scaling'] = {
            'time_increase_factor': float(time_increase_factor),
            'complexity_increase_factor': float(complexity_increase_factor),
            'scaling_ratio': float(time_increase_factor / complexity_increase_factor if complexity_increase_factor > 0 else 0),
            'interpretation': 'sub-linear' if time_increase_factor < complexity_increase_factor else 
                            'linear' if abs(time_increase_factor - complexity_increase_factor) < 0.2 else 
                            'super-linear'
        }
        
        low_l2 = stats_by_complexity['low']['l2_error']['mean']
        high_l2 = stats_by_complexity['high']['l2_error']['mean']
        quality_degradation = (high_l2 - low_l2) / low_l2 * 100 if low_l2 > 0 else 0
        
        scaling_analysis['quality_scaling'] = {
            'l2_increase_pct': float(quality_degradation),
            'graceful_degradation': quality_degradation < 30  # Threshold for acceptable degradation
        }
    
    # Save results
    output_data = {
        'summary': stats_by_complexity,
        'scaling_analysis': scaling_analysis,
        'num_samples': len(all_results),
        'num_steps': args.steps
    }
    
    output_file = "analysis/ablation_complexity_scaling_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS: Complexity Scaling")
    print("="*70)
    print(f"Total samples: {len(all_results)}")
    print()
    
    for complexity in ['low', 'medium', 'high']:
        if complexity in stats_by_complexity:
            s = stats_by_complexity[complexity]
            print(f"{complexity.upper()} Complexity ({s['count']} samples):")
            print(f"  Path count: {s['path_count']['mean']:.1f} ± {s['path_count']['std']:.1f} ({s['path_count']['min']}-{s['path_count']['max']})")
            print(f"  L2 error:   {s['l2_error']['mean']:.4f} ± {s['l2_error']['std']:.4f}")
            print(f"  Time:       {s['time']['mean']:.1f}s ± {s['time']['std']:.1f}s")
            print(f"  Failures:   {s['failure_rate']:.1f}%")
            print()
    
    if scaling_analysis:
        print("Scaling Analysis:")
        if 'time_scaling' in scaling_analysis:
            ts = scaling_analysis['time_scaling']
            print(f"  Time scaling: {ts['interpretation']}")
            print(f"    Complexity ↑ {ts['complexity_increase_factor']:.1f}×")
            print(f"    Time ↑ {ts['time_increase_factor']:.1f}×")
            print(f"    Ratio: {ts['scaling_ratio']:.2f}")
        
        if 'quality_scaling' in scaling_analysis:
            qs = scaling_analysis['quality_scaling']
            print(f"  Quality degradation: {qs['l2_increase_pct']:+.1f}%")
            print(f"  Graceful? {'✅ Yes' if qs['graceful_degradation'] else '❌ No'}")
        print()
    
    # Overall assessment
    print("Assessment:")
    if scaling_analysis.get('time_scaling', {}).get('interpretation') in ['linear', 'sub-linear']:
        print("  ✅ Time scales efficiently with complexity")
    else:
        print("  ⚠️  Time scaling is super-linear (potential concern)")
    
    if scaling_analysis.get('quality_scaling', {}).get('graceful_degradation'):
        print("  ✅ Quality degrades gracefully")
    else:
        print("  ⚠️  Significant quality loss for complex icons")
    
    # Check failure rates
    all_failure_rates = [s['failure_rate'] for s in stats_by_complexity.values()]
    max_failure_rate = max(all_failure_rates) if all_failure_rates else 0
    
    if max_failure_rate < 10:
        print("  ✅ Low failure rate across all complexity levels")
    elif max_failure_rate < 25:
        print("  ⚠️  Moderate failure rate for complex icons")
    else:
        print("  ❌ High failure rate (major concern)")
    
    print()
    print(f"Results saved: {output_file}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
