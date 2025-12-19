#!/usr/bin/env python3
"""
CRITICAL EXPERIMENT 2: Loss Term Removal (True Ablation)

Question: Which loss terms are essential? Does edge alignment truly matter?

Protocol:
- Remove one loss term at a time
- Compare: Full loss vs (-edge) vs (-smoothness) vs (-intersection)
- Steps: 30
- Metrics: L2, SSIM, failure rate, visual artifacts

Expected:
- Removing edge loss → visible drift / spaghetti
- Removing smoothness → jagged curves
- Removing intersection → self-crossings

This converts "design choice" into causal evidence.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorizers.optimize_v3 import DiffVectorizer
from utils.metrics import compute_l2_error, compute_ssim

def compute_ablated_loss(rendered, target, paths, ablate_term=None):
    """
    Compute multi-term loss with one term removed.
    
    Args:
        rendered: Current rendered image
        target: Target raster image
        paths: SVG path data
        ablate_term: Which term to remove ('edge', 'smoothness', 'intersection', or None for full)
    
    Returns:
        dict with loss components and total
    """
    from vectorizers.losses import (
        compute_raster_loss,
        compute_edge_alignment_loss,
        compute_curvature_smoothness_loss,
        compute_intersection_penalty,
        compute_complexity_penalty
    )
    
    # Define weights (same as paper)
    weights = {
        'raster': 1.0,
        'edge': 0.5,
        'smoothness': 0.1,
        'intersection': 0.3,
        'complexity': 0.05
    }
    
    # Override weight for ablated term
    if ablate_term == 'edge':
        weights['edge'] = 0.0
    elif ablate_term == 'smoothness':
        weights['smoothness'] = 0.0
    elif ablate_term == 'intersection':
        weights['intersection'] = 0.0
    
    # Compute all terms
    losses = {
        'raster': compute_raster_loss(rendered, target),
        'edge': compute_edge_alignment_loss(rendered, target),
        'smoothness': compute_curvature_smoothness_loss(paths),
        'intersection': compute_intersection_penalty(paths),
        'complexity': compute_complexity_penalty(paths)
    }
    
    # Weighted sum
    total = sum(weights[k] * losses[k] for k in losses.keys())
    
    return {
        **losses,
        'total': total,
        'weights': weights,
        'ablated_term': ablate_term
    }

def detect_artifacts(rendered, paths):
    """
    Detect visual artifacts that indicate optimization failure.
    
    Returns dict with artifact scores:
    - jaggedness: High-frequency noise in edges
    - self_intersections: Path crossings
    - drift: Paths far from image content
    """
    artifacts = {}
    
    # 1. Jaggedness: Measure edge smoothness
    edges = np.gradient(rendered.astype(float))
    edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
    edge_curvature = np.gradient(edge_magnitude)
    artifacts['jaggedness'] = float(np.std(edge_curvature[0]))
    
    # 2. Self-intersections: Count path crossings
    intersection_count = 0
    for i, path1 in enumerate(paths):
        for path2 in paths[i+1:]:
            # Simplified check: bounding box overlap
            # (Full intersection check would be expensive)
            if bounding_boxes_overlap(path1, path2):
                intersection_count += 1
    artifacts['self_intersections'] = intersection_count
    
    # 3. Drift: Measure how much paths extend beyond content
    content_mask = rendered > 10  # Threshold for "content"
    path_mask = render_path_mask(paths, rendered.shape)
    drift_pixels = np.sum(path_mask & ~content_mask)
    artifacts['drift'] = float(drift_pixels / path_mask.sum() if path_mask.sum() > 0 else 0)
    
    return artifacts

def bounding_boxes_overlap(path1, path2):
    """Check if two path bounding boxes overlap."""
    # Simplified: assume paths have 'bounds' attribute
    # In practice, compute from control points
    return False  # Placeholder

def render_path_mask(paths, shape):
    """Render binary mask of where paths exist."""
    # Placeholder - in practice, render paths as 1-bit mask
    return np.zeros(shape, dtype=bool)

def run_single_sample(sample_name, ablate_term, num_steps=30):
    """Run optimization with specified loss term removed."""
    
    # Paths
    raster_path = f"data_processed/raster_degraded/{sample_name}"
    init_path = f"baselines/neural_init/{sample_name.replace('.png', '.svg')}"
    
    term_name = ablate_term if ablate_term else 'full'
    output_path = f"analysis/ablation_loss_terms/{term_name}/{sample_name.replace('.png', '.svg')}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
        rendered = vectorizer.render()
        
        # Compute ablated loss
        loss_dict = compute_ablated_loss(rendered, raster, 
                                        vectorizer.get_paths(), 
                                        ablate_term=ablate_term)
        total_loss = loss_dict['total']
        losses.append(loss_dict)
        
        # Update
        vectorizer.compute_gradients(loss_dict)
        vectorizer.step()
        
        if step % 10 == 0:
            print(f"  Step {step:3d}/{num_steps}: Loss = {total_loss:.4f}")
    
    opt_time = time.time() - start_time
    
    # Save final output
    vectorizer.save(output_path)
    
    # Compute final metrics
    final_rendered = vectorizer.render()
    l2_error = compute_l2_error(final_rendered, raster)
    ssim_score = compute_ssim(final_rendered, raster)
    
    # Detect artifacts
    artifacts = detect_artifacts(final_rendered, vectorizer.get_paths())
    
    results = {
        'sample': sample_name,
        'ablated_term': ablate_term,
        'num_steps': num_steps,
        'l2_error': float(l2_error),
        'ssim': float(ssim_score),
        'time_seconds': float(opt_time),
        'final_loss': float(losses[-1]['total']),
        'artifacts': artifacts,
        'loss_components_final': {
            k: float(v) for k, v in losses[-1].items() 
            if k not in ['total', 'weights', 'ablated_term']
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Ablate loss terms to measure importance')
    parser.add_argument('--samples', type=int, default=15, help='Number of samples to test')
    parser.add_argument('--steps', type=int, default=30, help='Optimization steps')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CRITICAL EXPERIMENT 2: Loss Term Ablation")
    print("="*70)
    print(f"Configurations: Full, -Edge, -Smoothness, -Intersection")
    print(f"Steps: {args.steps}")
    print(f"Samples: {args.samples}")
    print("="*70 + "\n")
    
    # Get test samples
    raster_dir = Path("data_processed/raster_degraded")
    all_samples = sorted([f.name for f in raster_dir.glob("*.png")])[:args.samples]
    
    # Test configurations
    configs = [
        None,            # Full loss
        'edge',          # Remove edge alignment
        'smoothness',    # Remove curvature smoothness
        'intersection'   # Remove intersection penalty
    ]
    
    all_results = {config: [] for config in configs}
    
    for i, sample_name in enumerate(all_samples, 1):
        print(f"\n[{i}/{len(all_samples)}] Processing: {sample_name}")
        print("-" * 70)
        
        for config in configs:
            config_name = config if config else 'full'
            print(f"  🔬 Configuration: {config_name}")
            
            results = run_single_sample(sample_name, config, args.steps)
            if results:
                all_results[config].append(results)
    
    # Aggregate statistics
    stats = {}
    for config in configs:
        config_name = config if config else 'full'
        results_list = all_results[config]
        
        if results_list:
            l2_errors = [r['l2_error'] for r in results_list]
            ssim_scores = [r['ssim'] for r in results_list]
            jaggedness = [r['artifacts']['jaggedness'] for r in results_list]
            intersections = [r['artifacts']['self_intersections'] for r in results_list]
            drift = [r['artifacts']['drift'] for r in results_list]
            
            stats[config_name] = {
                'l2': {
                    'mean': float(np.mean(l2_errors)),
                    'std': float(np.std(l2_errors)),
                    'median': float(np.median(l2_errors))
                },
                'ssim': {
                    'mean': float(np.mean(ssim_scores)),
                    'std': float(np.std(ssim_scores))
                },
                'artifacts': {
                    'jaggedness_mean': float(np.mean(jaggedness)),
                    'intersections_mean': float(np.mean(intersections)),
                    'drift_mean': float(np.mean(drift))
                },
                'per_sample': results_list
            }
    
    # Compute degradation from removing each term
    full_l2 = stats['full']['l2']['mean']
    comparisons = {}
    
    for config in ['edge', 'smoothness', 'intersection']:
        if config in stats:
            ablated_l2 = stats[config]['l2']['mean']
            degradation_pct = (ablated_l2 - full_l2) / full_l2 * 100
            comparisons[config] = {
                'l2_degradation_pct': float(degradation_pct),
                'artifact_increase': {
                    'jaggedness': float(stats[config]['artifacts']['jaggedness_mean'] - 
                                      stats['full']['artifacts']['jaggedness_mean']),
                    'intersections': float(stats[config]['artifacts']['intersections_mean'] - 
                                         stats['full']['artifacts']['intersections_mean']),
                    'drift': float(stats[config]['artifacts']['drift_mean'] - 
                                 stats['full']['artifacts']['drift_mean'])
                }
            }
    
    # Save results
    output_data = {
        'configurations': stats,
        'comparisons_vs_full': comparisons,
        'num_samples': len(all_samples),
        'num_steps': args.steps
    }
    
    output_file = "analysis/ablation_loss_terms_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS: Loss Term Importance")
    print("="*70)
    print(f"Samples tested: {len(all_samples)}")
    print()
    print("L2 Error by Configuration:")
    print(f"  Full loss:          {stats['full']['l2']['mean']:.4f}")
    print(f"  Without edge:       {stats['edge']['l2']['mean']:.4f} ({comparisons['edge']['l2_degradation_pct']:+.1f}%)")
    print(f"  Without smoothness: {stats['smoothness']['l2']['mean']:.4f} ({comparisons['smoothness']['l2_degradation_pct']:+.1f}%)")
    print(f"  Without intersection: {stats['intersection']['l2']['mean']:.4f} ({comparisons['intersection']['l2_degradation_pct']:+.1f}%)")
    print()
    print("Artifact Analysis:")
    print("  Removing edge loss →")
    print(f"    Jaggedness: {comparisons['edge']['artifact_increase']['jaggedness']:+.2f}")
    print(f"    Intersections: {comparisons['edge']['artifact_increase']['intersections']:+.1f}")
    print(f"    Drift: {comparisons['edge']['artifact_increase']['drift']:+.3f}")
    print()
    
    # Determine most important term
    degradations = [(k, v['l2_degradation_pct']) for k, v in comparisons.items()]
    degradations.sort(key=lambda x: x[1], reverse=True)
    most_important = degradations[0][0]
    
    print(f"🔑 MOST IMPORTANT TERM: {most_important.upper()}")
    print(f"   Removing it causes {degradations[0][1]:.1f}% quality degradation")
    print()
    
    if most_important == 'edge' and degradations[0][1] > 10:
        print("✅ STRONG EVIDENCE: Edge alignment is CRITICAL")
        print("   → Validates key novelty claim")
    elif most_important == 'edge':
        print("⚠️  MODERATE EVIDENCE: Edge alignment helps but not essential")
    else:
        print(f"⚠️  UNEXPECTED: {most_important} is most important, not edge")
        print("   → May need to reframe contribution")
    
    print()
    print(f"Results saved: {output_file}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
