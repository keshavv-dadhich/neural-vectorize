#!/usr/bin/env python3
"""
Statistical significance tests for vectorization results.
Computes t-tests, effect sizes, and confidence intervals.

Usage:
    python3 scripts/statistical_analysis.py
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import sys

def load_results():
    """Load benchmark results."""
    results_file = Path("baselines/optimization_full/results.json")
    
    if not results_file.exists():
        print(f"❌ ERROR: Results file not found: {results_file}")
        print("   Run benchmark_full.py first to generate results.")
        sys.exit(1)
    
    with open(results_file) as f:
        return json.load(f)

def extract_method_scores(results):
    """Extract L2 scores by method."""
    methods = {}
    
    for sample_data in results:
        sample_name = sample_data.get('sample', 'unknown')
        
        for method_name, metrics in sample_data.items():
            if method_name == 'sample':
                continue
            
            if method_name not in methods:
                methods[method_name] = {'samples': [], 'l2_errors': [], 'ssim': [], 'times': []}
            
            methods[method_name]['samples'].append(sample_name)
            methods[method_name]['l2_errors'].append(metrics.get('l2_error', 0))
            methods[method_name]['ssim'].append(metrics.get('ssim', 0))
            methods[method_name]['times'].append(metrics.get('time', 0))
    
    return methods

def paired_ttest(scores1, scores2, name1, name2):
    """Perform paired t-test between two methods."""
    arr1 = np.array(scores1)
    arr2 = np.array(scores2)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(arr1, arr2)
    
    # Cohen's d effect size
    diff = arr1 - arr2
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    # Effect size interpretation
    if abs(d) < 0.2:
        effect = "negligible"
    elif abs(d) < 0.5:
        effect = "small"
    elif abs(d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    # Significance stars
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    return {
        'method1': name1,
        'method2': name2,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'significance_level': sig,
        'cohens_d': float(d),
        'effect_size': effect,
        'mean_diff': float(np.mean(diff)),
        'mean_diff_pct': float(np.mean(diff) / np.mean(arr2) * 100)
    }

def compute_ci(scores, confidence=0.95):
    """Compute confidence interval."""
    mean = np.mean(scores)
    sem = stats.sem(scores)
    ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=sem)
    return float(mean), (float(ci[0]), float(ci[1]))

def shapiro_test(scores, method_name):
    """Test normality with Shapiro-Wilk."""
    stat, p_value = stats.shapiro(scores)
    return {
        'method': method_name,
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_normal': p_value > 0.05
    }

def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading results...")
    results = load_results()
    methods = extract_method_scores(results)
    
    print(f"Found {len(results)} samples")
    print(f"Found {len(methods)} methods: {', '.join(methods.keys())}")
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS (L2 Error)")
    print("-" * 80)
    
    summary_stats = {}
    for method, data in methods.items():
        l2_errors = data['l2_errors']
        mean, ci = compute_ci(l2_errors)
        std = np.std(l2_errors)
        
        summary_stats[method] = {
            'mean': mean,
            'std': std,
            'median': float(np.median(l2_errors)),
            'min': float(np.min(l2_errors)),
            'max': float(np.max(l2_errors)),
            'ci_95': ci,
            'n': len(l2_errors)
        }
        
        print(f"{method:25s}: {mean:.4f} ± {std:.4f}  [95% CI: {ci[0]:.4f}-{ci[1]:.4f}]  (n={len(l2_errors)})")
    
    print()
    
    # Normality tests
    print("NORMALITY TESTS (Shapiro-Wilk)")
    print("-" * 80)
    
    normality_results = {}
    for method, data in methods.items():
        result = shapiro_test(data['l2_errors'], method)
        normality_results[method] = result
        
        status = "✓ Normal" if result['is_normal'] else "✗ Not normal"
        print(f"{method:25s}: W={result['statistic']:.4f}, p={result['p_value']:.4f}  ({status})")
    
    print()
    
    # Pairwise comparisons
    print("PAIRWISE COMPARISONS (Paired t-tests)")
    print("-" * 80)
    
    comparisons = []
    method_names = list(methods.keys())
    
    # Key comparisons
    key_pairs = []
    
    # Find "ours" methods
    ours_methods = [m for m in method_names if 'ours' in m.lower() or 'oracle' in m.lower()]
    baseline_methods = [m for m in method_names if m not in ours_methods]
    
    # Compare each "ours" with each baseline
    for our_method in ours_methods:
        for baseline in baseline_methods:
            key_pairs.append((our_method, baseline))
    
    # Compare different "ours" configurations
    if len(ours_methods) > 1:
        for i, m1 in enumerate(ours_methods):
            for m2 in ours_methods[i+1:]:
                key_pairs.append((m1, m2))
    
    for method1, method2 in key_pairs:
        if method1 not in methods or method2 not in methods:
            continue
        
        scores1 = methods[method1]['l2_errors']
        scores2 = methods[method2]['l2_errors']
        
        # Ensure same length
        min_len = min(len(scores1), len(scores2))
        scores1 = scores1[:min_len]
        scores2 = scores2[:min_len]
        
        result = paired_ttest(scores1, scores2, method1, method2)
        comparisons.append(result)
        
        print(f"\n{method1} vs {method2}:")
        print(f"  t-statistic:  {result['t_statistic']:8.3f}")
        print(f"  p-value:      {result['p_value']:8.6f}  ({result['significance_level']})")
        print(f"  Cohen's d:    {result['cohens_d']:8.3f}  ({result['effect_size']} effect)")
        print(f"  Mean diff:    {result['mean_diff']:8.4f}  ({result['mean_diff_pct']:+.1f}%)")
        
        if not result['significant']:
            print(f"  → No significant difference (good for efficiency claims!)")
        elif result['mean_diff'] < 0:
            print(f"  → {method1} is significantly better!")
        else:
            print(f"  → {method2} is significantly better!")
    
    print()
    print("=" * 80)
    
    # Save results
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    output = {
        'summary_statistics': summary_stats,
        'normality_tests': normality_results,
        'pairwise_comparisons': comparisons
    }
    
    output_file = output_dir / "statistical_tests.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print()
    print("FOR PAPER:")
    print("  • Add significance markers to tables: *** (p<0.001), ** (p<0.01), * (p<0.05)")
    print("  • Report Cohen's d for main comparisons (e.g., 'd=1.2, large effect')")
    print("  • State non-significant differences explicitly (e.g., 'p=0.12, not significant')")
    print()

if __name__ == "__main__":
    main()
