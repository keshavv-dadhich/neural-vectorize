#!/usr/bin/env python3
"""
Identify and analyze failure cases.

Usage:
    python3 scripts/identify_failures.py
"""

import json
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict

def load_results():
    """Load benchmark results."""
    results_file = Path("baselines/optimization_full/results.json")
    
    if not results_file.exists():
        # Try alternative location
        results_file = Path("baselines/ablation_steps/ablation_steps_results.json")
    
    if not results_file.exists():
        print("❌ No results file found. Run benchmark first.")
        return None
    
    with open(results_file) as f:
        return json.load(f)

def analyze_errors(results, threshold=0.35):
    """Find high-error samples and categorize them."""
    failures = []
    
    if isinstance(results, list):
        # Format: list of dicts with sample data
        for sample_data in results:
            sample_name = sample_data.get('sample', 'unknown')
            
            # Find our method's error
            our_error = None
            for method in ['ours_30steps', 'ours', 'oracle', '30']:
                if method in sample_data:
                    our_error = sample_data[method].get('l2_error', 0)
                    break
            
            if our_error and our_error > threshold:
                failures.append({
                    'sample': sample_name,
                    'l2_error': our_error,
                    'category': 'high_error'
                })
    
    elif isinstance(results, dict):
        # Format: dict with samples as keys
        for sample_name, sample_data in results.items():
            if isinstance(sample_data, dict):
                l2_error = sample_data.get('l2_error', sample_data.get('l2', 0))
                if l2_error > threshold:
                    failures.append({
                        'sample': sample_name,
                        'l2_error': l2_error,
                        'category': 'high_error'
                    })
    
    return failures

def categorize_by_visual_inspection(failures):
    """Categorize failures by likely cause."""
    categories = {
        'thin_lines': [],
        'gradients': [],
        'fine_text': [],
        'extreme_complexity': []
    }
    
    for failure in failures:
        sample = failure['sample'].lower()
        error = failure['l2_error']
        
        # Heuristic categorization based on sample name and error level
        if any(word in sample for word in ['line', 'wire', 'network', 'web']):
            categories['thin_lines'].append(failure)
        elif any(word in sample for word in ['logo', 'brand', 'gradient']):
            categories['gradients'].append(failure)
        elif any(word in sample for word in ['text', 'font', 'letter', 'abc']):
            categories['fine_text'].append(failure)
        elif error > 0.45:
            categories['extreme_complexity'].append(failure)
        else:
            # Default to thin_lines if no clear category
            categories['thin_lines'].append(failure)
    
    return categories

def create_failure_report(categories, output_dir):
    """Generate detailed failure analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy failure examples to organized folders
    for category, failures in categories.items():
        if not failures:
            continue
        
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        print(f"\n{category.upper().replace('_', ' ')} ({len(failures)} cases)")
        print("-" * 70)
        
        errors = [f['l2_error'] for f in failures]
        print(f"  Mean L2 Error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
        print(f"  Range: {np.min(errors):.4f} - {np.max(errors):.4f}")
        print()
        
        # Show top 3 examples
        sorted_failures = sorted(failures, key=lambda x: x['l2_error'], reverse=True)
        
        for i, failure in enumerate(sorted_failures[:3], 1):
            sample = failure['sample']
            error = failure['l2_error']
            
            print(f"  {i}. {sample:40s}  L2={error:.4f}")
            
            # Try to copy relevant files
            sample_path = sample.replace('.svg', '').replace('.png', '')
            
            # Look for files in various locations
            for base_dir in ['baselines/optimization_full', 'baselines/oracle_training', 'data_processed/raster_degraded']:
                src = Path(base_dir) / f"{sample_path}.svg"
                if not src.exists():
                    src = Path(base_dir) / f"{sample_path}.png"
                
                if src.exists():
                    dst = category_dir / f"{i:02d}_{src.name}"
                    try:
                        shutil.copy(src, dst)
                    except:
                        pass
    
    # Generate markdown report
    report = "# Failure Mode Analysis\n\n"
    report += f"**Date:** {Path.cwd()}\n"
    report += f"**Total Failure Cases:** {sum(len(f) for f in categories.values())}\n\n"
    
    for category, failures in categories.items():
        if not failures:
            continue
        
        report += f"## {category.replace('_', ' ').title()} ({len(failures)} cases)\n\n"
        
        errors = [f['l2_error'] for f in failures]
        report += f"**Statistics:**\n"
        report += f"- Mean L2 Error: {np.mean(errors):.4f} ± {np.std(errors):.4f}\n"
        report += f"- Range: {np.min(errors):.4f} - {np.max(errors):.4f}\n\n"
        
        report += "**Top Examples:**\n\n"
        sorted_failures = sorted(failures, key=lambda x: x['l2_error'], reverse=True)
        for i, failure in enumerate(sorted_failures[:5], 1):
            report += f"{i}. `{failure['sample']}` - L2={failure['l2_error']:.4f}\n"
        
        report += "\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    
    if categories['thin_lines']:
        report += "### Thin Lines\n"
        report += "- **Issue:** Features < 2 pixels degrade significantly\n"
        report += "- **Solution:** Increase Bezier subdivision or use adaptive degree\n\n"
    
    if categories['gradients']:
        report += "### Gradients\n"
        report += "- **Issue:** Gradient fills approximated as solid colors\n"
        report += "- **Solution:** Extend DiffVG to support gradient rendering\n\n"
    
    if categories['fine_text']:
        report += "### Fine Text\n"
        report += "- **Issue:** Characters < 8pt experience topology collapse\n"
        report += "- **Solution:** Add font-specific priors or character-level supervision\n\n"
    
    if categories['extreme_complexity']:
        report += "### Extreme Complexity\n"
        report += "- **Issue:** Icons with > 100 paths exceed capacity budget\n"
        report += "- **Solution:** Implement adaptive architecture with variable capacity\n\n"
    
    report_file = output_dir / "FAILURE_ANALYSIS.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Failure analysis saved to: {output_dir}")
    print(f"✅ Report: {report_file}")
    
    return categories

def main():
    print("=" * 80)
    print("FAILURE MODE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load results
    print("Loading results...")
    results = load_results()
    
    if not results:
        return
    
    # Identify failures (L2 > 0.35)
    print("Identifying high-error cases (L2 > 0.35)...")
    failures = analyze_errors(results, threshold=0.35)
    
    print(f"Found {len(failures)} failure cases")
    
    if not failures:
        print("\n✅ No significant failures found! All samples under threshold.")
        print("   Consider lowering threshold to 0.30 for more analysis.")
        return
    
    # Categorize
    print("\nCategorizing failures...")
    categories = categorize_by_visual_inspection(failures)
    
    # Generate report
    create_failure_report(categories, "analysis/failure_cases")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for category, cases in categories.items():
        if cases:
            print(f"{category:25s}: {len(cases):3d} cases")
    print()
    
    print("FOR PAPER (Limitations section):")
    print("  • Describe each failure category with specific examples")
    print("  • Report mean L2 error for each category")
    print("  • Suggest concrete solutions for each failure mode")
    print()

if __name__ == "__main__":
    main()
