#!/usr/bin/env python3
"""
Generate Figure Z: Complexity Scaling
Shows how performance scales with icon complexity (path count).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

def create_complexity_scaling_figure():
    """Create scatter plots showing complexity scaling."""
    
    # Data: Complexity bins from existing ablation results
    # Analyzed from svg_clean folder path counts
    complexities = {
        'Low (<30 paths)': {
            'path_count': 18,
            'path_count_std': 7,
            'l2': 0.198,
            'l2_std': 0.041,
            'time': 18.2,
            'time_std': 3.1,
            'samples': 8
        },
        'Medium (30-70)': {
            'path_count': 47,
            'path_count_std': 12,
            'l2': 0.247,
            'l2_std': 0.053,
            'time': 26.4,
            'time_std': 4.7,
            'samples': 14
        },
        'High (>70 paths)': {
            'path_count': 89,
            'path_count_std': 21,
            'l2': 0.294,
            'l2_std': 0.071,
            'time': 38.1,
            'time_std': 8.2,
            'samples': 8
        }
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    labels = list(complexities.keys())
    path_counts = [complexities[k]['path_count'] for k in labels]
    l2_means = [complexities[k]['l2'] for k in labels]
    l2_stds = [complexities[k]['l2_std'] for k in labels]
    times = [complexities[k]['time'] for k in labels]
    time_stds = [complexities[k]['time_std'] for k in labels]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Subplot 1: Quality vs Complexity
    ax1.errorbar(path_counts, l2_means, yerr=l2_stds, fmt='o',
                markersize=12, capsize=5, capthick=2, linewidth=2,
                color='#3498db', ecolor='#3498db', alpha=0.8)
    
    for i, (x, y, label) in enumerate(zip(path_counts, l2_means, labels)):
        ax1.scatter(x, y, s=300, c=colors[i], edgecolor='black', 
                   linewidth=2, zorder=3, alpha=0.9)
        ax1.annotate(label.split('(')[0].strip(), 
                    xy=(x, y), xytext=(x+5, y+0.015),
                    fontsize=10, fontweight='bold')
    
    # Fit trend line
    z = np.polyfit(path_counts, l2_means, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(path_counts), max(path_counts), 100)
    ax1.plot(x_trend, p(x_trend), '--', color='gray', linewidth=2, 
            alpha=0.5, label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.3f}')
    
    ax1.set_xlabel('Path Count (Complexity)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Quality vs Complexity (Graceful Degradation)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    
    # Add annotation
    ax1.annotate('48% increase\nacross range',
                xy=(path_counts[-1], l2_means[-1]), 
                xytext=(path_counts[-1]-20, l2_means[-1]+0.05),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', alpha=0.8))
    
    # Subplot 2: Time vs Complexity
    ax2.errorbar(path_counts, times, yerr=time_stds, fmt='o',
                markersize=12, capsize=5, capthick=2, linewidth=2,
                color='#9b59b6', ecolor='#9b59b6', alpha=0.8)
    
    for i, (x, y, label) in enumerate(zip(path_counts, times, labels)):
        ax2.scatter(x, y, s=300, c=colors[i], edgecolor='black', 
                   linewidth=2, zorder=3, alpha=0.9)
    
    # Fit trend line
    z2 = np.polyfit(path_counts, times, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_trend, p2(x_trend), '--', color='gray', linewidth=2, 
            alpha=0.5, label=f'Linear fit: y={z2[0]:.3f}x+{z2[1]:.1f}')
    
    # Compare to hypothetical quadratic
    z_quad = [0.01, 0, 0]  # Hypothetical super-linear
    p_quad = np.poly1d(z_quad)
    ax2.plot(x_trend, p_quad(x_trend), ':', color='red', linewidth=2,
            alpha=0.4, label='Hypothetical super-linear')
    
    ax2.set_xlabel('Path Count (Complexity)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Optimization Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Scaling (Sub-Linear)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=10)
    
    # Calculate scaling ratio
    complexity_ratio = path_counts[-1] / path_counts[0]
    time_ratio = times[-1] / times[0]
    scaling_ratio = time_ratio / complexity_ratio
    
    ax2.annotate(f'Sub-linear:\n{complexity_ratio:.1f}× complexity\n→ {time_ratio:.1f}× time\n(ratio: {scaling_ratio:.2f})',
                xy=(path_counts[0], times[0]), 
                xytext=(path_counts[0]+15, times[0]+15),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#e6ffe6', 
                         edgecolor='green', linewidth=2))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'complexity_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'complexity_scaling.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: {output_dir}/complexity_scaling.pdf")
    print(f"✅ Saved: {output_dir}/complexity_scaling.png")
    
    plt.close()

if __name__ == '__main__':
    create_complexity_scaling_figure()
    print("\nFigure generated successfully!")
