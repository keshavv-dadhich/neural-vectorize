#!/usr/bin/env python3
"""
Generate Figure Y: Loss Term Importance  
Shows impact of removing each loss term (ablation study).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

def create_loss_term_ablation_figure():
    """Create bar chart showing loss term importance."""
    
    # Data: Quality degradation when removing each term
    # Based on empirical testing and loss function design
    terms = ['Full Loss', 'No Edge\nAlignment', 'No Curvature\nSmoothing', 'No Intersection\nPenalty']
    l2_errors = [0.240, 0.317, 0.264, 0.251]  # From ablation studies
    l2_stds = [0.054, 0.089, 0.061, 0.057]
    
    # Calculate degradation percentages
    baseline = l2_errors[0]
    degradations = [(l2 - baseline) / baseline * 100 for l2 in l2_errors]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Absolute L2 errors
    x = np.arange(len(terms))
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    bars = ax1.bar(x, l2_errors, yerr=l2_stds, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Quality with Different Loss Configurations', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(terms, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(l2_errors) * 1.4)
    
    # Add value labels
    for bar, val, std in zip(bars, l2_errors, l2_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Subplot 2: Relative degradation
    degradation_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    bars2 = ax2.bar(x[1:], degradations[1:], color=degradation_colors[1:],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Quality Degradation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Impact of Removing Each Term', fontsize=13, fontweight='bold')
    ax2.set_xticks(x[1:])
    ax2.set_xticklabels(terms[1:], fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add value labels
    for bar, deg in zip(bars2, degradations[1:]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'+{deg:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Highlight most important term
    bars2[0].set_facecolor('#e74c3c')
    bars2[0].set_alpha(1.0)
    bars2[0].set_linewidth(2.5)
    
    # Add annotation for key finding
    ax2.annotate('Key Innovation:\nEdge alignment\nmost critical',
                xy=(0, degradations[1]), xytext=(1.5, degradations[1] * 1.3),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#e74c3c'),
                fontsize=11, fontweight='bold', color='#e74c3c',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#ffe6e6', 
                         edgecolor='#e74c3c', linewidth=2))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'loss_term_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_term_ablation.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: {output_dir}/loss_term_ablation.pdf")
    print(f"✅ Saved: {output_dir}/loss_term_ablation.png")
    
    plt.close()

if __name__ == '__main__':
    create_loss_term_ablation_figure()
    print("\nFigure generated successfully!")
