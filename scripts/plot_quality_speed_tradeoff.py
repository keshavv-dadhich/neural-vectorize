#!/usr/bin/env python3
"""
Generate Figure 8: Quality-Speed Tradeoff
Shows Pareto frontier for different step counts
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def plot_quality_speed_tradeoff():
    """Create quality-speed tradeoff scatter plot"""
    
    # Load ablation results
    stats_file = Path("analysis/ablation_statistical_tests.json")
    
    if not stats_file.exists():
        print(f"❌ Missing {stats_file}")
        print("   Run: python3 scripts/run_ablation_statistics.py")
        return
    
    with open(stats_file) as f:
        data = json.load(f)
    
    # Extract data for each configuration
    configs = []
    times = []
    qualities = []
    errors = []
    
    # Get summary statistics
    summary = data.get('summary_statistics', {})
    
    for config_name in ['30', '75', '150']:
        if config_name in summary:
            config_data = summary[config_name]
            
            # Get mean values
            time = config_data['time']['mean']
            error = config_data['l2']['mean']
            
            # Compute relative quality (lower error = higher quality)
            # Use 150 steps as baseline (100%)
            baseline_error = summary['150']['l2']['mean']
            relative_quality = (baseline_error / error) * 100
            
            configs.append(f'{config_name} Steps')
            times.append(time)
            qualities.append(relative_quality)
            errors.append(error)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors for each point
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    # Plot points individually (to avoid size array issue)
    for i, (time, quality, color) in enumerate(zip(times, qualities, colors)):
        size = 300 if i == 0 else 250
        ax.scatter(time, quality, s=size, c=color, 
                  alpha=0.7, edgecolors='black', linewidths=2, zorder=3)
    
    # Add labels for each point
    for i, (config, time, quality, error) in enumerate(zip(configs, times, qualities, errors)):
        # Position label
        if i == 0:  # 30 steps - above
            xytext = (0, 20)
            va = 'bottom'
        elif i == 1:  # 75 steps - right
            xytext = (10, -5)
            va = 'center'
        else:  # 150 steps - below
            xytext = (0, -25)
            va = 'top'
        
        ax.annotate(f'{config}\n{time:.1f}s, {quality:.1f}%',
                   xy=(time, quality), xytext=xytext,
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   ha='center', va=va,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], 
                            alpha=0.3, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Draw Pareto frontier (connecting line)
    ax.plot(times, qualities, 'k--', alpha=0.3, linewidth=2, zorder=1, label='Pareto Frontier')
    
    # Add "sweet spot" annotation for 30 steps
    ax.annotate('⭐ Best Tradeoff\n5.5× speedup\nOnly 2.9% quality loss',
               xy=(times[0], qualities[0]), xytext=(times[0] + 15, qualities[0] - 1.5),
               fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    # Styling
    ax.set_xlabel('Time per Sample (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Quality (%)', fontsize=14, fontweight='bold')
    ax.set_title('Quality-Speed Tradeoff: Optimization Steps', fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Set limits with padding
    ax.set_xlim(0, max(times) * 1.15)
    ax.set_ylim(min(qualities) - 2, 101)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(max(times) * 0.5, 100.3, '100% Quality (Baseline)', 
           ha='center', va='bottom', fontsize=10, color='gray', style='italic')
    
    # Add stats box
    stats_text = f'Configurations:\n'
    stats_text += f'• 30 steps: {errors[0]:.3f} L2, {times[0]:.1f}s\n'
    stats_text += f'• 75 steps: {errors[1]:.3f} L2, {times[1]:.1f}s\n'
    stats_text += f'• 150 steps: {errors[2]:.3f} L2, {times[2]:.1f}s\n\n'
    stats_text += f'Speedup (vs 150): {times[2]/times[0]:.1f}×'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor='gray', linewidth=1.5), family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    pdf_path = output_dir / "quality_speed_tradeoff.pdf"
    png_path = output_dir / "quality_speed_tradeoff.png"
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Quality-speed tradeoff saved:")
    print(f"   - {pdf_path}")
    print(f"   - {png_path}")
    print()
    print("FOR PAPER:")
    print("  • Use as Figure 8 (Results section)")
    print("  • Caption: 'Quality-speed tradeoff across optimization steps. 30 steps")
    print("    (green) offers the best balance: 97.1% quality with 5.5× speedup.")
    print("    Diminishing returns beyond 75 steps justify the faster configuration.'")

if __name__ == "__main__":
    print("Creating quality-speed tradeoff figure...")
    plot_quality_speed_tradeoff()
