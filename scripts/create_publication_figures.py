"""
Generate publication-quality plots for ablation study.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import json

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


def create_training_curves():
    """Plot neural training convergence."""
    history_path = Path('models/neural_init/history.json')
    
    if not history_path.exists():
        print(f"⚠️  Training history not found")
        return
    
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['train_point_loss']) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Point loss
    ax1.plot(epochs, history['train_point_loss'], label='Train', linewidth=2, color='#2E86AB', alpha=0.8)
    ax1.plot(epochs, history['val_point_loss'], label='Val', linewidth=2, color='#A23B72', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Point Loss (MSE)')
    ax1.set_title('(a) Point Reconstruction Loss')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.2, linestyle=':')
    ax1.set_yscale('log')
    
    # Mask loss  
    ax2.plot(epochs, history['train_mask_loss'], label='Train', linewidth=2, color='#2E86AB', alpha=0.8)
    ax2.plot(epochs, history['val_mask_loss'], label='Val', linewidth=2, color='#A23B72', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mask Loss (BCE)')
    ax2.set_title('(b) Validity Mask Loss')
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    output_path = 'visualizations/fig_training_curves.pdf'
    Path('visualizations').mkdir(exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_ablation_tradeoff():
    """Quality vs speed tradeoff plot."""
    results_path = Path('baselines/ablation_steps/ablation_steps_results.json')
    
    if not results_path.exists():
        print(f"⚠️  Ablation results not found")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    steps_list = [30, 75, 150]
    l2_means, l2_stds, ssim_means, times = [], [], [], []
    
    for steps in steps_list:
        data = results[str(steps)]
        l2_means.append(np.mean([r['l2'] for r in data]))
        l2_stds.append(np.std([r['l2'] for r in data]))
        ssim_means.append(np.mean([r['ssim'] for r in data]))
        times.append(np.mean([r['time'] for r in data]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # L2 vs steps
    ax1.errorbar(steps_list, l2_means, yerr=l2_stds, marker='o', markersize=8,
                 linewidth=2, capsize=4, color='#E63946', capthick=2)
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('(a) Reconstruction Quality')
    ax1.grid(True, alpha=0.2, linestyle=':')
    ax1.set_xticks(steps_list)
    
    # Time vs steps with speedup annotations
    baseline_time = times[2]
    ax2.plot(steps_list, times, marker='s', markersize=8, linewidth=2,
             color='#457B9D', linestyle='--')
    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('Time per Sample (s)')
    ax2.set_title('(b) Computational Cost')
    ax2.grid(True, alpha=0.2, linestyle=':')
    ax2.set_xticks(steps_list)
    
    # Add speedup annotations
    for steps, time in zip(steps_list[:-1], times[:-1]):
        speedup = baseline_time / time
        ax2.annotate(f'{speedup:.1f}×',
                    xy=(steps, time), xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', fontsize=10, color='#457B9D',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#457B9D', linewidth=1.5))
    
    plt.tight_layout()
    output_path = 'visualizations/fig_ablation_tradeoff.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_results_summary():
    """Create comprehensive results summary figure."""
    results_path = Path('baselines/ablation_steps/ablation_steps_results.json')
    
    if not results_path.exists():
        print(f"⚠️  Ablation results not found")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    steps_list = [30, 75, 150]
    colors = ['#F77F00', '#06FFA5', '#2E86AB']
    
    # Collect data
    all_l2, all_ssim, all_time, all_segments = [], [], [], []
    for steps in steps_list:
        data = results[str(steps)]
        all_l2.append([r['l2'] for r in data])
        all_ssim.append([r['ssim'] for r in data])
        all_time.append([r['time'] for r in data])
        all_segments.append([r['segments'] for r in data])
    
    # L2 distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bp1 = ax1.boxplot(all_l2, labels=steps_list, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('(a) Quality Distribution')
    ax1.grid(True, alpha=0.2, axis='y', linestyle=':')
    
    # SSIM distribution
    ax2 = fig.add_subplot(gs[0, 1])
    bp2 = ax2.boxplot(all_ssim, labels=steps_list, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('SSIM')
    ax2.set_title('(b) Structural Similarity')
    ax2.grid(True, alpha=0.2, axis='y', linestyle=':')
    
    # Time distribution
    ax3 = fig.add_subplot(gs[0, 2])
    bp3 = ax3.boxplot(all_time, labels=steps_list, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('(c) Processing Time')
    ax3.grid(True, alpha=0.2, axis='y', linestyle=':')
    
    # Quality vs Time scatter
    ax4 = fig.add_subplot(gs[1, :2])
    for i, steps in enumerate(steps_list):
        data = results[str(steps)]
        times = [r['time'] for r in data]
        l2s = [r['l2'] for r in data]
        ax4.scatter(times, l2s, s=80, alpha=0.6, color=colors[i],
                   label=f'{steps} steps', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Time per Sample (s)')
    ax4.set_ylabel('L2 Error')
    ax4.set_title('(d) Quality-Speed Pareto Frontier')
    ax4.legend(frameon=False, loc='upper right')
    ax4.grid(True, alpha=0.2, linestyle=':')
    
    # Summary table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    table_data = []
    for steps in steps_list:
        data = results[str(steps)]
        l2_mean = np.mean([r['l2'] for r in data])
        time_mean = np.mean([r['time'] for r in data])
        speedup = times[-1] / time_mean if steps != 150 else 1.0
        table_data.append([f'{steps}', f'{l2_mean:.4f}', f'{time_mean:.1f}s', f'{speedup:.1f}×'])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Steps', 'L2', 'Time', 'Speedup'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.3, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    ax5.set_title('(e) Summary Statistics', pad=20)
    
    output_path = 'visualizations/fig_results_summary.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    print("\n" + "="*60)
    print("Creating Publication-Quality Figures")
    print("="*60 + "\n")
    
    create_training_curves()
    create_ablation_tradeoff()
    create_results_summary()
    
    print("\n" + "="*60)
    print("✅ All publication figures created!")
    print("📁 Location: visualizations/")
    print("📄 Formats: PDF (vector) + PNG (raster)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
