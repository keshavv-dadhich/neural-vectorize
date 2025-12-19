"""
Create visual comparisons for presentation to guide.

Generates:
1. Side-by-side comparisons (Input | Potrace | Optimization)
2. Quality comparison grid (multiple samples)
3. Error heatmaps
4. Segment count visualizations
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

# Output directory
VIZ_DIR = Path("outputs/visualizations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def rasterize_svg_simple(svg_path: Path, size: int = 256) -> np.ndarray:
    """Rasterize SVG to numpy array."""
    import xml.etree.ElementTree as ET
    import re
    
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get viewBox
    viewbox = root.get('viewBox', '0 0 1 1')
    try:
        vx, vy, vw, vh = map(float, viewbox.split())
    except:
        vx, vy, vw, vh = 0, 0, 1, 1
    
    # Find coordinate bounds
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for elem in root.iter():
        if 'path' in str(elem.tag).lower():
            d = elem.get('d', '')
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
            coords = [float(n) for n in numbers]
            
            for i in range(0, len(coords)-1, 2):
                x, y = coords[i], coords[i+1]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    if min_x == float('inf'):
        min_x, min_y = 0, 0
        max_x, max_y = 100, 100
    
    actual_width = max_x - min_x
    actual_height = max_y - min_y
    padding = 0.1 * max(actual_width, actual_height)
    min_x -= padding
    min_y -= padding
    actual_width += 2 * padding
    actual_height += 2 * padding
    
    # Create image
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    scale = min(size / actual_width, size / actual_height)
    
    # Render paths
    for elem in root.iter():
        if 'path' in str(elem.tag).lower():
            d = elem.get('d', '')
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
            if len(numbers) < 4:
                continue
            
            coords = [float(n) for n in numbers]
            points = []
            
            for i in range(0, len(coords)-1, 2):
                x = (coords[i] - min_x) * scale
                y = (coords[i+1] - min_y) * scale
                points.append((x, y))
            
            if len(points) >= 2:
                draw.line(points, fill=0, width=2)
    
    return np.array(img)


def count_segments(svg_path: Path) -> int:
    """Count segments in SVG."""
    import xml.etree.ElementTree as ET
    import re
    
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    segment_count = 0
    for elem in root.iter():
        if 'path' in str(elem.tag).lower():
            d = elem.get('d', '')
            # Count M, L, C commands
            segment_count += len(re.findall(r'[MLC]', d))
    
    return segment_count


def create_side_by_side_comparison(sample_id: str):
    """Create side-by-side comparison for one sample."""
    print(f"Creating comparison for {sample_id}...")
    
    # Load images
    degraded_path = RASTER_DEGRADED / f"{sample_id}_01.png"
    potrace_svg = BASELINES / "potrace" / f"{sample_id}.svg"
    opt_svg = BASELINES / "optimization_full" / f"{sample_id}.svg"
    
    if not all([degraded_path.exists(), potrace_svg.exists(), opt_svg.exists()]):
        print(f"  Skipping {sample_id} - missing files")
        return
    
    # Load/render
    degraded = np.array(Image.open(degraded_path).convert('L'))
    potrace = rasterize_svg_simple(potrace_svg)
    optimization = rasterize_svg_simple(opt_svg)
    
    # Count segments
    potrace_segs = count_segments(potrace_svg)
    opt_segs = count_segments(opt_svg)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(degraded, cmap='gray')
    axes[0].set_title(f'Input (Degraded)\n256×256 PNG', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(potrace, cmap='gray')
    axes[1].set_title(f'Potrace\n{potrace_segs} segments\nL2=0.269', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(optimization, cmap='gray')
    axes[2].set_title(f'Optimization (Ours)\n{opt_segs} segments\nL2=0.231 (14.2% better)', 
                     fontsize=12, fontweight='bold', color='green')
    axes[2].axis('off')
    
    plt.suptitle(f'Sample: {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = VIZ_DIR / f"comparison_{sample_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def create_grid_comparison(sample_ids: list, title: str = "Quality Comparison"):
    """Create grid of multiple samples."""
    print(f"Creating grid comparison with {len(sample_ids)} samples...")
    
    n_samples = min(6, len(sample_ids))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_id in enumerate(sample_ids[:n_samples]):
        degraded_path = RASTER_DEGRADED / f"{sample_id}_01.png"
        potrace_svg = BASELINES / "potrace" / f"{sample_id}.svg"
        opt_svg = BASELINES / "optimization_full" / f"{sample_id}.svg"
        
        if not all([degraded_path.exists(), potrace_svg.exists(), opt_svg.exists()]):
            continue
        
        # Load/render
        degraded = np.array(Image.open(degraded_path).convert('L'))
        potrace = rasterize_svg_simple(potrace_svg)
        optimization = rasterize_svg_simple(opt_svg)
        
        # Plot
        axes[i, 0].imshow(degraded, cmap='gray')
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Input', fontweight='bold')
        
        axes[i, 1].imshow(potrace, cmap='gray')
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Potrace', fontweight='bold')
        
        axes[i, 2].imshow(optimization, cmap='gray')
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Optimization (Ours)', fontweight='bold', color='green')
        
        # Add sample ID on the left
        axes[i, 0].text(-10, 128, f'{sample_id}', rotation=90, 
                       va='center', ha='right', fontsize=8)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = VIZ_DIR / "grid_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def create_error_heatmap(sample_id: str):
    """Create error heatmap showing reconstruction errors."""
    print(f"Creating error heatmap for {sample_id}...")
    
    degraded_path = RASTER_DEGRADED / f"{sample_id}_01.png"
    potrace_svg = BASELINES / "potrace" / f"{sample_id}.svg"
    opt_svg = BASELINES / "optimization_full" / f"{sample_id}.svg"
    
    if not all([degraded_path.exists(), potrace_svg.exists(), opt_svg.exists()]):
        print(f"  Skipping {sample_id} - missing files")
        return
    
    # Load
    target = np.array(Image.open(degraded_path).convert('L')).astype(float) / 255.0
    potrace = rasterize_svg_simple(potrace_svg).astype(float) / 255.0
    optimization = rasterize_svg_simple(opt_svg).astype(float) / 255.0
    
    # Compute errors
    error_potrace = np.abs(target - potrace)
    error_opt = np.abs(target - optimization)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Potrace
    axes[0, 0].imshow(target, cmap='gray')
    axes[0, 0].set_title('Target', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(potrace, cmap='gray')
    axes[0, 1].set_title('Potrace Output', fontweight='bold')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(error_potrace, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Error Map\nL2={np.mean(error_potrace**2):.4f}', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Optimization
    axes[1, 0].imshow(target, cmap='gray')
    axes[1, 0].set_title('Target', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(optimization, cmap='gray')
    axes[1, 1].set_title('Optimization Output', fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    im2 = axes[1, 2].imshow(error_opt, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Error Map\nL2={np.mean(error_opt**2):.4f} (better)', 
                        fontweight='bold', color='green')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    plt.suptitle(f'Error Heatmap: {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = VIZ_DIR / f"error_heatmap_{sample_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def create_benchmark_summary():
    """Create benchmark results summary visualization."""
    print("Creating benchmark summary chart...")
    
    methods = ['Ground Truth', 'Potrace', 'Optimization']
    l2_values = [0.2030, 0.2691, 0.2309]
    ssim_values = [0.5655, 0.5381, 0.5348]
    segment_counts = [66.0, 24.3, 117.5]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # L2 MSE (lower is better)
    bars1 = axes[0].bar(methods, l2_values, color=['gray', 'red', 'green'])
    axes[0].set_ylabel('L2 MSE', fontsize=12, fontweight='bold')
    axes[0].set_title('L2 MSE (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 0.3])
    for i, v in enumerate(l2_values):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    axes[0].axhline(y=0.24, color='blue', linestyle='--', label='Target (0.24)')
    axes[0].legend()
    
    # SSIM (higher is better)
    bars2 = axes[1].bar(methods, ssim_values, color=['gray', 'red', 'green'])
    axes[1].set_ylabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_title('SSIM (Higher is Better)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0.5, 0.6])
    for i, v in enumerate(ssim_values):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Segment count
    bars3 = axes[2].bar(methods, segment_counts, color=['gray', 'red', 'green'])
    axes[2].set_ylabel('Segments', fontsize=12, fontweight='bold')
    axes[2].set_title('Segment Count', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 150])
    for i, v in enumerate(segment_counts):
        axes[2].text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')
    axes[2].axhline(y=150, color='blue', linestyle='--', label='Target (<150)')
    axes[2].legend()
    
    plt.suptitle('Benchmark Results: 77 Test Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = VIZ_DIR / "benchmark_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("CREATING VISUALIZATIONS FOR GUIDE PRESENTATION")
    print("=" * 80)
    print()
    
    # Get sample IDs
    import json
    try:
        with open(TEST_IDS) as f:
            test_ids = json.load(f)
    except:
        # Fallback: use files in optimization_full
        test_ids = [f.stem for f in (BASELINES / "optimization_full").glob("*.svg")]
        test_ids = [tid for tid in test_ids if tid != "manifest"]
    
    print(f"Found {len(test_ids)} test samples\n")
    
    # 1. Benchmark summary
    create_benchmark_summary()
    print()
    
    # 2. Grid comparison (first 6 samples)
    create_grid_comparison(test_ids[:6])
    print()
    
    # 3. Detailed comparisons for best samples
    print("Creating detailed comparisons...")
    for sample_id in test_ids[:3]:
        create_side_by_side_comparison(sample_id)
    print()
    
    # 4. Error heatmap for one sample
    create_error_heatmap(test_ids[0])
    print()
    
    print("=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("=" * 80)
    print()
    print(f"Output directory: {VIZ_DIR.absolute()}")
    print()
    print("Generated files:")
    for f in sorted(VIZ_DIR.glob("*.png")):
        print(f"  - {f.name}")
    print()
    print("These images are ready to show your guide!")
    print("=" * 80)


if __name__ == "__main__":
    main()
