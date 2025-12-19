"""Visualize optimization results."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def render_svg_simple(svg_path: Path, size: int = 256) -> np.ndarray:
    """Simple SVG rendering for visualization."""
    from PIL import ImageDraw
    import xml.etree.ElementTree as ET
    import re
    
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path.get('d', '')
        
        coords = []
        tokens = re.findall(r'[ML]\s*[\d.]+,[\d.]+', d)
        
        for token in tokens:
            match = re.search(r'([\d.]+),([\d.]+)', token)
            if match:
                x = float(match.group(1)) * size
                y = float(match.group(2)) * size
                coords.append((x, y))
        
        if len(coords) >= 2:
            draw.line(coords, fill=0, width=2)
    
    return np.array(img)


def compare_optimization(
    raster_path: Path,
    init_svg_path: Path,
    optimized_svg_path: Path,
    output_path: Path
):
    """Create comparison visualization."""
    # Load images
    target = np.array(Image.open(raster_path).convert('L'))
    init_render = render_svg_simple(init_svg_path)
    opt_render = render_svg_simple(optimized_svg_path)
    
    # Compute errors
    init_error = np.abs(target.astype(float) - init_render.astype(float))
    opt_error = np.abs(target.astype(float) - opt_render.astype(float))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(target, cmap='gray')
    axes[0, 0].set_title('Target Raster')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(init_render, cmap='gray')
    axes[0, 1].set_title('Initialized SVG')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(opt_render, cmap='gray')
    axes[0, 2].set_title('Optimized SVG')
    axes[0, 2].axis('off')
    
    # Row 2: Errors
    axes[1, 0].text(0.5, 0.5, 'Target\n(Ground Truth)', 
                    ha='center', va='center', fontsize=14)
    axes[1, 0].axis('off')
    
    im1 = axes[1, 1].imshow(init_error, cmap='hot', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Init Error (MSE: {np.mean(init_error**2):.1f})')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    
    im2 = axes[1, 2].imshow(opt_error, cmap='hot', vmin=0, vmax=255)
    axes[1, 2].set_title(f'Opt Error (MSE: {np.mean(opt_error**2):.1f})')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison saved to: {output_path}")
    print(f"\nError Reduction:")
    print(f"  Before: MSE = {np.mean(init_error**2):.1f}")
    print(f"  After:  MSE = {np.mean(opt_error**2):.1f}")
    print(f"  Improvement: {100*(1 - np.mean(opt_error**2)/np.mean(init_error**2)):.1f}%")


if __name__ == '__main__':
    raster_path = Path('data_processed/raster/icon_772062_base.png')
    init_svg_path = Path('outputs/test_initialization.svg')
    opt_svg_path = Path('outputs/test_optimized.svg')
    output_path = Path('outputs/optimization_comparison.png')
    
    compare_optimization(raster_path, init_svg_path, opt_svg_path, output_path)
