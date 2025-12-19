"""Visualize initialization results."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import re


def render_svg_simple(svg_path: Path, size: int = 256) -> np.ndarray:
    """Simple SVG rendering (lines only) for visualization."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Create blank canvas
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Extract all path elements
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path.get('d', '')
        
        # Parse path commands (simplified - just M and L)
        coords = []
        tokens = re.findall(r'[ML]\s*[\d.]+,[\d.]+', d)
        
        for token in tokens:
            match = re.search(r'([\d.]+),([\d.]+)', token)
            if match:
                x = float(match.group(1)) * size
                y = float(match.group(2)) * size
                coords.append((x, y))
        
        # Draw polyline
        if len(coords) >= 2:
            draw.line(coords, fill=0, width=2)
    
    return np.array(img)


def visualize_initialization(
    raster_path: Path,
    svg_path: Path,
    output_path: Path
):
    """
    Create side-by-side visualization of input raster and initialized SVG.
    
    Args:
        raster_path: Input PNG path
        svg_path: Initialized SVG path
        output_path: Where to save comparison
    """
    # Load raster
    raster = Image.open(raster_path).convert('L')
    
    # Render SVG
    svg_rendered = render_svg_simple(svg_path, size=256)
    
    # Create comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(raster, cmap='gray')
    axes[0].set_title('Input Raster')
    axes[0].axis('off')
    
    # Show edges for debugging
    image = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Detected Edges')
    axes[1].axis('off')
    
    axes[2].imshow(svg_rendered, cmap='gray')
    axes[2].set_title('Initialized SVG')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")


if __name__ == '__main__':
    # Test on the same sample
    raster_path = Path('data_processed/raster/icon_772062_base.png')
    svg_path = Path('outputs/test_initialization.svg')
    output_path = Path('outputs/init_comparison.png')
    
    visualize_initialization(raster_path, svg_path, output_path)
