"""
SVG utility functions.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import re


def svg_to_raster(svg_path: Path, size: int = 256) -> np.ndarray:
    """
    Convert SVG to raster image using simple line rendering.
    
    Args:
        svg_path: Path to SVG file
        size: Output image size
        
    Returns:
        Grayscale image (H, W) in [0, 1]
    """
    # Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Create blank image
    img = Image.new('L', (size, size), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Extract paths and draw
    for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path_elem.get('d', '')
        
        # Extract coordinates
        coords = []
        tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
        
        for token in tokens:
            match = re.search(r'([\d.]+),([\d.]+)', token)
            if match:
                x = float(match.group(1)) * size
                y = float(match.group(2)) * size
                coords.append((x, y))
        
        # Draw lines
        if len(coords) >= 2:
            for i in range(len(coords) - 1):
                draw.line([coords[i], coords[i+1]], fill=0, width=2)
    
    # Convert to numpy array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    
    return arr
