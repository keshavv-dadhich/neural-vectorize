#!/usr/bin/env python3
"""
Generate edge-based initializations for all test samples.
Uses Canny edge detection + simple contour tracing.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2
import argparse

def generate_edge_init(raster_path, output_path):
    """Generate simple edge-based SVG initialization."""
    
    # Load image
    img = np.array(Image.open(raster_path).convert('L'))
    
    # Canny edge detection
    edges = cv2.Canny(img, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create simple SVG with paths
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{img.shape[1]}" height="{img.shape[0]}" xmlns="http://www.w3.org/2000/svg">
'''
    
    for contour in contours:
        if len(contour) < 3:
            continue
            
        # Convert contour to path
        points = contour.reshape(-1, 2)
        path_d = f"M {points[0][0]} {points[0][1]} "
        
        for point in points[1:]:
            path_d += f"L {point[0]} {point[1]} "
        
        path_d += "Z"
        
        svg_content += f'  <path d="{path_d}" fill="black" stroke="none"/>\n'
    
    svg_content += '</svg>'
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(svg_content)

def main():
    parser = argparse.ArgumentParser(description='Generate edge-based initializations')
    parser.add_argument('--samples', type=int, default=15, help='Number of samples')
    args = parser.parse_args()
    
    print(f"\nGenerating edge-based initializations for {args.samples} samples...")
    print("=" * 70)
    
    raster_dir = Path("data_processed/raster_degraded")
    output_dir = Path("baselines/edge_init")
    
    samples = sorted([f for f in raster_dir.glob("*.png")])[:args.samples]
    
    for i, raster_path in enumerate(samples, 1):
        output_path = output_dir / raster_path.name.replace('.png', '.svg')
        
        print(f"[{i}/{len(samples)}] {raster_path.name} → {output_path.name}")
        
        try:
            generate_edge_init(raster_path, output_path)
            print(f"  ✓ Generated")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ Generated {len(samples)} edge initializations")
    print(f"   Output: {output_dir}/")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
