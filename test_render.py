"""
Quick test to debug SVG rasterization.
"""

from pathlib import Path
from PIL import Image
import sys

svg_path = Path("data_processed/svg_clean/icon_000003.svg")
output_path = Path("test_output.png")

print(f"Testing rasterization of: {svg_path}")
print(f"Output: {output_path}\n")

# Test 1: svglib + reportlab
print("Test 1: svglib + reportlab")
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    
    drawing = svg2rlg(str(svg_path))
    if drawing:
        print(f"  ✓ Drawing loaded: {drawing.width} x {drawing.height}")
        
        drawing.width = 256
        drawing.height = 256
        drawing.scale(256/drawing.width if drawing.width else 1, 
                     256/drawing.height if drawing.height else 1)
        
        renderPM.drawToFile(drawing, str(output_path), fmt='PNG', bg=0xffffff, dpi=96)
        
        img = Image.open(output_path)
        img = img.convert('L')
        img.save(output_path)
        
        print(f"  ✓ Saved to {output_path}")
        print(f"  ✓ Image size: {img.size}")
        print(f"  ✓ Image mode: {img.mode}")
        
        # Check if image is not blank
        pixels = list(img.getdata())
        unique_vals = len(set(pixels))
        min_val = min(pixels)
        max_val = max(pixels)
        print(f"  ✓ Pixel values: min={min_val}, max={max_val}, unique={unique_vals}")
        
        if unique_vals > 1:
            print("  ✅ SUCCESS - Image has content!")
        else:
            print("  ❌ FAIL - Image is blank")
    else:
        print("  ❌ Could not load drawing")
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\nDone! Check test_output.png")
