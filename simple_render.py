"""
Pure Python SVG to PNG rasterizer using Pillow only.
No system dependencies required.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import re
from typing import List, Tuple

def parse_path_commands(d: str) -> List[Tuple[str, List[float]]]:
    """Parse SVG path d attribute into commands."""
    commands = []
    # Match command letter followed by numbers
    pattern = r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)'
    
    for match in re.finditer(pattern, d):
        cmd = match.group(1)
        params_str = match.group(2).strip()
        
        # Extract numbers
        params = []
        if params_str:
            # Match numbers (including negative and decimals)
            num_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
            params = [float(x) for x in re.findall(num_pattern, params_str)]
        
        commands.append((cmd, params))
    
    return commands

def render_svg_to_png(svg_path: Path, output_path: Path, size: int = 256):
    """
    Render SVG to PNG using pure PIL.
    
    This is a simplified renderer that handles basic paths only.
    """
    # Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Remove namespace
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    
    # Create white image
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Get viewBox
    viewbox = root.get('viewBox', '0 0 1 1')
    vx, vy, vw, vh = map(float, viewbox.split())
    
    # Scale factor from viewBox to pixel coordinates
    scale_x = size / vw
    scale_y = size / vh
    
    # Find all paths
    paths = root.findall('.//path')
    
    for path in paths:
        d = path.get('d', '')
        stroke_width = float(path.get('stroke-width', 0.06))
        
        if not d:
            continue
        
        # Parse path commands
        commands = parse_path_commands(d)
        
        # Convert to pixel coordinates and draw
        current_x, current_y = 0, 0
        path_points = []
        
        for cmd, params in commands:
            if cmd in ['M', 'm']:  # MoveTo
                if cmd == 'M':  # Absolute
                    current_x, current_y = params[0], params[1]
                else:  # Relative
                    current_x += params[0]
                    current_y += params[1]
                
                px = int(current_x * scale_x)
                py = int(current_y * scale_y)
                path_points.append((px, py))
            
            elif cmd in ['L', 'l']:  # LineTo
                for i in range(0, len(params), 2):
                    if cmd == 'L':  # Absolute
                        current_x, current_y = params[i], params[i+1]
                    else:  # Relative
                        current_x += params[i]
                        current_y += params[i+1]
                    
                    px = int(current_x * scale_x)
                    py = int(current_y * scale_y)
                    path_points.append((px, py))
            
            elif cmd in ['C', 'c']:  # Cubic Bezier
                # Simplified: just use endpoints
                for i in range(0, len(params), 6):
                    if cmd == 'C':  # Absolute
                        current_x, current_y = params[i+4], params[i+5]
                    else:  # Relative
                        current_x += params[i+4]
                        current_y += params[i+5]
                    
                    px = int(current_x * scale_x)
                    py = int(current_y * scale_y)
                    path_points.append((px, py))
            
            elif cmd in ['Z', 'z']:  # Close path
                if path_points:
                    path_points.append(path_points[0])
        
        # Draw the path
        if len(path_points) >= 2:
            line_width = max(1, int(stroke_width * size))
            draw.line(path_points, fill=0, width=line_width)
    
    # Save
    img.save(output_path, 'PNG')
    return True

# Test
if __name__ == "__main__":
    svg_file = Path("data_processed/svg_clean/icon_000003.svg")
    output_file = Path("test_simple_render.png")
    
    print(f"Rendering {svg_file}...")
    success = render_svg_to_png(svg_file, output_file)
    
    if success:
        print(f"✅ Saved to {output_file}")
        
        # Check content
        img = Image.open(output_file)
        pixels = list(img.getdata())
        unique = len(set(pixels))
        print(f"Unique pixel values: {unique}")
        print(f"Min: {min(pixels)}, Max: {max(pixels)}")
        
        if unique > 1 and min(pixels) < 200:
            print("✅ Image has visible content!")
        else:
            print("⚠️  Image might be blank")
    else:
        print("❌ Rendering failed")
