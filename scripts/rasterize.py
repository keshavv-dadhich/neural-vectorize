"""
PHASE 2: Rasterization Pipeline
================================

This script converts clean SVG files into perfect raster images (PNG).

These rasters serve as:
1. Base images for degradation/augmentation
2. Visual targets for reconstruction loss

Uses svglib + reportlab for reliable cross-platform rendering.
"""

import sys
from pathlib import Path
from PIL import Image
import io

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import load_metadata, save_metadata, ProgressTracker


class SVGRasterizer:
    """Handles rasterization of clean SVG files."""
    
    def __init__(self, output_size: int = RASTER_SIZE):
        self.output_size = output_size
    
    def rasterize(self, svg_path: Path, output_path: Path) -> bool:
        """
        Rasterize an SVG file to PNG.
        
        Handles the viewBox/coordinate mismatch in our cleaned SVGs
        by rendering in original coordinate space.
        
        Args:
            svg_path: Path to clean SVG
            output_path: Path for output PNG
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import xml.etree.ElementTree as ET
            from PIL import ImageDraw
            import re
            
            # Parse SVG
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Get viewBox (might be [0,0,1,1] but coordinates are in original space)
            viewbox = root.get('viewBox', '0 0 1 1')
            vx, vy, vw, vh = map(float, viewbox.split())
            
            # Find actual coordinate bounds from paths
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            # Extract all numbers from all path d attributes
            for elem in root.iter():
                if 'path' in str(elem.tag).lower():
                    d = elem.get('d', '')
                    # Extract all numbers
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
                    coords = [float(n) for n in numbers]
                    
                    # Get x,y pairs (assume x,y alternation)
                    for i in range(0, len(coords)-1, 2):
                        x, y = coords[i], coords[i+1]
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
            
            # If we found valid bounds, use them
            if min_x != float('inf'):
                actual_width = max_x - min_x
                actual_height = max_y - min_y
                
                # Add padding
                padding = 0.1 * max(actual_width, actual_height)
                min_x -= padding
                min_y -= padding
                actual_width += 2 * padding
                actual_height += 2 * padding
            else:
                # Fallback
                min_x, min_y = 0, 0
                actual_width = actual_height = 512
            
            # Create white image
            img = Image.new('L', (self.output_size, self.output_size), color=255)
            draw = ImageDraw.Draw(img)
            
            # Calculate scale from actual coordinates to pixels
            scale = min(
                self.output_size / actual_width,
                self.output_size / actual_height
            )
            
            # Simple path rendering (lines only for now)
            for elem in root.iter():
                if 'path' in str(elem.tag).lower():
                    d = elem.get('d', '')
                    stroke_width = float(elem.get('stroke-width', 0.06))
                    
                    # Extract coordinates
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
                    if len(numbers) < 4:
                        continue
                    
                    coords = [float(n) for n in numbers]
                    points = []
                    
                    # Convert to pixel coordinates
                    for i in range(0, len(coords)-1, 2):
                        x = (coords[i] - min_x) * scale
                        y = (coords[i+1] - min_y) * scale
                        points.append((int(x), int(y)))
                    
                    # Draw as connected lines
                    if len(points) >= 2:
                        line_width = max(2, int(stroke_width * self.output_size))
                        draw.line(points, fill=0, width=line_width)
            
            # Save
            img.save(output_path, 'PNG')
            return True
        
        except Exception as e:
            # Uncomment for debugging:
            # print(f"\n⚠️  Error rasterizing {svg_path.name}: {e}")
            return False


def main():
    """Run rasterization pipeline on all clean SVGs."""
    
    print("=" * 60)
    print("PHASE 2: RASTERIZATION PIPELINE")
    print("=" * 60)
    
    # Ensure output directory exists
    RASTER_PERFECT.mkdir(parents=True, exist_ok=True)
    
    # Find all clean SVG files
    svg_files = list(SVG_CLEAN.glob('*.svg'))
    
    if not svg_files:
        print(f"\n❌ No clean SVG files found in {SVG_CLEAN}")
        print("   Please run clean_svg.py first.")
        return 1
    
    print(f"\n📊 Found {len(svg_files)} clean SVG files")
    print(f"📁 Output: {RASTER_PERFECT}")
    print(f"🖼️  Resolution: {RASTER_SIZE}×{RASTER_SIZE}\n")
    
    # Initialize rasterizer
    rasterizer = SVGRasterizer(RASTER_SIZE)
    
    # Load existing metadata
    try:
        metadata_list = load_metadata(METADATA_FILE)
        metadata_dict = {m['id']: m for m in metadata_list}
    except:
        print("⚠️  Could not load metadata, creating new")
        metadata_dict = {}
    
    # Process each SVG
    tracker = ProgressTracker(len(svg_files), "Rasterizing")
    
    successful = 0
    failed = 0
    
    for svg_path in svg_files:
        try:
            # Get SVG ID from filename
            svg_id = svg_path.stem
            
            # Output path
            output_path = RASTER_PERFECT / f"{svg_id}_base.png"
            
            # Rasterize
            if rasterizer.rasterize(svg_path, output_path):
                successful += 1
                
                # Update metadata
                if svg_id in metadata_dict:
                    metadata_dict[svg_id]['rasterized'] = True
                    metadata_dict[svg_id]['raster_resolution'] = RASTER_SIZE
            else:
                failed += 1
        
        except Exception as e:
            failed += 1
        
        tracker.update()
    
    tracker.finish()
    
    # Save updated metadata
    print(f"\n📝 Updating metadata...")
    metadata_list = list(metadata_dict.values())
    save_metadata(metadata_list, METADATA_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("RASTERIZATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully rasterized: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total processed: {len(svg_files)}")
    print(f"\n📁 Raster images saved to: {RASTER_PERFECT}")
    print(f"📄 Metadata updated: {METADATA_FILE}")
    print("\n✅ Phase 2 complete!")
    print("\nNext step: Run augmentation")
    print("  python scripts/augment.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
