"""
PHASE 1: Clean SVG Pipeline
============================

This script processes raw SVG files into clean, normalized ground truth SVGs.

Steps:
1. Remove background rectangles/paths
2. Normalize geometry to [0,1] viewBox
3. Convert fills → strokes (outlines)
4. Simplify paths (remove tiny segments, merge collinear lines)
5. Output stroke-only, normalized SVG

This is the MOST IMPORTANT step in the pipeline.
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import re
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import generate_svg_id, save_metadata, validate_svg_file, ProgressTracker

# SVG namespace
SVG_NS = {'svg': 'http://www.w3.org/2000/svg'}


class SVGCleaner:
    """Handles cleaning and normalization of raw SVG files."""
    
    def __init__(self):
        self.stroke_width = STROKE_WIDTH
        self.epsilon = EPSILON
        self.min_segment_length = MIN_SEGMENT_LENGTH
    
    def parse_viewbox(self, svg_root: ET.Element) -> Tuple[float, float, float, float]:
        """Extract viewBox dimensions from SVG root."""
        viewbox = svg_root.get('viewBox')
        if viewbox:
            parts = viewbox.split()
            return tuple(map(float, parts))
        
        # Fallback to width/height
        width = float(svg_root.get('width', 100))
        height = float(svg_root.get('height', 100))
        return (0, 0, width, height)
    
    def is_background_rect(self, element: ET.Element, viewbox: Tuple[float, float, float, float]) -> bool:
        """
        Detect if a rectangle/path is a background element.
        
        Heuristics:
        - Covers entire viewBox
        - Has white or light fill
        - Is a simple rectangle
        """
        tag = element.tag.replace('{http://www.w3.org/2000/svg}', '')
        
        if tag == 'rect':
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            w = float(element.get('width', 0))
            h = float(element.get('height', 0))
            
            vx, vy, vw, vh = viewbox
            
            # Check if rect covers viewBox
            covers_viewbox = (
                abs(x - vx) < 1 and abs(y - vy) < 1 and
                abs(w - vw) < 1 and abs(h - vh) < 1
            )
            
            if covers_viewbox:
                fill = element.get('fill', '').lower()
                if fill in ['white', '#fff', '#ffffff', 'none', '']:
                    return True
        
        return False
    
    def remove_backgrounds(self, svg_root: ET.Element):
        """Remove background rectangles and paths."""
        viewbox = self.parse_viewbox(svg_root)
        
        to_remove = []
        for elem in svg_root.iter():
            if self.is_background_rect(elem, viewbox):
                to_remove.append(elem)
        
        for elem in to_remove:
            parent = svg_root.find('.//*[.="%s"]/..' % elem)
            if parent is not None:
                parent.remove(elem)
            else:
                svg_root.remove(elem)
    
    def normalize_viewbox(self, svg_root: ET.Element):
        """
        Normalize viewBox to [0,1] × [0,1] AND transform all path coordinates.
        
        This is THE CRITICAL STEP - we must transform geometry, not just viewBox.
        """
        vx, vy, vw, vh = self.parse_viewbox(svg_root)
        
        # Calculate transformation: scale and translate to [0,1]
        scale_x = VIEWBOX_SIZE / vw if vw > 0 else 1
        scale_y = VIEWBOX_SIZE / vh if vh > 0 else 1
        
        # Transform all path coordinates
        for elem in svg_root.iter():
            # Handle paths
            if 'path' in str(elem.tag).lower():
                d = elem.get('d', '')
                if d:
                    transformed_d = self._transform_path_data(d, vx, vy, scale_x, scale_y)
                    elem.set('d', transformed_d)
            
            # Handle other geometric elements
            elif 'rect' in str(elem.tag).lower():
                self._transform_rect(elem, vx, vy, scale_x, scale_y)
            elif 'circle' in str(elem.tag).lower():
                self._transform_circle(elem, vx, vy, scale_x, scale_y)
            elif 'ellipse' in str(elem.tag).lower():
                self._transform_ellipse(elem, vx, vy, scale_x, scale_y)
        
        # NOW set normalized viewBox
        svg_root.set('viewBox', f'0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}')
        svg_root.set('width', str(VIEWBOX_SIZE))
        svg_root.set('height', str(VIEWBOX_SIZE))
        
        return scale_x, scale_y
    
    def _transform_path_data(self, d: str, tx: float, ty: float, sx: float, sy: float) -> str:
        """
        Transform SVG path data coordinates.
        
        Args:
            d: Path data string
            tx, ty: Translation (viewBox origin)
            sx, sy: Scale factors
        
        Returns:
            Transformed path data string
        """
        import re
        
        # Split into commands and parameters
        result = []
        current_pos = [0.0, 0.0]  # Track current position for relative commands
        
        # Pattern to match commands and their parameters
        cmd_pattern = r'([MmLlHhVvCcSsQqTtAaZz])((?:\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?\s*,?\s*)*)'
        
        for match in re.finditer(cmd_pattern, d):
            cmd = match.group(1)
            params_str = match.group(2).strip()
            
            # Extract numeric parameters
            if params_str:
                params = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', params_str)]
            else:
                params = []
            
            # Transform based on command type
            if cmd in ['M', 'L', 'T']:  # Absolute commands with x,y pairs
                transformed = []
                for i in range(0, len(params), 2):
                    x = (params[i] - tx) * sx
                    y = (params[i+1] - ty) * sy
                    transformed.extend([x, y])
                    current_pos = [x, y]
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd in ['m', 'l', 't']:  # Relative commands with dx,dy pairs
                transformed = []
                for i in range(0, len(params), 2):
                    dx = params[i] * sx
                    dy = params[i+1] * sy
                    transformed.extend([dx, dy])
                    current_pos[0] += dx
                    current_pos[1] += dy
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd == 'H':  # Absolute horizontal
                transformed = [(params[i] - tx) * sx for i in range(len(params))]
                current_pos[0] = transformed[-1] if transformed else current_pos[0]
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd == 'h':  # Relative horizontal
                transformed = [params[i] * sx for i in range(len(params))]
                current_pos[0] += transformed[-1] if transformed else 0
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd == 'V':  # Absolute vertical
                transformed = [(params[i] - ty) * sy for i in range(len(params))]
                current_pos[1] = transformed[-1] if transformed else current_pos[1]
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd == 'v':  # Relative vertical
                transformed = [params[i] * sy for i in range(len(params))]
                current_pos[1] += transformed[-1] if transformed else 0
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd in ['C', 'S', 'Q']:  # Absolute curve commands
                transformed = []
                for i in range(0, len(params), 2):
                    x = (params[i] - tx) * sx
                    y = (params[i+1] - ty) * sy
                    transformed.extend([x, y])
                if transformed:
                    current_pos = transformed[-2:]
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd in ['c', 's', 'q']:  # Relative curve commands
                transformed = []
                for i in range(0, len(params), 2):
                    dx = params[i] * sx
                    dy = params[i+1] * sy
                    transformed.extend([dx, dy])
                if len(transformed) >= 2:
                    current_pos[0] += transformed[-2]
                    current_pos[1] += transformed[-1]
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd in ['A']:  # Arc (absolute)
                # A rx ry x-axis-rotation large-arc-flag sweep-flag x y
                transformed = []
                for i in range(0, len(params), 7):
                    rx = params[i] * sx
                    ry = params[i+1] * sy
                    rotation = params[i+2]
                    large_arc = params[i+3]
                    sweep = params[i+4]
                    x = (params[i+5] - tx) * sx
                    y = (params[i+6] - ty) * sy
                    transformed.extend([rx, ry, rotation, large_arc, sweep, x, y])
                    current_pos = [x, y]
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd in ['a']:  # Arc (relative)
                transformed = []
                for i in range(0, len(params), 7):
                    rx = params[i] * sx
                    ry = params[i+1] * sy
                    rotation = params[i+2]
                    large_arc = params[i+3]
                    sweep = params[i+4]
                    dx = params[i+5] * sx
                    dy = params[i+6] * sy
                    transformed.extend([rx, ry, rotation, large_arc, sweep, dx, dy])
                    current_pos[0] += dx
                    current_pos[1] += dy
                result.append(cmd + ' ' + ' '.join(f'{v:.6f}' for v in transformed))
            
            elif cmd in ['Z', 'z']:  # Close path
                result.append(cmd)
        
        return ' '.join(result)
    
    def _transform_rect(self, elem: ET.Element, tx: float, ty: float, sx: float, sy: float):
        """Transform rectangle coordinates."""
        if elem.get('x'):
            elem.set('x', str((float(elem.get('x')) - tx) * sx))
        if elem.get('y'):
            elem.set('y', str((float(elem.get('y')) - ty) * sy))
        if elem.get('width'):
            elem.set('width', str(float(elem.get('width')) * sx))
        if elem.get('height'):
            elem.set('height', str(float(elem.get('height')) * sy))
    
    def _transform_circle(self, elem: ET.Element, tx: float, ty: float, sx: float, sy: float):
        """Transform circle coordinates."""
        if elem.get('cx'):
            elem.set('cx', str((float(elem.get('cx')) - tx) * sx))
        if elem.get('cy'):
            elem.set('cy', str((float(elem.get('cy')) - ty) * sy))
        if elem.get('r'):
            elem.set('r', str(float(elem.get('r')) * min(sx, sy)))
    
    def _transform_ellipse(self, elem: ET.Element, tx: float, ty: float, sx: float, sy: float):
        """Transform ellipse coordinates."""
        if elem.get('cx'):
            elem.set('cx', str((float(elem.get('cx')) - tx) * sx))
        if elem.get('cy'):
            elem.set('cy', str((float(elem.get('cy')) - ty) * sy))
        if elem.get('rx'):
            elem.set('rx', str(float(elem.get('rx')) * sx))
        if elem.get('ry'):
            elem.set('ry', str(float(elem.get('ry')) * sy))
    
    def convert_fill_to_stroke(self, svg_root: ET.Element):
        """
        Convert filled paths to stroke-only paths.
        
        This is a simplified version. Full implementation would:
        - Extract path boundary
        - Convert to outline
        - Set stroke, remove fill
        """
        for path in svg_root.findall('.//svg:path', SVG_NS):
            # Remove fill, set stroke
            path.set('fill', 'none')
            path.set('stroke', 'black')
            path.set('stroke-width', str(self.stroke_width))
            
            # Remove color attributes
            for attr in ['fill-opacity', 'opacity', 'fill-rule']:
                if attr in path.attrib:
                    del path.attrib[attr]
        
        # Handle other shape elements (circle, rect, etc.)
        for tag in ['circle', 'rect', 'ellipse', 'line', 'polyline', 'polygon']:
            for elem in svg_root.findall(f'.//svg:{tag}', SVG_NS):
                elem.set('fill', 'none')
                elem.set('stroke', 'black')
                elem.set('stroke-width', str(self.stroke_width))
    
    def remove_transforms(self, svg_root: ET.Element):
        """Remove transform attributes (flatten transformations)."""
        # TODO: Proper implementation would apply transforms to coordinates
        for elem in svg_root.iter():
            if 'transform' in elem.attrib:
                del elem.attrib['transform']
    
    def simplify_paths(self, svg_root: ET.Element):
        """
        Simplify paths by:
        - Removing tiny segments
        - Merging collinear lines
        
        This is a placeholder - full implementation requires svgpathtools.
        """
        # TODO: Implement with svgpathtools
        pass
    
    def clean_svg(self, input_path: Path, output_path: Path) -> dict:
        """
        Main cleaning pipeline for a single SVG.
        
        Returns:
            Metadata dict for this SVG
        """
        # Parse SVG
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # Ensure proper namespace
        if not root.tag.endswith('svg'):
            return None
        
        # Apply cleaning steps
        self.remove_backgrounds(root)
        self.normalize_viewbox(root)
        self.convert_fill_to_stroke(root)
        self.remove_transforms(root)
        self.simplify_paths(root)
        
        # Write cleaned SVG
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        # Count primitives
        num_paths = len(root.findall('.//svg:path', SVG_NS))
        
        # Generate metadata
        svg_id = generate_svg_id(input_path)
        metadata = {
            'id': svg_id,
            'original_file': input_path.name,
            'num_paths': num_paths,
            'cleaned': True
        }
        
        return metadata


def main():
    """Run the SVG cleaning pipeline on all raw SVGs."""
    
    print("=" * 60)
    print("PHASE 1: SVG CLEANING PIPELINE")
    print("=" * 60)
    
    # Ensure output directory exists
    SVG_CLEAN.mkdir(parents=True, exist_ok=True)
    
    # Find all raw SVG files
    svg_files = list(SVGS_RAW.glob('*.svg'))
    
    if not svg_files:
        print(f"\n❌ No SVG files found in {SVGS_RAW}")
        print("   Please add raw SVG files and run again.")
        return 1
    
    print(f"\n📊 Found {len(svg_files)} raw SVG files")
    print(f"📁 Output: {SVG_CLEAN}\n")
    
    # Initialize cleaner
    cleaner = SVGCleaner()
    
    # Process each SVG
    metadata_list = []
    tracker = ProgressTracker(len(svg_files), "Cleaning SVGs")
    
    successful = 0
    failed = 0
    
    for svg_path in svg_files:
        try:
            # Validate first
            if not validate_svg_file(svg_path):
                failed += 1
                tracker.update()
                continue
            
            # Generate output filename
            svg_id = generate_svg_id(svg_path)
            output_path = SVG_CLEAN / f"{svg_id}.svg"
            
            # Clean the SVG
            metadata = cleaner.clean_svg(svg_path, output_path)
            
            if metadata:
                metadata_list.append(metadata)
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            failed += 1
            # Uncomment for debugging:
            # print(f"\n⚠️  Error processing {svg_path.name}: {e}")
        
        tracker.update()
    
    tracker.finish()
    
    # Save metadata
    print(f"\n📝 Saving metadata...")
    save_metadata(metadata_list, METADATA_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully cleaned: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total processed: {len(svg_files)}")
    print(f"\n📁 Clean SVGs saved to: {SVG_CLEAN}")
    print(f"📄 Metadata saved to: {METADATA_FILE}")
    print("\n✅ Phase 1 complete!")
    print("\nNext step: Run rasterization")
    print("  python scripts/rasterize.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
