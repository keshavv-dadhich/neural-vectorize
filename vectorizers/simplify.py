"""
Phase C: Post-optimization simplification.

Operations:
1. Merge collinear segments (reduce point count)
2. Remove tiny/insignificant paths
3. Optional: Re-optimize for a few more steps

This is where we beat Potrace on segment count while maintaining quality.
"""

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from typing import List, Tuple


def compute_collinearity(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute how collinear three points are.
    
    Returns angle deviation from 180° (perfectly collinear).
    
    Args:
        p1, p2, p3: Points, each shape (2,)
        
    Returns:
        Angle deviation in radians (0 = perfectly collinear)
    """
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Normalize
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Dot product gives cos(angle)
    cos_angle = np.dot(v1, v2)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle = np.arccos(cos_angle)
    
    # Deviation from 180° (π radians)
    return abs(angle - np.pi)


def merge_collinear_segments(
    points: np.ndarray,
    threshold: float = 0.1  # radians, ~5.7 degrees
) -> np.ndarray:
    """
    Merge nearly collinear consecutive segments.
    
    Args:
        points: Shape (N, 2), path vertices
        threshold: Max angle deviation to consider collinear (radians)
        
    Returns:
        Simplified points, shape (M, 2) where M <= N
    """
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    
    for i in range(1, len(points) - 1):
        prev = simplified[-1]
        curr = points[i]
        next_pt = points[i + 1]
        
        # Check collinearity
        deviation = compute_collinearity(prev, curr, next_pt)
        
        if deviation > threshold:
            # Not collinear, keep the point
            simplified.append(curr)
    
    # Always keep last point
    simplified.append(points[-1])
    
    return np.array(simplified)


def compute_path_length(points: np.ndarray) -> float:
    """Compute total path length."""
    if len(points) < 2:
        return 0.0
    
    diffs = points[1:] - points[:-1]
    lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(lengths)


def parse_svg_paths(svg_path: Path) -> List[Tuple[np.ndarray, dict]]:
    """
    Parse SVG and extract paths.
    
    Returns:
        List of (points, attributes) tuples
        - points: shape (N, 2), path vertices
        - attributes: dict of SVG attributes (stroke, fill, etc.)
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    paths = []
    
    for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path_elem.get('d', '')
        
        # Extract coordinates
        coords = []
        tokens = re.findall(r'[ML]\s*[\d.]+,[\d.]+', d)
        
        for token in tokens:
            match = re.search(r'([\d.]+),([\d.]+)', token)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                coords.append([x, y])
        
        if coords:
            points = np.array(coords)
            
            # Get attributes
            attrs = {
                'stroke': path_elem.get('stroke', 'black'),
                'stroke-width': path_elem.get('stroke-width', '0.06'),
                'fill': path_elem.get('fill', 'none')
            }
            
            paths.append((points, attrs))
    
    return paths


def build_svg_from_paths(
    paths: List[Tuple[np.ndarray, dict]],
    output_path: Path
):
    """Build and save SVG from path list."""
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('viewBox', '0 0 1 1')
    svg.set('width', '256')
    svg.set('height', '256')
    
    for points, attrs in paths:
        # Build path d attribute
        d_parts = [f"M {points[0][0]:.6f},{points[0][1]:.6f}"]
        for i in range(1, len(points)):
            d_parts.append(f"L {points[i][0]:.6f},{points[i][1]:.6f}")
        
        d = ' '.join(d_parts)
        
        # Create path element
        path = ET.SubElement(svg, 'path')
        path.set('d', d)
        for key, value in attrs.items():
            path.set(key, value)
    
    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(ET.tostring(svg, encoding='unicode', method='xml'))


def simplify_svg(
    input_svg: Path,
    output_svg: Path,
    merge_threshold: float = 0.1,  # radians for collinearity
    min_path_length: float = 0.02,  # minimum path length (normalized)
    verbose: bool = True
):
    """
    Simplify an SVG by merging collinear segments and removing tiny paths.
    
    Args:
        input_svg: Input SVG path
        output_svg: Output SVG path
        merge_threshold: Max angle deviation for merging (radians)
        min_path_length: Minimum path length to keep
        verbose: Print stats
    """
    # Parse
    paths = parse_svg_paths(input_svg)
    
    if verbose:
        total_points_before = sum(len(p[0]) for p in paths)
        print(f"  Input: {len(paths)} paths, {total_points_before} control points")
    
    # Simplify each path
    simplified_paths = []
    
    for points, attrs in paths:
        # Merge collinear segments
        simplified_points = merge_collinear_segments(points, merge_threshold)
        
        # Check minimum length
        path_length = compute_path_length(simplified_points)
        if path_length >= min_path_length:
            simplified_paths.append((simplified_points, attrs))
    
    # Build output SVG
    build_svg_from_paths(simplified_paths, output_svg)
    
    if verbose:
        total_points_after = sum(len(p[0]) for p in simplified_paths)
        print(f"  Output: {len(simplified_paths)} paths, {total_points_after} control points")
        print(f"  Reduction: {100*(1 - total_points_after/total_points_before):.1f}% fewer points")


def test_simplification():
    """Test simplification on optimized SVG."""
    input_svg = Path('outputs/test_optimized.svg')
    output_svg = Path('outputs/test_simplified.svg')
    
    if not input_svg.exists():
        print("Run vectorizers/optimize.py first!")
        return
    
    print("\nTesting simplification...")
    simplify_svg(
        input_svg,
        output_svg,
        merge_threshold=0.1,  # ~5.7 degrees
        min_path_length=0.02,
        verbose=True
    )
    print(f"\n✅ Simplified SVG: {output_svg}")


if __name__ == '__main__':
    test_simplification()
