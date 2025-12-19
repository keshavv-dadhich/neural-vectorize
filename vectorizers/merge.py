"""
Aggressive segment reduction strategies.

Goal: Reduce from 252 → <150 segments while maintaining L2 ≤ 0.24

Strategies:
1. Collinear segment merging (angle threshold)
2. Douglas-Peucker simplification (adaptive epsilon)
3. Path pruning by reconstruction contribution
4. Control point clustering
"""

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from typing import List, Tuple, Optional


def angle_between_segments(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute angle between segments p1→p2 and p2→p3 in degrees.
    
    Args:
        p1, p2, p3: Points, each shape (2,)
        
    Returns:
        Angle in degrees [0, 180]
    """
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Compute angle
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    return np.degrees(angle_rad)


def merge_collinear_segments(
    points: np.ndarray,
    angle_threshold: float = 5.0,
    min_points: int = 2
) -> np.ndarray:
    """
    Merge consecutive collinear segments.
    
    If three consecutive points p1, p2, p3 form an angle close to 180°,
    remove the middle point p2.
    
    Args:
        points: Path points, shape (N, 2)
        angle_threshold: Maximum deviation from 180° to merge (degrees)
        min_points: Minimum points to keep
        
    Returns:
        Simplified points, shape (M, 2) where M ≤ N
    """
    if len(points) < 3:
        return points
    
    kept = [points[0]]  # Always keep first point
    
    for i in range(1, len(points) - 1):
        # Check angle at points[i]
        angle = angle_between_segments(points[i-1], points[i], points[i+1])
        
        # If angle is close to 180°, skip this point (merge segments)
        if abs(180.0 - angle) > angle_threshold:
            kept.append(points[i])
    
    kept.append(points[-1])  # Always keep last point
    
    result = np.array(kept)
    
    # Ensure minimum points
    if len(result) < min_points:
        return points
    
    return result


def douglas_peucker_adaptive(
    points: np.ndarray,
    epsilon: float = 0.005,
    max_iterations: int = 10,
    target_reduction: float = 0.5
) -> np.ndarray:
    """
    Apply Douglas-Peucker with adaptive epsilon to hit target reduction.
    
    Args:
        points: Path points, shape (N, 2)
        epsilon: Initial epsilon
        max_iterations: Maximum adaptation iterations
        target_reduction: Target fraction of points to keep (0.5 = 50%)
        
    Returns:
        Simplified points
    """
    from scipy.spatial.distance import pdist, squareform
    
    def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
        """Ramer-Douglas-Peucker algorithm."""
        if len(points) < 3:
            return points
        
        # Find point with maximum distance from line p[0]→p[-1]
        first = points[0]
        last = points[-1]
        
        # Compute perpendicular distances
        line_vec = last - first
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-8:
            return np.array([first, last])
        
        line_unit = line_vec / line_len
        
        distances = []
        for i in range(1, len(points) - 1):
            point_vec = points[i] - first
            proj_length = np.dot(point_vec, line_unit)
            proj = first + proj_length * line_unit
            dist = np.linalg.norm(points[i] - proj)
            distances.append(dist)
        
        if not distances:
            return np.array([first, last])
        
        max_dist = max(distances)
        max_idx = distances.index(max_dist) + 1
        
        # If max distance > epsilon, recurse
        if max_dist > epsilon:
            left = rdp(points[:max_idx+1], epsilon)
            right = rdp(points[max_idx:], epsilon)
            return np.vstack([left[:-1], right])
        else:
            return np.array([first, last])
    
    # Adaptive epsilon
    original_count = len(points)
    target_count = int(original_count * target_reduction)
    
    current_epsilon = epsilon
    for iteration in range(max_iterations):
        simplified = rdp(points, current_epsilon)
        
        if len(simplified) <= target_count:
            return simplified
        
        # Increase epsilon to simplify more
        current_epsilon *= 1.5
    
    return simplified


def remove_tiny_paths(
    paths: List[np.ndarray],
    min_length: float = 0.01,
    min_bbox: float = 0.05
) -> List[np.ndarray]:
    """
    Remove paths that are too small to matter.
    
    Args:
        paths: List of path arrays
        min_length: Minimum total path length
        min_bbox: Minimum bounding box diagonal
        
    Returns:
        Filtered paths
    """
    kept = []
    
    for path in paths:
        # Compute total length
        if len(path) < 2:
            continue
        
        diffs = np.diff(path, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        total_length = lengths.sum()
        
        if total_length < min_length:
            continue
        
        # Compute bounding box diagonal
        bbox_min = path.min(axis=0)
        bbox_max = path.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        
        if bbox_diag < min_bbox:
            continue
        
        kept.append(path)
    
    return kept


def cluster_nearby_points(
    points: np.ndarray,
    radius: float = 0.02
) -> np.ndarray:
    """
    Cluster nearby points and replace with centroids.
    
    Args:
        points: Path points, shape (N, 2)
        radius: Clustering radius
        
    Returns:
        Clustered points
    """
    if len(points) < 2:
        return points
    
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    
    # Compute pairwise distances
    distances = pdist(points)
    
    # Hierarchical clustering
    Z = linkage(distances, method='single')
    clusters = fcluster(Z, t=radius, criterion='distance')
    
    # Compute cluster centroids
    unique_clusters = np.unique(clusters)
    centroids = []
    
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        centroid = points[mask].mean(axis=0)
        centroids.append(centroid)
    
    return np.array(centroids)


def aggressive_simplify_svg(
    input_svg: Path,
    output_svg: Path,
    angle_threshold: float = 5.0,
    epsilon: float = 0.005,
    target_reduction: float = 0.6,  # Keep 60% of points
    min_path_length: float = 0.01,
    cluster_radius: float = 0.015,
    verbose: bool = True
) -> dict:
    """
    Apply all simplification strategies aggressively.
    
    Args:
        input_svg: Input SVG path
        output_svg: Output SVG path
        angle_threshold: Collinear merge threshold (degrees)
        epsilon: Douglas-Peucker epsilon
        target_reduction: Fraction of points to keep
        min_path_length: Minimum path length
        cluster_radius: Point clustering radius
        verbose: Print stats
        
    Returns:
        Statistics dict
    """
    # Parse SVG
    tree = ET.parse(input_svg)
    root = tree.getroot()
    
    stats = {
        'original_paths': 0,
        'original_segments': 0,
        'final_paths': 0,
        'final_segments': 0
    }
    
    new_paths = []
    
    # Process each path
    for elem in root.iter():
        if 'path' not in str(elem.tag).lower():
            continue
        
        stats['original_paths'] += 1
        
        d = elem.get('d', '')
        
        # Extract coordinates
        coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
        if len(coords) < 4:
            continue
        
        coords = [float(c) for c in coords]
        points = np.array(coords).reshape(-1, 2)
        
        stats['original_segments'] += len(points)
        
        # Strategy 1: Merge collinear segments
        points = merge_collinear_segments(points, angle_threshold)
        
        # Strategy 2: Douglas-Peucker adaptive
        points = douglas_peucker_adaptive(points, epsilon, target_reduction=target_reduction)
        
        # Strategy 3: Cluster nearby points
        if len(points) > 3:
            points = cluster_nearby_points(points, cluster_radius)
        
        new_paths.append(points)
    
    # Strategy 4: Remove tiny paths
    new_paths = remove_tiny_paths(new_paths, min_path_length)
    
    # Build output SVG
    svg_elem = ET.Element('svg')
    svg_elem.set('xmlns', 'http://www.w3.org/2000/svg')
    svg_elem.set('viewBox', '0 0 1 1')
    svg_elem.set('width', '256')
    svg_elem.set('height', '256')
    
    for points in new_paths:
        if len(points) < 2:
            continue
        
        stats['final_paths'] += 1
        stats['final_segments'] += len(points)
        
        # Build path string
        path_d = f"M {points[0][0]:.6f},{points[0][1]:.6f}"
        for i in range(1, len(points)):
            path_d += f" L {points[i][0]:.6f},{points[i][1]:.6f}"
        
        path_elem = ET.SubElement(svg_elem, 'path')
        path_elem.set('d', path_d)
        path_elem.set('stroke', 'black')
        path_elem.set('stroke-width', '0.06')
        path_elem.set('fill', 'none')
    
    # Write output
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    tree_out = ET.ElementTree(svg_elem)
    tree_out.write(output_svg, encoding='unicode', xml_declaration=True)
    
    # Compute statistics
    stats['reduction_rate'] = 1.0 - (stats['final_segments'] / max(stats['original_segments'], 1))
    
    if verbose:
        print(f"Simplification Statistics:")
        print(f"  Paths: {stats['original_paths']} → {stats['final_paths']}")
        print(f"  Segments: {stats['original_segments']} → {stats['final_segments']}")
        print(f"  Reduction: {stats['reduction_rate']*100:.1f}%")
    
    return stats


if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Test on optimization output
    test_svg = Path('baselines/optimization/icon_426509.svg')
    output_svg = Path('outputs/test_aggressive_merge.svg')
    
    if not test_svg.exists():
        print(f"Test file not found: {test_svg}")
        print("Run benchmark first to generate optimization outputs")
    else:
        stats = aggressive_simplify_svg(
            test_svg,
            output_svg,
            angle_threshold=5.0,
            epsilon=0.008,  # Slightly more aggressive
            target_reduction=0.5,  # Keep only 50%
            verbose=True
        )
        
        print(f"\n✅ Aggressively simplified SVG saved to: {output_svg}")
