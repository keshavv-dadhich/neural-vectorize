"""
SVG initialization from raster images using edge detection.

Pipeline:
1. PNG → Canny edges → contours
2. Contours → polyline simplification
3. Polyline → cubic Bézier curves
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET


class EdgeInitializer:
    """Initialize SVG paths from raster images using edge detection."""
    
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        epsilon_factor: float = 0.002,  # For polyline simplification
        min_contour_length: int = 20,
    ):
        """
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            epsilon_factor: Douglas-Peucker approximation factor (relative to perimeter)
            min_contour_length: Minimum contour perimeter to keep
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.epsilon_factor = epsilon_factor
        self.min_contour_length = min_contour_length
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection to image.
        
        Args:
            image: Grayscale image (H, W) or RGB (H, W, 3)
            
        Returns:
            Binary edge map (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        return edges
    
    def extract_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from edge map.
        
        Args:
            edges: Binary edge map (H, W)
            
        Returns:
            List of contours, each shape (N, 1, 2)
        """
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_LIST,  # Get all contours without hierarchy
            cv2.CHAIN_APPROX_NONE  # Store all contour points
        )
        
        # Filter by minimum length
        filtered = [
            c for c in contours 
            if cv2.arcLength(c, closed=False) >= self.min_contour_length
        ]
        
        return filtered
    
    def simplify_polyline(self, contour: np.ndarray, closed: bool = False) -> np.ndarray:
        """
        Simplify contour using Douglas-Peucker algorithm.
        
        Args:
            contour: Shape (N, 1, 2) or (N, 2)
            closed: Whether the contour is closed
            
        Returns:
            Simplified contour, shape (M, 2) where M <= N
        """
        perimeter = cv2.arcLength(contour, closed)
        epsilon = self.epsilon_factor * perimeter
        
        approx = cv2.approxPolyDP(contour, epsilon, closed)
        
        # Reshape to (M, 2)
        if approx.shape[1] == 1:
            approx = approx.squeeze(1)
        
        return approx
    
    def polyline_to_bezier(self, points: np.ndarray, closed: bool = False) -> List[Tuple[str, np.ndarray]]:
        """
        Convert polyline to cubic Bézier curves.
        
        Strategy: For each pair of consecutive line segments, create a cubic Bézier
        that smoothly connects them. Control points are placed along the line segments.
        
        Args:
            points: Shape (N, 2), polyline vertices
            closed: Whether to close the path
            
        Returns:
            List of (command, points) tuples for SVG path commands
            - command: 'M' (moveto), 'L' (lineto), or 'C' (cubic bezier)
            - points: Associated coordinates
        """
        if len(points) < 2:
            return []
        
        path_commands = []
        
        # Start with MoveTo
        path_commands.append(('M', points[0]))
        
        # If only 2 points, just use a line
        if len(points) == 2:
            path_commands.append(('L', points[1]))
            if closed:
                path_commands.append(('Z', None))
            return path_commands
        
        # Convert segments to cubic Bézier curves
        # For each triplet of points (p0, p1, p2), create a Bézier curve from p0 to p2
        # that passes through or near p1
        
        for i in range(1, len(points)):
            if i == 1:
                # First segment: simple line or gentle curve
                path_commands.append(('L', points[i]))
            else:
                # Create smooth Bézier curve
                p0 = points[i-2]
                p1 = points[i-1]
                p2 = points[i] if i < len(points) else points[0]
                
                # Place control points at 1/3 and 2/3 along the segments
                c1 = p0 + (p1 - p0) * 2/3
                c2 = p1 + (p2 - p1) * 1/3
                
                # Create cubic Bézier: from p1 to p2 with controls c1, c2
                bezier_points = np.array([c1, c2, p2])
                path_commands.append(('C', bezier_points))
        
        # Close path if requested
        if closed:
            path_commands.append(('Z', None))
        
        return path_commands
    
    def normalize_coordinates(self, points: np.ndarray, image_size: int) -> np.ndarray:
        """
        Normalize coordinates from image space [0, size] to [0, 1].
        
        Args:
            points: Shape (N, 2) or (2,)
            image_size: Image dimension (assuming square)
            
        Returns:
            Normalized points in [0, 1] range
        """
        return points / image_size
    
    def path_commands_to_svg_string(self, commands: List[Tuple[str, Optional[np.ndarray]]]) -> str:
        """
        Convert path commands to SVG path 'd' attribute string.
        
        Args:
            commands: List of (command, points) tuples
            
        Returns:
            SVG path data string
        """
        parts = []
        
        for cmd, coords in commands:
            if cmd == 'Z':
                parts.append('Z')
            elif cmd == 'M' or cmd == 'L':
                parts.append(f"{cmd} {coords[0]:.6f},{coords[1]:.6f}")
            elif cmd == 'C':
                # coords should be shape (3, 2) for cubic Bézier
                c1, c2, end = coords
                parts.append(
                    f"{cmd} {c1[0]:.6f},{c1[1]:.6f} "
                    f"{c2[0]:.6f},{c2[1]:.6f} "
                    f"{end[0]:.6f},{end[1]:.6f}"
                )
        
        return ' '.join(parts)
    
    def initialize_from_raster(
        self, 
        image_path: Path, 
        output_svg_path: Optional[Path] = None,
        stroke_width: float = 0.06
    ) -> str:
        """
        Complete pipeline: raster → edges → contours → Bézier → SVG.
        
        Args:
            image_path: Path to input PNG
            output_svg_path: Optional path to save SVG
            stroke_width: Stroke width in normalized coordinates
            
        Returns:
            SVG string
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_size = image.shape[0]  # Assuming square
        
        # Detect edges
        edges = self.detect_edges(image)
        
        # Extract contours
        contours = self.extract_contours(edges)
        
        # Process each contour
        svg_paths = []
        for contour in contours:
            # Simplify
            simplified = self.simplify_polyline(contour, closed=False)
            
            # Normalize to [0, 1]
            normalized = self.normalize_coordinates(simplified, image_size)
            
            # Convert to Bézier commands
            commands = self.polyline_to_bezier(normalized, closed=False)
            
            # Convert to SVG path string
            path_d = self.path_commands_to_svg_string(commands)
            svg_paths.append(path_d)
        
        # Build SVG document
        svg_string = self._build_svg_document(svg_paths, stroke_width)
        
        # Optionally save
        if output_svg_path:
            output_svg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_svg_path, 'w') as f:
                f.write(svg_string)
        
        return svg_string
    
    def _build_svg_document(self, path_strings: List[str], stroke_width: float) -> str:
        """
        Build complete SVG document from path strings.
        
        Args:
            path_strings: List of SVG path 'd' attribute values
            stroke_width: Stroke width in normalized coordinates
            
        Returns:
            Complete SVG string
        """
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('viewBox', '0 0 1 1')
        svg.set('width', '256')
        svg.set('height', '256')
        
        for path_d in path_strings:
            path = ET.SubElement(svg, 'path')
            path.set('d', path_d)
            path.set('stroke', 'black')
            path.set('stroke-width', str(stroke_width))
            path.set('fill', 'none')
        
        # Convert to string
        return ET.tostring(svg, encoding='unicode', method='xml')


def test_initialization():
    """Test edge initialization on a sample image."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RASTER_PERFECT, STROKE_WIDTH
    
    # Get a test raster
    test_rasters = list(RASTER_PERFECT.glob('*.png'))
    if not test_rasters:
        print("No test rasters found!")
        return
    
    test_image = test_rasters[0]
    print(f"Testing initialization on: {test_image.name}")
    
    # Initialize
    initializer = EdgeInitializer(
        canny_low=50,
        canny_high=150,
        epsilon_factor=0.002,
        min_contour_length=20
    )
    
    # Run pipeline
    output_path = Path('outputs/test_initialization.svg')
    svg_string = initializer.initialize_from_raster(
        test_image, 
        output_path,
        stroke_width=STROKE_WIDTH
    )
    
    print(f"✅ Initialized SVG saved to: {output_path}")
    print(f"SVG length: {len(svg_string)} characters")


if __name__ == '__main__':
    test_initialization()
