"""
Optimization-based vectorizer using differentiable rendering.

This is Phase B: Take an initialized SVG and optimize it to better match the target raster.

Pipeline:
1. Load initialized SVG (from Phase A)
2. Convert to differentiable representation (control points as tensors)
3. Define loss functions (reconstruction + complexity + coverage)
4. Optimize with Adam for ~300-500 steps
5. Export optimized SVG

For now, we implement our own simple differentiable rasterizer instead of DiffVG.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
import re
from PIL import Image


class DifferentiableVectorizer:
    """Optimize SVG paths to match target raster using gradient descent."""
    
    def __init__(
        self,
        image_size: int = 256,
        device: str = 'cpu'
    ):
        """
        Args:
            image_size: Raster resolution
            device: 'cpu' or 'cuda'
        """
        self.image_size = image_size
        self.device = device
    
    def svg_to_tensors(self, svg_path: Path) -> List[torch.Tensor]:
        """
        Parse SVG and extract control points as tensors.
        
        Args:
            svg_path: Path to SVG file
            
        Returns:
            List of tensors, each shape (N, 2) representing path control points
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        path_tensors = []
        
        for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
            d = path_elem.get('d', '')
            
            # Extract all coordinates
            coords = []
            tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
            
            for token in tokens:
                match = re.search(r'([\d.]+),([\d.]+)', token)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    coords.append([x, y])
            
            if coords:
                tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
                tensor.requires_grad = True
                path_tensors.append(tensor)
        
        return path_tensors
    
    def render_paths_differentiable(
        self, 
        path_tensors: List[torch.Tensor],
        stroke_width: float = 0.06
    ) -> torch.Tensor:
        """
        Differentiable rendering of paths to raster image.
        
        Simplified approach: For each path, draw lines between consecutive points
        using anti-aliased line rendering.
        
        Args:
            path_tensors: List of path control points
            stroke_width: Stroke width in normalized coordinates
            
        Returns:
            Rendered image tensor, shape (H, W), values in [0, 1]
        """
        # Create blank canvas
        canvas = torch.zeros(
            (self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.device
        )
        
        # Convert stroke width to pixels
        stroke_px = int(stroke_width * self.image_size)
        
        for path in path_tensors:
            # Scale coordinates to image space
            scaled = path * self.image_size
            
            # Draw lines between consecutive points
            for i in range(len(scaled) - 1):
                p1 = scaled[i]
                p2 = scaled[i + 1]
                
                # Simple line rendering: mark pixels along the line
                canvas = self._draw_line_differentiable(canvas, p1, p2, stroke_px)
        
        # Invert (0=black stroke, 1=white background)
        return 1.0 - canvas
    
    def _draw_line_differentiable(
        self,
        canvas: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        width: int = 2
    ) -> torch.Tensor:
        """
        Draw a line on canvas (differentiable).
        
        Uses distance field approach: for each pixel, compute distance to line segment,
        apply smooth threshold.
        
        Args:
            canvas: Current canvas (H, W)
            p1, p2: Line endpoints, shape (2,)
            width: Line width in pixels
            
        Returns:
            Updated canvas
        """
        H, W = canvas.shape
        
        # Create coordinate grid
        y = torch.arange(H, dtype=torch.float32, device=self.device)
        x = torch.arange(W, dtype=torch.float32, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Compute distance from each pixel to line segment
        # Vector from p1 to p2
        v = p2 - p1
        v_len = torch.norm(v) + 1e-6
        v_unit = v / v_len
        
        # Vector from p1 to each pixel
        px = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        w = px - p1.unsqueeze(0).unsqueeze(0)  # (H, W, 2)
        
        # Project onto line segment
        t = torch.clamp(
            torch.sum(w * v_unit.unsqueeze(0).unsqueeze(0), dim=-1) / v_len,
            0, 1
        )  # (H, W)
        
        # Closest point on segment
        closest = p1.unsqueeze(0).unsqueeze(0) + t.unsqueeze(-1) * v.unsqueeze(0).unsqueeze(0)
        
        # Distance to closest point
        dist = torch.norm(px - closest, dim=-1)
        
        # Smooth threshold: dist < width/2 → 1, else → 0
        # Use smooth sigmoid
        intensity = torch.sigmoid((width/2 - dist) * 2)
        
        # Accumulate (max to handle overlaps)
        canvas = torch.maximum(canvas, intensity)
        
        return canvas
    
    def compute_losses(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        path_tensors: List[torch.Tensor],
        lambda_complexity: float = 0.01
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute optimization losses.
        
        Args:
            rendered: Rendered SVG, shape (H, W), values in [0, 1]
            target: Target raster, shape (H, W), values in [0, 1]
            path_tensors: Path control points
            lambda_complexity: Weight for complexity penalty
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # L2 reconstruction loss
        l2_loss = F.mse_loss(rendered, target)
        
        # Complexity penalty: number of control points
        total_points = sum(len(p) for p in path_tensors)
        complexity_loss = lambda_complexity * total_points
        
        # Stroke coverage loss: ensure strokes are on dark regions of target
        # Where target is dark (near 0), rendered should also be dark
        dark_target = (1 - target) > 0.5  # Binary mask of dark regions
        coverage_loss = F.mse_loss(
            rendered[dark_target],
            target[dark_target]
        ) if dark_target.any() else torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = l2_loss + complexity_loss + 0.5 * coverage_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'l2': l2_loss.item(),
            'complexity': complexity_loss,
            'coverage': coverage_loss.item()
        }
        
        return total_loss, loss_dict
    
    def optimize(
        self,
        svg_path: Path,
        target_image_path: Path,
        output_svg_path: Path,
        num_steps: int = 300,
        lr: float = 0.001,
        lambda_complexity: float = 0.01,
        verbose: bool = True
    ):
        """
        Optimize SVG to match target raster.
        
        Args:
            svg_path: Input SVG (initialized)
            target_image_path: Target raster PNG
            output_svg_path: Where to save optimized SVG
            num_steps: Number of optimization steps
            lr: Learning rate
            lambda_complexity: Weight for complexity penalty
            verbose: Print progress
        """
        # Load target image
        target_img = Image.open(target_image_path).convert('L')
        target_np = np.array(target_img) / 255.0  # Normalize to [0, 1]
        target = torch.tensor(target_np, dtype=torch.float32, device=self.device)
        
        # Parse SVG to tensors
        path_tensors = self.svg_to_tensors(svg_path)
        
        if not path_tensors:
            raise ValueError(f"No paths found in {svg_path}")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(path_tensors, lr=lr)
        
        if verbose:
            print(f"Optimizing {len(path_tensors)} paths with {sum(len(p) for p in path_tensors)} control points")
            print(f"Target: {target_image_path.name}")
        
        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Render current SVG
            rendered = self.render_paths_differentiable(path_tensors)
            
            # Compute losses
            loss, loss_dict = self.compute_losses(
                rendered, target, path_tensors, lambda_complexity
            )
            
            # Backward pass
            loss.backward()
            
            # Update
            optimizer.step()
            
            # Clamp coordinates to [0, 1]
            with torch.no_grad():
                for p in path_tensors:
                    p.clamp_(0, 1)
            
            # Log progress
            if verbose and (step % 50 == 0 or step == num_steps - 1):
                print(f"Step {step:3d} | Loss: {loss_dict['total']:.4f} "
                      f"(L2: {loss_dict['l2']:.4f}, "
                      f"Complexity: {loss_dict['complexity']:.0f}, "
                      f"Coverage: {loss_dict['coverage']:.4f})")
        
        # Export optimized SVG
        self._export_svg(path_tensors, output_svg_path)
        
        if verbose:
            print(f"✅ Optimized SVG saved to: {output_svg_path}")
    
    def _export_svg(self, path_tensors: List[torch.Tensor], output_path: Path, stroke_width: float = 0.06):
        """Export optimized paths to SVG file."""
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('viewBox', '0 0 1 1')
        svg.set('width', '256')
        svg.set('height', '256')
        
        for path_tensor in path_tensors:
            # Convert to numpy
            coords = path_tensor.detach().cpu().numpy()
            
            # Build path string
            d_parts = [f"M {coords[0][0]:.6f},{coords[0][1]:.6f}"]
            for i in range(1, len(coords)):
                d_parts.append(f"L {coords[i][0]:.6f},{coords[i][1]:.6f}")
            
            d = ' '.join(d_parts)
            
            # Create path element
            path = ET.SubElement(svg, 'path')
            path.set('d', d)
            path.set('stroke', 'black')
            path.set('stroke-width', str(stroke_width))
            path.set('fill', 'none')
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(ET.tostring(svg, encoding='unicode', method='xml'))


def test_optimization():
    """Test optimization on a sample."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RASTER_PERFECT, STROKE_WIDTH
    
    # Get test raster
    test_raster = list(RASTER_PERFECT.glob('*.png'))[0]
    print(f"Test raster: {test_raster.name}")
    
    # Input SVG (from initialization)
    input_svg = Path('outputs/test_initialization.svg')
    if not input_svg.exists():
        print("ERROR: Run vectorizers/initialize.py first to create test_initialization.svg")
        return
    
    # Output
    output_svg = Path('outputs/test_optimized.svg')
    
    # Optimize
    vectorizer = DifferentiableVectorizer(image_size=256, device='cpu')
    vectorizer.optimize(
        svg_path=input_svg,
        target_image_path=test_raster,
        output_svg_path=output_svg,
        num_steps=300,
        lr=0.001,
        lambda_complexity=0.001,
        verbose=True
    )


if __name__ == '__main__':
    test_optimization()
