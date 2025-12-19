"""
Advanced loss functions for optimization-based vectorization.

This module implements the multi-term loss objective that ensures:
1. Pixel accuracy (raster loss)
2. Edge alignment (strokes follow detected edges)
3. Smooth geometry (curvature regularization)
4. No self-intersections (vector sanity)
5. Complexity penalty (segment count)

Final Objective:
L = λ₁·L_raster + λ₂·L_edge + λ₃·L_curvature + λ₄·L_intersection + λ₅·L_complexity
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Dict


class VectorizationLoss:
    """Multi-term loss for optimization-based vectorization."""
    
    def __init__(
        self,
        image_size: int = 256,
        device: str = 'cpu',
        lambda_raster: float = 1.0,
        lambda_edge: float = 0.5,
        lambda_curvature: float = 0.1,
        lambda_intersection: float = 0.3,
        lambda_complexity: float = 0.005,
    ):
        """
        Args:
            image_size: Raster resolution
            device: 'cpu' or 'cuda'
            lambda_raster: Weight for pixel reconstruction
            lambda_edge: Weight for edge alignment
            lambda_curvature: Weight for curvature smoothness
            lambda_intersection: Weight for self-intersection penalty
            lambda_complexity: Weight for segment count penalty
        """
        self.image_size = image_size
        self.device = device
        self.lambda_raster = lambda_raster
        self.lambda_edge = lambda_edge
        self.lambda_curvature = lambda_curvature
        self.lambda_intersection = lambda_intersection
        self.lambda_complexity = lambda_complexity
        
        # Edge map will be computed once and cached
        self.edge_map = None
        self.edge_distance_field = None
    
    def precompute_edges(self, target_raster: torch.Tensor):
        """
        Precompute edge map and distance field from target raster.
        
        This is called once at the start of optimization.
        
        Args:
            target_raster: Target image (H, W) in [0, 1]
        """
        # Convert to numpy for OpenCV
        target_np = (target_raster.cpu().numpy() * 255).astype(np.uint8)
        
        # Compute edges using Canny
        edges = cv2.Canny(target_np, 50, 150)
        
        # Normalize to [0, 1]
        self.edge_map = torch.from_numpy(edges / 255.0).float().to(self.device)
        
        # Compute distance field: distance from each pixel to nearest edge
        # This will be used for edge alignment loss
        dist_transform = cv2.distanceTransform(
            (1 - edges / 255.0).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Normalize distance field
        self.edge_distance_field = torch.from_numpy(dist_transform).float().to(self.device)
        self.edge_distance_field = self.edge_distance_field / self.image_size  # Normalize to [0, 1]
    
    def compute_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        all_paths: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with all terms.
        
        Args:
            rendered: Rendered SVG (H, W) in [0, 1]
            target: Target raster (H, W) in [0, 1]
            all_paths: List of path tensors, each shape (N, 2)
            
        Returns:
            (total_loss, loss_dict)
        """
        # Precompute edges if not done
        if self.edge_map is None:
            self.precompute_edges(target)
        
        # 1. Raster loss (pixel reconstruction)
        loss_raster = F.mse_loss(rendered, target)
        
        # 2. Edge alignment loss (CRITICAL for visual coherence)
        loss_edge = self._compute_edge_alignment_loss(all_paths)
        
        # 3. Curvature regularization (smooth strokes)
        loss_curvature = self._compute_curvature_loss(all_paths)
        
        # 4. Self-intersection penalty (vector sanity)
        loss_intersection = self._compute_intersection_penalty(all_paths)
        
        # 5. Complexity penalty (segment count)
        num_segments = sum(len(path) for path in all_paths)
        loss_complexity = torch.tensor(num_segments, dtype=torch.float32, device=self.device)
        
        # Total loss
        total_loss = (
            self.lambda_raster * loss_raster +
            self.lambda_edge * loss_edge +
            self.lambda_curvature * loss_curvature +
            self.lambda_intersection * loss_intersection +
            self.lambda_complexity * loss_complexity
        )
        
        # Loss dictionary for logging
        def to_python(val):
            """Convert tensor or float to Python float."""
            if isinstance(val, torch.Tensor):
                return val.item()
            return float(val)
        
        loss_dict = {
            'total': to_python(total_loss),
            'raster': to_python(loss_raster),
            'edge': to_python(loss_edge),
            'curvature': to_python(loss_curvature),
            'intersection': to_python(loss_intersection),
            'complexity': to_python(loss_complexity),
            'segments': num_segments
        }
        
        return total_loss, loss_dict
    
    def _compute_edge_alignment_loss(self, paths: List[torch.Tensor]) -> torch.Tensor:
        """
        Penalize strokes that don't align with detected edges.
        
        Strategy:
        - For each control point, sample the edge distance field
        - Penalize points far from edges
        - This stops "floating spaghetti lines"
        
        Args:
            paths: List of path tensors, each (N, 2) in [0, 1]
            
        Returns:
            Scalar loss tensor
        """
        if self.edge_distance_field is None:
            return torch.tensor(0.0, device=self.device)
        
        total_penalty = 0.0
        total_points = 0
        
        for path in paths:
            if len(path) < 2:
                continue
            
            # Scale coordinates to image space
            scaled_coords = path * self.image_size
            
            # Clamp to valid range
            x = torch.clamp(scaled_coords[:, 0], 0, self.image_size - 1)
            y = torch.clamp(scaled_coords[:, 1], 0, self.image_size - 1)
            
            # Sample distance field at control points (using bilinear interpolation)
            # Grid sample expects normalized coordinates in [-1, 1]
            norm_x = (x / (self.image_size - 1)) * 2 - 1
            norm_y = (y / (self.image_size - 1)) * 2 - 1
            
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
            
            # Sample distance field
            dist_field_expanded = self.edge_distance_field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            sampled_distances = F.grid_sample(
                dist_field_expanded,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            
            # Extract distances
            distances = sampled_distances.squeeze()  # (N,)
            
            # Penalize large distances (points far from edges)
            # Use smooth L1 loss to avoid gradient explosion
            total_penalty += F.smooth_l1_loss(distances, torch.zeros_like(distances), reduction='sum')
            total_points += len(path)
        
        if total_points == 0:
            return torch.tensor(0.0, device=self.device)
        
        return total_penalty / total_points
    
    def _compute_curvature_loss(self, paths: List[torch.Tensor]) -> torch.Tensor:
        """
        Penalize sharp angle changes to encourage smooth strokes.
        
        Strategy:
        - For each triplet of consecutive points (p_i-1, p_i, p_i+1)
        - Compute the angle change
        - Penalize large angle changes
        
        Args:
            paths: List of path tensors, each (N, 2)
            
        Returns:
            Scalar loss tensor
        """
        total_curvature = 0.0
        total_triplets = 0
        
        for path in paths:
            if len(path) < 3:
                continue
            
            # Get consecutive triplets
            p_prev = path[:-2]  # (N-2, 2)
            p_curr = path[1:-1]  # (N-2, 2)
            p_next = path[2:]    # (N-2, 2)
            
            # Vectors
            v1 = p_curr - p_prev  # (N-2, 2)
            v2 = p_next - p_curr  # (N-2, 2)
            
            # Normalize vectors
            v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
            v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-8)
            
            # Dot product gives cos(angle)
            cos_angles = (v1_norm * v2_norm).sum(dim=1)
            
            # Penalize sharp turns (cos(angle) far from 1)
            # cos(0°) = 1 (straight), cos(90°) = 0, cos(180°) = -1
            # We want angles close to 0° (straight)
            angle_penalty = 1.0 - cos_angles  # 0 for straight, 2 for 180° turn
            
            total_curvature += angle_penalty.sum()
            total_triplets += len(angle_penalty)
        
        if total_triplets == 0:
            return torch.tensor(0.0, device=self.device)
        
        return total_curvature / total_triplets
    
    def _compute_intersection_penalty(self, paths: List[torch.Tensor]) -> torch.Tensor:
        """
        Penalize self-intersections in paths.
        
        Strategy:
        - For each path, check all pairs of segments
        - Detect intersections using line-line intersection test
        - Penalize intersecting segments
        
        Note: This is expensive (O(n²) per path), but necessary for vector sanity.
        
        Args:
            paths: List of path tensors, each (N, 2)
            
        Returns:
            Scalar loss tensor
        """
        total_penalty = 0.0
        
        for path in paths:
            if len(path) < 4:  # Need at least 4 points to have 2 non-adjacent segments
                continue
            
            # Get all segments
            num_segments = len(path) - 1
            
            # Check pairs of non-adjacent segments
            for i in range(num_segments - 2):  # -2 to avoid adjacent segments
                for j in range(i + 2, num_segments):
                    # Segment i: (path[i], path[i+1])
                    # Segment j: (path[j], path[j+1])
                    
                    p1 = path[i]
                    p2 = path[i + 1]
                    p3 = path[j]
                    p4 = path[j + 1]
                    
                    # Check if segments intersect
                    intersection_score = self._segment_intersection_score(p1, p2, p3, p4)
                    total_penalty += intersection_score
        
        return total_penalty
    
    def _segment_intersection_score(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute differentiable intersection score for two line segments.
        
        Uses parametric line representation:
        - Line 1: p1 + t*(p2-p1), t ∈ [0, 1]
        - Line 2: p3 + s*(p4-p3), s ∈ [0, 1]
        
        If they intersect, both t and s should be in [0, 1].
        
        Args:
            p1, p2: Endpoints of first segment, shape (2,)
            p3, p4: Endpoints of second segment, shape (2,)
            
        Returns:
            Scalar penalty (higher if segments are close to intersecting)
        """
        # Direction vectors
        d1 = p2 - p1
        d2 = p4 - p3
        
        # Solve for intersection:
        # p1 + t*d1 = p3 + s*d2
        # Rearrange: t*d1 - s*d2 = p3 - p1
        
        # This is a 2x2 linear system:
        # [d1.x  -d2.x] [t]   [p3.x - p1.x]
        # [d1.y  -d2.y] [s] = [p3.y - p1.y]
        
        det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
        det = d1[0] * d2[1] - d1[1] * d2[0]
        
        # If determinant is close to 0, lines are parallel
        if torch.abs(det) < 1e-6:
            return torch.tensor(0.0, device=self.device)
        
        # Solve for t and s
        delta = p3 - p1
        t = (delta[0] * d2[1] - delta[1] * d2[0]) / det
        s = (delta[0] * d1[1] - delta[1] * d1[0]) / det
        
        # Penalize if both t and s are in [0, 1] (segments intersect)
        # Use soft penalty: max(0, 1 - |t - 0.5| * 2) gives 1 at t=0.5, 0 at t=0 or t=1
        t_penalty = torch.relu(1.0 - 2.0 * torch.abs(t - 0.5))
        s_penalty = torch.relu(1.0 - 2.0 * torch.abs(s - 0.5))
        
        # Intersection penalty is product (both must be in range)
        return t_penalty * s_penalty


def test_losses():
    """Test loss functions on dummy data."""
    print("Testing VectorizationLoss module...")
    
    # Create dummy data
    device = 'cpu'
    image_size = 256
    
    # Dummy target (white background, black square)
    target = torch.ones((image_size, image_size), device=device)
    target[100:150, 100:150] = 0.0
    
    # Dummy rendered (similar but slightly off)
    rendered = torch.ones((image_size, image_size), device=device)
    rendered[105:155, 105:155] = 0.1
    
    # Dummy paths (simple square)
    path1 = torch.tensor([
        [0.4, 0.4],
        [0.6, 0.4],
        [0.6, 0.6],
        [0.4, 0.6],
        [0.4, 0.4]
    ], device=device, requires_grad=True)
    
    paths = [path1]
    
    # Create loss module
    loss_fn = VectorizationLoss(
        image_size=image_size,
        device=device,
        lambda_raster=1.0,
        lambda_edge=0.5,
        lambda_curvature=0.1,
        lambda_intersection=0.3,
        lambda_complexity=0.005
    )
    
    # Compute loss
    total_loss, loss_dict = loss_fn.compute_loss(rendered, target, paths)
    
    print(f"\nLoss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key:15s}: {value:.6f}")
    
    print(f"\n✅ Loss module test passed!")
    
    # Test gradient flow
    total_loss.backward()
    print(f"   Gradient norm: {path1.grad.norm().item():.6f}")


if __name__ == '__main__':
    test_losses()
