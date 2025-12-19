"""
Enhanced optimization with aggressive segment reduction.

Key improvements over optimize.py:
1. Stronger complexity penalty (adaptive)
2. Total stroke length penalty
3. Iterative prune-and-optimize
4. Target: <150 segments with L2 ≤ 0.24
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import xml.etree.ElementTree as ET
import re
from PIL import Image
from tqdm import tqdm

# Import base optimizer
import sys
sys.path.append(str(Path(__file__).parent))
from optimize import DifferentiableVectorizer


class EnhancedVectorizer(DifferentiableVectorizer):
    """Enhanced optimizer with stronger segment reduction."""
    
    def __init__(
        self,
        image_size: int = 256,
        device: str = 'cpu',
        lambda_complexity: float = 0.005,  # 5x stronger than before
        lambda_length: float = 0.001,       # Penalize total path length
        prune_iterations: int = 2,          # Number of prune-optimize cycles
    ):
        """
        Args:
            image_size: Raster resolution
            device: 'cpu' or 'cuda'
            lambda_complexity: Segment count penalty (stronger)
            lambda_length: Total stroke length penalty (new)
            prune_iterations: How many prune→optimize cycles
        """
        super().__init__(image_size, device)
        self.lambda_complexity = lambda_complexity
        self.lambda_length = lambda_length
        self.prune_iterations = prune_iterations
    
    def compute_path_length(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Compute total path length (sum of segment lengths).
        
        Args:
            control_points: Shape (N, 2)
            
        Returns:
            Scalar tensor: total length
        """
        if len(control_points) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute distances between consecutive points
        diffs = control_points[1:] - control_points[:-1]
        lengths = torch.norm(diffs, dim=1)
        
        return lengths.sum()
    
    def compute_enhanced_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        all_paths: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Enhanced loss with stronger regularization.
        
        Args:
            rendered: Rendered SVG (H, W)
            target: Target raster (H, W)
            all_paths: List of path tensors
            
        Returns:
            total_loss, loss_dict
        """
        # 1. Reconstruction loss (L2)
        loss_recon = F.mse_loss(rendered, target)
        
        # 2. Complexity penalty (number of segments)
        num_segments = sum(len(path) for path in all_paths)
        loss_complexity = torch.tensor(
            self.lambda_complexity * num_segments,
            device=self.device
        )
        
        # 3. Total stroke length penalty
        total_length = sum(self.compute_path_length(path) for path in all_paths)
        loss_length = self.lambda_length * total_length
        
        # 4. Coverage loss (ensure strokes are used)
        loss_coverage = torch.relu(0.5 - rendered.mean()) * 10.0
        
        # Total loss
        total_loss = loss_recon + loss_complexity + loss_length + loss_coverage
        
        return total_loss, {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
            'complexity': loss_complexity.item(),
            'length': loss_length.item(),
            'coverage': loss_coverage.item()
        }
    
    def optimize_with_pruning(
        self,
        svg_path: Path,
        target_image: np.ndarray,
        output_path: Optional[Path] = None,
        num_steps_per_round: int = 100,
        verbose: bool = True
    ) -> str:
        """
        Iterative prune-and-optimize strategy.
        
        Strategy:
        1. Optimize for N steps
        2. Prune tiny/redundant segments
        3. Re-optimize for N more steps
        4. Repeat
        
        Args:
            svg_path: Initial SVG
            target_image: Target raster (H, W) in [0, 1]
            output_path: Where to save optimized SVG
            num_steps_per_round: Steps per prune-optimize cycle
            verbose: Print progress
            
        Returns:
            Optimized SVG string
        """
        # Load initial paths
        paths = self.svg_to_tensors(svg_path)
        paths = [p.requires_grad_(True) for p in paths]
        
        # Target as tensor
        target_tensor = torch.from_numpy(target_image).float().to(self.device)
        
        best_loss = float('inf')
        best_svg = None
        
        for iteration in range(self.prune_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Prune-Optimize Iteration {iteration+1}/{self.prune_iterations}")
                print(f"Current paths: {len(paths)}, segments: {sum(len(p) for p in paths)}")
                print(f"{'='*60}")
            
            # Optimize current paths
            optimizer = torch.optim.Adam(paths, lr=0.01)
            
            pbar = tqdm(range(num_steps_per_round), desc=f"Optimizing Round {iteration+1}")
            for step in pbar:
                optimizer.zero_grad()
                
                # Render current state
                rendered = self.render_paths_differentiable(paths)
                
                # Compute loss
                loss, loss_dict = self.compute_enhanced_loss(rendered, target_tensor, paths)
                
                # Backprop
                loss.backward()
                optimizer.step()
                
                # Clamp to [0, 1]
                with torch.no_grad():
                    for path in paths:
                        path.clamp_(0.0, 1.0)
                
                if step % 20 == 0:
                    pbar.set_postfix({
                        'L2': f"{loss_dict['recon']:.4f}",
                        'segments': sum(len(p) for p in paths)
                    })
                
                # Track best
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_svg = self.tensors_to_svg([p.detach().cpu().numpy() for p in paths])
            
            # Prune after optimization (except last iteration)
            if iteration < self.prune_iterations - 1:
                paths = self._prune_paths(paths, target_tensor, verbose=verbose)
                paths = [p.requires_grad_(True) for p in paths]
        
        # Final SVG
        final_svg = self.tensors_to_svg([p.detach().cpu().numpy() for p in paths])
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(final_svg)
        
        if verbose:
            final_segments = sum(len(p) for p in paths)
            print(f"\n✅ Optimization complete:")
            print(f"   Final segments: {final_segments}")
            print(f"   Final L2: {loss_dict['recon']:.4f}")
            print(f"   Best loss: {best_loss:.4f}")
        
        return final_svg
    
    def _prune_paths(
        self,
        paths: List[torch.Tensor],
        target: torch.Tensor,
        min_length: float = 0.015,  # Minimum path length to keep
        verbose: bool = True
    ) -> List[torch.Tensor]:
        """
        Prune paths that contribute little to reconstruction.
        
        Strategy:
        1. Remove paths shorter than min_length
        2. Remove paths that don't reduce loss when removed
        
        Args:
            paths: List of path tensors
            target: Target image
            min_length: Minimum total path length
            verbose: Print pruning stats
            
        Returns:
            Pruned list of paths
        """
        initial_count = len(paths)
        initial_segments = sum(len(p) for p in paths)
        
        # Compute initial loss
        with torch.no_grad():
            initial_rendered = self.render_paths_differentiable(paths)
            initial_loss = F.mse_loss(initial_rendered, target).item()
        
        kept_paths = []
        
        for path in paths:
            # Check path length
            path_length = self.compute_path_length(path).item()
            
            if path_length < min_length:
                continue  # Too short, prune it
            
            # Test if removing this path hurts reconstruction
            test_paths = [p for p in paths if p is not path]
            
            with torch.no_grad():
                if len(test_paths) == 0:
                    # Keep at least one path
                    kept_paths.append(path)
                    continue
                
                test_rendered = self.render_paths_differentiable(test_paths)
                test_loss = F.mse_loss(test_rendered, target).item()
            
            # Keep if removing it increases loss significantly
            if test_loss > initial_loss + 0.001:  # Tolerance
                kept_paths.append(path)
        
        final_segments = sum(len(p) for p in kept_paths)
        
        if verbose:
            print(f"   Pruning: {initial_count}→{len(kept_paths)} paths, "
                  f"{initial_segments}→{final_segments} segments")
        
        return kept_paths if kept_paths else [paths[0]]  # Keep at least one


def test_enhanced_optimization():
    """Test enhanced optimizer on a sample."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RASTER_PERFECT, TEST_IDS
    
    # Get a test sample
    test_ids = TEST_IDS.read_text().strip().split('\n')
    test_id = test_ids[0]
    
    raster_path = RASTER_PERFECT / f"{test_id}_base.png"
    init_svg_path = Path('outputs/test_initialization.svg')
    output_svg_path = Path('outputs/test_enhanced_optimization.svg')
    
    print(f"Testing enhanced optimization on: {test_id}")
    
    # Load target
    target = np.array(Image.open(raster_path).convert('L')) / 255.0
    
    # Create enhanced vectorizer
    vectorizer = EnhancedVectorizer(
        image_size=256,
        lambda_complexity=0.005,  # Strong penalty
        lambda_length=0.001,
        prune_iterations=3        # 3 prune-optimize cycles
    )
    
    # Optimize with pruning
    svg_string = vectorizer.optimize_with_pruning(
        init_svg_path,
        target,
        output_svg_path,
        num_steps_per_round=100,
        verbose=True
    )
    
    print(f"\n✅ Enhanced optimization saved to: {output_svg_path}")


if __name__ == '__main__':
    test_enhanced_optimization()
