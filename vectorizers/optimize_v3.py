"""
Optimization-based vectorizer with advanced multi-term loss.

This replaces optimize_v2.py with the new loss formulation:
L = λ₁·L_raster + λ₂·L_edge + λ₃·L_curvature + λ₄·L_intersection + λ₅·L_complexity

Key improvements:
1. Edge alignment loss → stops "floating spaghetti lines"
2. Curvature regularization → encourages smooth strokes
3. Self-intersection penalty → enforces vector sanity
4. Complexity penalty → keeps segment count low

This is the FINAL optimizer before neural methods.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
from tqdm import tqdm
import sys

# Import base optimizer and new losses
sys.path.append(str(Path(__file__).parent))
from optimize import DifferentiableVectorizer
from losses import VectorizationLoss


class AdvancedVectorizer(DifferentiableVectorizer):
    """Optimizer with multi-term loss for visual coherence."""
    
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
            lambda_edge: Weight for edge alignment (CRITICAL)
            lambda_curvature: Weight for curvature smoothness
            lambda_intersection: Weight for self-intersection penalty
            lambda_complexity: Weight for segment count penalty
        """
        super().__init__(image_size, device)
        
        # Create loss module
        self.loss_fn = VectorizationLoss(
            image_size=image_size,
            device=device,
            lambda_raster=lambda_raster,
            lambda_edge=lambda_edge,
            lambda_curvature=lambda_curvature,
            lambda_intersection=lambda_intersection,
            lambda_complexity=lambda_complexity
        )
    
    def optimize(
        self,
        svg_path: Path,
        target_image: np.ndarray,
        output_path: Optional[Path] = None,
        num_steps: int = 150,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> str:
        """
        Optimize SVG to match target with multi-term loss.
        
        Args:
            svg_path: Initial SVG (from edge initialization)
            target_image: Target raster (H, W) in [0, 1]
            output_path: Where to save optimized SVG
            num_steps: Number of optimization steps
            learning_rate: Adam learning rate
            verbose: Print progress
            
        Returns:
            Optimized SVG string
        """
        # Load initial paths
        paths = self.svg_to_tensors(svg_path)
        
        if not paths or len(paths) == 0:
            raise ValueError(f"No paths found in SVG: {svg_path}")
        
        paths = [p.requires_grad_(True) for p in paths]
        
        # Target as tensor
        target_tensor = torch.from_numpy(target_image).float().to(self.device)
        
        # Precompute edges for edge alignment loss
        self.loss_fn.precompute_edges(target_tensor)
        
        # Optimizer
        optimizer = torch.optim.Adam(paths, lr=learning_rate)
        
        # Track best
        best_loss = float('inf')
        best_svg = None
        
        # Optimization loop
        pbar = tqdm(range(num_steps), desc="Optimizing", disable=not verbose)
        for step in pbar:
            optimizer.zero_grad()
            
            # Render current state
            rendered = self.render_paths_differentiable(paths)
            
            # Compute multi-term loss
            loss, loss_dict = self.loss_fn.compute_loss(rendered, target_tensor, paths)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            # Clamp coordinates to [0, 1]
            with torch.no_grad():
                for path in paths:
                    path.clamp_(0.0, 1.0)
            
            # Update progress bar
            if step % 10 == 0:
                pbar.set_postfix({
                    'L2': f"{loss_dict['raster']:.4f}",
                    'edge': f"{loss_dict['edge']:.3f}",
                    'curv': f"{loss_dict['curvature']:.3f}",
                    'segs': loss_dict['segments']
                })
            
            # Track best
            if isinstance(loss, torch.Tensor):
                loss_value = loss.item()
            else:
                loss_value = float(loss)
                
            if loss_value < best_loss:
                best_loss = loss_value
                # Just track loss, we'll export at the end
        
        # Export final SVG
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._export_svg(paths, output_path)
        
        if verbose:
            print(f"\n✅ Optimization complete:")
            print(f"   Final L2: {loss_dict['raster']:.4f}")
            print(f"   Edge alignment: {loss_dict['edge']:.4f}")
            print(f"   Curvature: {loss_dict['curvature']:.4f}")
            print(f"   Intersections: {loss_dict['intersection']:.4f}")
            print(f"   Segments: {loss_dict['segments']}")
        
        return str(output_path) if output_path else ""


def test_advanced_optimizer():
    """Test advanced optimizer on a sample."""
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RASTER_PERFECT, TEST_IDS, BASELINES
    
    # Get a test sample
    test_ids = TEST_IDS.read_text().strip().split('\n')
    test_id = test_ids[0]
    
    print(f"Testing advanced optimizer on: {test_id}")
    
    # Paths
    raster_path = RASTER_PERFECT / f"{test_id}_base.png"
    # Use existing optimized SVG as initialization
    init_svg_path = BASELINES / 'optimization_full' / f"{test_id}.svg"
    output_svg_path = Path('outputs/test_advanced_optimization.svg')
    
    # Load target
    target = np.array(Image.open(raster_path).convert('L')) / 255.0
    
    # Create advanced vectorizer
    vectorizer = AdvancedVectorizer(
        image_size=256,
        lambda_raster=1.0,
        lambda_edge=0.5,        # CRITICAL for visual coherence
        lambda_curvature=0.1,   # Smooth strokes
        lambda_intersection=0.3,# Vector sanity
        lambda_complexity=0.005 # Segment count
    )
    
    # Optimize
    svg_string = vectorizer.optimize(
        init_svg_path,
        target,
        output_svg_path,
        num_steps=150,
        learning_rate=0.01,
        verbose=True
    )
    
    print(f"\n✅ Advanced optimization saved to: {output_svg_path}")


if __name__ == '__main__':
    test_advanced_optimizer()
