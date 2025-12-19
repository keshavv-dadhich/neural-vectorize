"""
Evaluation metrics for vectorization.

Metrics:
- L2 MSE: Pixel-level reconstruction error
- SSIM: Structural similarity
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_l2(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Compute L2 MSE between predicted and target images.
    
    Args:
        predicted: Predicted image (H, W) in [0, 1]
        target: Target image (H, W) in [0, 1]
        
    Returns:
        L2 MSE score
    """
    return float(np.mean((predicted - target) ** 2))


def compute_ssim(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Compute SSIM between predicted and target images.
    
    Args:
        predicted: Predicted image (H, W) in [0, 1]
        target: Target image (H, W) in [0, 1]
        
    Returns:
        SSIM score in [-1, 1], higher is better
    """
    return float(ssim(predicted, target, data_range=1.0))
