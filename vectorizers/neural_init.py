"""
Neural Initialization for Vectorization

Architecture: CNN predicts initial control points from raster input

Strategy:
1. Encoder: Raster image (256x256) → latent features
2. Decoder: Latent features → control point coordinates
3. Training: Supervised on (raster, ground_truth_paths) pairs
4. Integration: Replace edge detection, keep optimization

This is cleaner than end-to-end because:
- ML handles initialization (fast, learned features)
- Optimization handles refinement (guarantees quality)
- Hybrid approach gets best of both worlds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class RasterEncoder(nn.Module):
    """
    Encode raster image to latent representation.
    
    Architecture: ResNet-style with skip connections
    Input: (B, 1, 256, 256)
    Output: (B, 512)
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 128x128
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 64x64
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 32x32
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 16x16
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # 8x8
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Latent projection
        self.fc = nn.Linear(512, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 256, 256)
            
        Returns:
            latent: (B, latent_dim)
        """
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Pool and project
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (B, 512)
        x = self.fc(x)  # (B, latent_dim)
        
        return x


class ControlPointDecoder(nn.Module):
    """
    Decode latent representation to control points.
    
    Two strategies:
    1. Fixed-size output: Predict N control points (padded/masked)
    2. Autoregressive: Predict one point at a time (transformer-style)
    
    We use Strategy 1 for simplicity.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        max_points: int = 200,  # Maximum control points per path
        max_paths: int = 10      # Maximum paths per image
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_points = max_points
        self.max_paths = max_paths
        
        # Path-level decoder: predict per-path parameters
        self.path_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, max_paths * 3)  # (num_points, start_x, start_y) per path
        )
        
        # Point-level decoder: predict control points for each path
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim + 3, 256),  # latent + path_params
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, max_points * 2)  # (x, y) for each point
        )
    
    def forward(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, latent_dim)
            
        Returns:
            points: (B, max_paths, max_points, 2) - control point coordinates [0, 1]
            masks: (B, max_paths, max_points) - binary masks (1=valid, 0=padding)
        """
        B = latent.size(0)
        
        # Decode path-level parameters
        path_params = self.path_decoder(latent)  # (B, max_paths * 3)
        path_params = path_params.view(B, self.max_paths, 3)
        
        # For each path, decode control points
        all_points = []
        
        for i in range(self.max_paths):
            # Concatenate latent + path_params for this path
            path_latent = torch.cat([latent, path_params[:, i]], dim=1)  # (B, latent_dim + 3)
            
            # Decode points
            points = self.point_decoder(path_latent)  # (B, max_points * 2)
            points = points.view(B, self.max_points, 2)
            
            # Apply sigmoid to ensure [0, 1] range
            points = torch.sigmoid(points)
            
            all_points.append(points)
        
        # Stack: (B, max_paths, max_points, 2)
        all_points = torch.stack(all_points, dim=1)
        
        # Generate masks based on path_params
        # First element of path_params encodes number of points
        num_points = torch.sigmoid(path_params[:, :, 0]) * self.max_points  # (B, max_paths)
        
        # Create masks
        masks = torch.zeros(B, self.max_paths, self.max_points, device=latent.device)
        for b in range(B):
            for p in range(self.max_paths):
                n = int(num_points[b, p].item())
                if n > 0:
                    masks[b, p, :n] = 1.0
        
        return all_points, masks


class NeuralInitializer(nn.Module):
    """
    Complete neural initialization model.
    
    Input: Raster image (256x256 grayscale)
    Output: Initial control points for SVG paths
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        max_points: int = 200,
        max_paths: int = 10
    ):
        super().__init__()
        
        self.encoder = RasterEncoder(latent_dim)
        self.decoder = ControlPointDecoder(latent_dim, max_points, max_paths)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, 256, 256) - raster images
            
        Returns:
            points: (B, max_paths, max_points, 2)
            masks: (B, max_paths, max_points)
        """
        latent = self.encoder(x)
        points, masks = self.decoder(latent)
        return points, masks
    
    def predict_svg_paths(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> List[List[torch.Tensor]]:
        """
        Predict SVG paths from raster image.
        
        Args:
            x: (B, 1, 256, 256)
            threshold: Mask threshold for valid points
            
        Returns:
            List of path lists, one per batch element
            Each path is a tensor of shape (N, 2)
        """
        self.eval()
        with torch.no_grad():
            points, masks = self.forward(x)  # (B, paths, points, 2), (B, paths, points)
        
        batch_paths = []
        
        for b in range(points.size(0)):
            sample_paths = []
            
            for p in range(points.size(1)):
                # Extract valid points for this path
                mask = masks[b, p] > threshold
                valid_points = points[b, p][mask]  # (N, 2)
                
                if len(valid_points) > 1:  # At least 2 points for a path
                    sample_paths.append(valid_points)
            
            batch_paths.append(sample_paths)
        
        return batch_paths


def create_training_dataset():
    """
    Create training dataset from ground truth SVGs.
    
    Format:
    - Input: Rasterized perfect images (256x256 grayscale)
    - Target: Control points extracted from clean SVGs
    
    Dataset structure:
    {
        'raster': (N, 1, 256, 256),
        'paths': List[List[np.ndarray]],  # List of paths per sample
        'masks': (N, max_paths, max_points)  # Valid point masks
    }
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config import RASTER_PERFECT, SVG_CLEAN, TRAIN_IDS
    from PIL import Image
    import numpy as np
    import xml.etree.ElementTree as ET
    import re
    
    print("Creating neural initialization training dataset...")
    
    # Load training IDs
    train_ids = TRAIN_IDS.read_text().strip().split('\n')
    
    print(f"Processing {len(train_ids)} training samples...")
    
    dataset = {
        'rasters': [],
        'control_points': [],
        'svg_ids': []
    }
    
    for svg_id in train_ids[:100]:  # Start with 100 samples for testing
        # Load raster
        raster_path = RASTER_PERFECT / f"{svg_id}_base.png"
        if not raster_path.exists():
            continue
        
        raster = np.array(Image.open(raster_path).convert('L'))
        raster = raster / 255.0  # Normalize
        
        # Load SVG and extract control points
        svg_path = SVG_CLEAN / f"{svg_id}.svg"
        if not svg_path.exists():
            continue
        
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            paths = []
            for elem in root.iter():
                if 'path' not in str(elem.tag).lower():
                    continue
                
                d = elem.get('d', '')
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
                
                if len(coords) < 4:
                    continue
                
                coords = [float(c) for c in coords]
                points = np.array(coords).reshape(-1, 2)
                
                paths.append(points)
            
            if paths:
                dataset['rasters'].append(raster)
                dataset['control_points'].append(paths)
                dataset['svg_ids'].append(svg_id)
        
        except Exception as e:
            print(f"Error processing {svg_id}: {e}")
            continue
    
    print(f"\n✅ Created dataset with {len(dataset['rasters'])} samples")
    print(f"   Average paths per sample: {np.mean([len(p) for p in dataset['control_points']]):.1f}")
    
    return dataset


def train_neural_initializer(
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4
):
    """
    Train the neural initializer.
    
    Loss: MSE between predicted and ground truth control points
    """
    # Create dataset
    dataset = create_training_dataset()
    
    # Initialize model
    model = NeuralInitializer(
        latent_dim=512,
        max_points=200,
        max_paths=10
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining neural initializer for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Training loop placeholder
    # (Full implementation would include data loader, batching, validation, etc.)
    
    print("\n⚠️  Full training implementation coming in next phase")
    print("    For now, model architecture is defined and ready")


if __name__ == '__main__':
    print("Neural Initialization Architecture")
    print("=" * 60)
    
    # Test model creation
    model = NeuralInitializer(latent_dim=512, max_points=200, max_paths=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 256, 256)
    points, masks = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output points shape: {points.shape}")
    print(f"  Output masks shape: {masks.shape}")
    
    print(f"\n✅ Neural initialization architecture ready")
    print(f"   Next: Create training dataset and train")
