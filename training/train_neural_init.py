"""
Train neural initializer for SVG control points.

Architecture: ResNet-18 encoder → MLP decoder → Control points + masks
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional
import json


class SVGDataset(Dataset):
    """Dataset of (degraded raster, oracle control points)."""
    
    def __init__(self, data_path: Path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'raster': sample['raster'],
            'points': sample['points'],
            'masks': sample['masks']
        }


class RasterEncoder(nn.Module):
    """ResNet-18 encoder for raster images."""
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Load pretrained ResNet-18
        import torchvision.models as models
        resnet = models.resnet18(pretrained=False)
        
        # Modify first conv for grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Global pooling + projection
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        # x: (B, 1, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x  # (B, latent_dim)


class ControlPointDecoder(nn.Module):
    """MLP decoder for control points."""
    
    def __init__(
        self, 
        latent_dim: int = 512,
        max_paths: int = 10,
        max_points_per_path: int = 50
    ):
        super().__init__()
        
        self.max_paths = max_paths
        self.max_points_per_path = max_points_per_path
        
        # Total outputs: points (10*50*2) + masks (10*50)
        num_point_outputs = max_paths * max_points_per_path * 2
        num_mask_outputs = max_paths * max_points_per_path
        
        # MLP for points
        self.point_mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_point_outputs),
            nn.Sigmoid()  # Normalize to [0, 1] (will scale to [0, 256] later)
        )
        
        # MLP for masks
        self.mask_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_mask_outputs),
            nn.Sigmoid()  # Probability of point being valid
        )
    
    def forward(self, z):
        # z: (B, latent_dim)
        
        # Decode points
        points = self.point_mlp(z)  # (B, 10*50*2)
        points = points.view(-1, self.max_paths, self.max_points_per_path, 2)
        points = points * 256.0  # Scale to [0, 256]
        
        # Decode masks
        masks = self.mask_mlp(z)  # (B, 10*50)
        masks = masks.view(-1, self.max_paths, self.max_points_per_path)
        
        return points, masks


class NeuralInitializer(nn.Module):
    """Full neural initializer model."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        max_paths: int = 10,
        max_points_per_path: int = 50
    ):
        super().__init__()
        
        self.encoder = RasterEncoder(latent_dim)
        self.decoder = ControlPointDecoder(latent_dim, max_paths, max_points_per_path)
    
    def forward(self, raster):
        # raster: (B, 1, 256, 256)
        z = self.encoder(raster)
        points, masks = self.decoder(z)
        return points, masks


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    
    model.train()
    
    total_point_loss = 0.0
    total_mask_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        raster = batch['raster'].to(device)
        target_points = batch['points'].to(device)
        target_masks = batch['masks'].to(device)
        
        # Forward pass
        pred_points, pred_masks = model(raster)
        
        # Point loss: MSE on valid points only
        point_loss = ((pred_points - target_points) ** 2 * target_masks.unsqueeze(-1)).sum()
        point_loss = point_loss / (target_masks.sum() + 1e-8)
        
        # Mask loss: BCE
        mask_loss = nn.functional.binary_cross_entropy(pred_masks, target_masks)
        
        # Combined loss
        loss = point_loss + mask_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track stats
        total_point_loss += point_loss.item()
        total_mask_loss += mask_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'point_loss': f'{point_loss.item():.4f}',
            'mask_loss': f'{mask_loss.item():.4f}'
        })
    
    avg_point_loss = total_point_loss / num_batches
    avg_mask_loss = total_mask_loss / num_batches
    
    return avg_point_loss, avg_mask_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model."""
    
    model.eval()
    
    total_point_loss = 0.0
    total_mask_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        raster = batch['raster'].to(device)
        target_points = batch['points'].to(device)
        target_masks = batch['masks'].to(device)
        
        # Forward pass
        pred_points, pred_masks = model(raster)
        
        # Losses
        point_loss = ((pred_points - target_points) ** 2 * target_masks.unsqueeze(-1)).sum()
        point_loss = point_loss / (target_masks.sum() + 1e-8)
        
        mask_loss = nn.functional.binary_cross_entropy(pred_masks, target_masks)
        
        total_point_loss += point_loss.item()
        total_mask_loss += mask_loss.item()
        num_batches += 1
    
    avg_point_loss = total_point_loss / num_batches
    avg_mask_loss = total_mask_loss / num_batches
    
    return avg_point_loss, avg_mask_loss


def train(
    data_path: Path = Path('data/training_dataset.pkl'),
    output_dir: Path = Path('models/neural_init'),
    num_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_split: float = 0.1,
    device: Optional[torch.device] = None
):
    """Train neural initializer."""
    
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = SVGDataset(data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Model
    model = NeuralInitializer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_point_loss': [],
        'train_mask_loss': [],
        'val_point_loss': [],
        'val_mask_loss': []
    }
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_point_loss, train_mask_loss = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_point_loss, val_mask_loss = validate(model, val_loader, device)
        
        # Log
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train - Point: {train_point_loss:.4f}, Mask: {train_mask_loss:.4f}")
        print(f"  Val   - Point: {val_point_loss:.4f}, Mask: {val_mask_loss:.4f}")
        
        history['train_point_loss'].append(train_point_loss)
        history['train_mask_loss'].append(train_mask_loss)
        history['val_point_loss'].append(val_point_loss)
        history['val_mask_loss'].append(val_mask_loss)
        
        # Save checkpoint
        val_loss = val_point_loss + val_mask_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest_model.pt')
    
    # Save final history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/training_dataset.pkl')
    parser.add_argument('--output', type=str, default='models/neural_init')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    device = None if args.device is None else torch.device(args.device)
    
    train(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
