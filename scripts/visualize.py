"""
Dataset Visualization
=====================

Visualize samples from the dataset to verify quality:
- Original SVG
- Perfect raster
- Degraded variants

This helps verify the pipeline is working correctly.
"""

import sys
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import load_metadata, load_id_list


def visualize_sample(svg_id: str, num_variants: int = 5):
    """
    Visualize one SVG with its raster variants.
    
    Args:
        svg_id: ID of the SVG to visualize
        num_variants: Number of degraded variants to show
    """
    # Paths
    svg_path = SVG_CLEAN / f"{svg_id}.svg"
    raster_path = RASTER_PERFECT / f"{svg_id}_base.png"
    
    if not svg_path.exists() or not raster_path.exists():
        print(f"⚠️  Files not found for {svg_id}")
        return
    
    # Find degraded variants
    variant_paths = sorted(RASTER_DEGRADED.glob(f"{svg_id}_*.png"))[:num_variants]
    
    # Create figure
    n_cols = 2 + len(variant_paths)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    
    # Show SVG info
    axes[0].text(0.5, 0.5, f"SVG ID:\n{svg_id}\n\nClean SVG →",
                 ha='center', va='center', fontsize=10, wrap=True)
    axes[0].axis('off')
    axes[0].set_title("Ground Truth")
    
    # Show perfect raster
    perfect_img = Image.open(raster_path)
    axes[1].imshow(perfect_img, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Perfect Raster")
    
    # Show degraded variants
    for i, variant_path in enumerate(variant_paths, start=2):
        variant_img = Image.open(variant_path)
        axes[i].imshow(variant_img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Variant {variant_path.stem.split('_')[-1]}")
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUTS / "visualizations" / f"{svg_id}_sample.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {output_path}")


def visualize_dataset_overview(n_samples: int = 9):
    """
    Create a grid showing multiple samples from the dataset.
    
    Args:
        n_samples: Number of samples to show (must be perfect square)
    """
    # Load metadata
    try:
        metadata_list = load_metadata(METADATA_FILE)
        svg_ids = [m['id'] for m in metadata_list if 'id' in m]
    except:
        print("❌ Could not load metadata")
        return
    
    # Random sample
    random.seed(42)
    sample_ids = random.sample(svg_ids, min(n_samples, len(svg_ids)))
    
    # Create grid
    grid_size = int(n_samples ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, svg_id in enumerate(sample_ids):
        raster_path = RASTER_PERFECT / f"{svg_id}_base.png"
        
        if raster_path.exists():
            img = Image.open(raster_path)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f"{svg_id[:12]}...", fontsize=8)
        
        axes[idx].axis('off')
    
    plt.suptitle("Dataset Overview (Perfect Rasters)", fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = OUTPUTS / "visualizations" / "dataset_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved overview: {output_path}")


def main():
    """Generate visualizations."""
    
    print("=" * 60)
    print("DATASET VISUALIZATION")
    print("=" * 60)
    
    # Ensure output directory exists
    (OUTPUTS / "visualizations").mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    try:
        metadata_list = load_metadata(METADATA_FILE)
        svg_ids = [m['id'] for m in metadata_list if 'id' in m]
    except:
        print("\n❌ Could not load metadata")
        print("   Please run the pipeline first.")
        return 1
    
    print(f"\n📊 Total SVGs in dataset: {len(svg_ids)}\n")
    
    # Generate overview
    print("📸 Creating dataset overview...")
    visualize_dataset_overview(n_samples=16)
    
    # Generate individual samples
    print("\n📸 Creating detailed sample visualizations...")
    sample_ids = random.sample(svg_ids, min(5, len(svg_ids)))
    
    for svg_id in sample_ids:
        visualize_sample(svg_id, num_variants=5)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"📁 Visualizations saved to: {OUTPUTS / 'visualizations'}")
    print("\n✅ Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
