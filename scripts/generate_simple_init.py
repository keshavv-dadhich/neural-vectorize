"""
Generate simple initial SVGs from raster images using edge detection.

Creates basic line-based SVGs that can be used as initialization for optimization.
This is simpler than Potrace and will definitely parse correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from config import RASTER_PERFECT, TRAIN_IDS


def create_simple_svg_from_edges(
    raster: np.ndarray,
    output_path: Path,
    num_lines: int = 50,
    viewbox_size: int = 256
) -> None:
    """
    Create a simple SVG with lines following detected edges.
    
    Args:
        raster: Input image (H, W) in [0, 1]
        output_path: Where to save SVG
        num_lines: Number of line segments to generate
        viewbox_size: SVG viewbox size
    """
    # Convert to uint8
    img_uint8 = (raster * 255).astype(np.uint8)
    
    # Detect edges
    edges = cv2.Canny(img_uint8, 50, 150)
    
    # Find edge points
    edge_points = np.argwhere(edges > 0)
    
    if len(edge_points) == 0:
        # No edges found, create a single path at center
        edge_points = np.array([[128, 128], [130, 130]])
    
    # Sample points for lines
    if len(edge_points) > num_lines * 2:
        # Randomly sample points
        indices = np.random.choice(len(edge_points), num_lines * 2, replace=False)
        sampled_points = edge_points[indices]
    else:
        sampled_points = edge_points
    
    # Create SVG
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{viewbox_size}" height="{viewbox_size}" viewBox="0 0 {viewbox_size} {viewbox_size}">',
    ]
    
    # Create paths from consecutive point pairs
    for i in range(0, len(sampled_points) - 1, 2):
        y1, x1 = sampled_points[i]
        y2, x2 = sampled_points[i + 1] if i + 1 < len(sampled_points) else sampled_points[i]
        
        # Simple line path
        path_d = f"M {x1},{y1} L {x2},{y2}"
        svg_lines.append(f'  <path d="{path_d}" stroke="black" stroke-width="1" fill="none"/>')
    
    svg_lines.append('</svg>')
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(svg_lines))


def generate_simple_svgs_for_training(
    output_dir: Path = Path('baselines/simple_init'),
    num_samples: int = None
):
    """Generate simple SVG initializations for training samples."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training IDs
    train_ids = TRAIN_IDS.read_text().strip().split('\n')
    if num_samples:
        train_ids = train_ids[:num_samples]
    
    print(f"Generating simple SVGs for {len(train_ids)} training samples...")
    print(f"Output directory: {output_dir}")
    
    success = 0
    failed = []
    
    for train_id in tqdm(train_ids, desc="Processing"):
        try:
            # Load perfect raster
            raster_path = RASTER_PERFECT / f"{train_id}_base.png"
            
            if not raster_path.exists():
                failed.append(train_id)
                continue
            
            # Load and convert to grayscale
            img = Image.open(raster_path).convert('L')
            raster = np.array(img) / 255.0
            
            # Create simple SVG
            output_svg = output_dir / f"{train_id}.svg"
            create_simple_svg_from_edges(raster, output_svg, num_lines=30)
            
            success += 1
            
        except Exception as e:
            print(f"Error processing {train_id}: {str(e)}")
            failed.append(train_id)
            continue
    
    print(f"\n✅ Simple SVG generation complete!")
    print(f"   Successful: {success}/{len(train_ids)}")
    print(f"   Failed: {len(failed)}")
    
    if failed and len(failed) < 10:
        print(f"   Failed IDs: {failed}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='baselines/simple_init')
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()
    
    generate_simple_svgs_for_training(
        output_dir=Path(args.output),
        num_samples=args.samples
    )
