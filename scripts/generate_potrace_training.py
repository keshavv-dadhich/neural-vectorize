"""
Generate Potrace SVGs for training samples to use as initialization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import subprocess
from tqdm import tqdm
from PIL import Image
import numpy as np
import tempfile

from config import RASTER_PERFECT, TRAIN_IDS


def generate_potrace_svgs(
    output_dir: Path = Path('baselines/potrace_training'),
    num_samples: int = None
):
    """Generate Potrace SVGs for training samples."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training IDs
    train_ids = TRAIN_IDS.read_text().strip().split('\n')
    if num_samples:
        train_ids = train_ids[:num_samples]
    
    print(f"Generating Potrace SVGs for {len(train_ids)} samples...")
    print(f"Output directory: {output_dir}")
    
    success = 0
    failed = []
    
    for train_id in tqdm(train_ids, desc="Processing"):
        try:
            # Input: perfect raster
            raster_path = RASTER_PERFECT / f"{train_id}_base.png"
            
            if not raster_path.exists():
                failed.append(train_id)
                continue
            
            # Convert PNG to PBM (Potrace only supports PBM/PGM/PPM)
            img = Image.open(raster_path).convert('L')
            # Threshold to binary
            img_array = np.array(img)
            binary = (img_array > 127).astype(np.uint8) * 255
            pbm_img = Image.fromarray(binary, mode='L').convert('1')
            
            # Save to temp PBM
            with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp:
                pbm_path = tmp.name
                pbm_img.save(pbm_path)
            
            # Output SVG
            output_svg = output_dir / f"{train_id}.svg"
            
            # Run Potrace
            result = subprocess.run(
                ['potrace', pbm_path, '-s', '-o', str(output_svg)],
                capture_output=True,
                text=True
            )
            
            # Clean up temp file
            Path(pbm_path).unlink()
            
            if result.returncode == 0 and output_svg.exists():
                success += 1
            else:
                failed.append(train_id)
                
        except Exception as e:
            print(f"Error processing {train_id}: {str(e)}")
            failed.append(train_id)
    
    print(f"\n✅ Potrace generation complete!")
    print(f"   Successful: {success}/{len(train_ids)}")
    print(f"   Failed: {len(failed)}")
    
    if failed:
        print(f"   Failed IDs: {failed[:10]}...")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='baselines/potrace_training')
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()
    
    generate_potrace_svgs(
        output_dir=Path(args.output),
        num_samples=args.samples
    )
