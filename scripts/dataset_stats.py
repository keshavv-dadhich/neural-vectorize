"""
Dataset Statistics and Summary
===============================

Generate comprehensive statistics about the dataset.
"""

import sys
from pathlib import Path
import json

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import load_metadata, load_id_list


def compute_statistics():
    """Compute and display dataset statistics."""
    
    print("\n" + "=" * 70)
    print(" " * 20 + "VECTIFY DATASET SUMMARY")
    print("=" * 70)
    
    # Load metadata
    try:
        metadata_list = load_metadata(METADATA_FILE)
    except:
        print("\n❌ Could not load metadata")
        return
    
    # Load splits
    try:
        train_ids = load_id_list(TRAIN_IDS)
        val_ids = load_id_list(VAL_IDS)
        test_ids = load_id_list(TEST_IDS)
    except:
        train_ids = val_ids = test_ids = []
    
    # Basic counts
    n_svgs = len(metadata_list)
    n_clean = sum(1 for m in metadata_list if m.get('cleaned', False))
    n_rasterized = sum(1 for m in metadata_list if m.get('rasterized', False))
    n_augmented = sum(1 for m in metadata_list if m.get('augmented', False))
    
    # Path counts
    total_paths = sum(m.get('num_paths', 0) for m in metadata_list)
    avg_paths = total_paths / n_svgs if n_svgs > 0 else 0
    
    # Variant counts
    total_variants = sum(m.get('num_variants', 0) for m in metadata_list)
    
    print("\n📊 DATASET STATISTICS")
    print("-" * 70)
    print(f"{'Total SVGs processed:':<35} {n_svgs:>10}")
    print(f"{'Successfully cleaned:':<35} {n_clean:>10}")
    print(f"{'Successfully rasterized:':<35} {n_rasterized:>10}")
    print(f"{'Successfully augmented:':<35} {n_augmented:>10}")
    print()
    print(f"{'Total degraded raster images:':<35} {total_variants:>10}")
    print(f"{'Average paths per SVG:':<35} {avg_paths:>10.1f}")
    
    print("\n📂 DATA SPLITS")
    print("-" * 70)
    print(f"{'Training SVGs:':<35} {len(train_ids):>10} ({len(train_ids) * NUM_VARIANTS} rasters)")
    print(f"{'Validation SVGs:':<35} {len(val_ids):>10} ({len(val_ids) * NUM_VARIANTS} rasters)")
    print(f"{'Test SVGs:':<35} {len(test_ids):>10} ({len(test_ids) * NUM_VARIANTS} rasters)")
    
    print("\n⚙️  CONFIGURATION")
    print("-" * 70)
    print(f"{'Raster resolution:':<35} {RASTER_SIZE}×{RASTER_SIZE}")
    print(f"{'Variants per SVG:':<35} {NUM_VARIANTS:>10}")
    print(f"{'Stroke width:':<35} {STROKE_WIDTH:>10.3f}")
    print(f"{'Blur sigma range:':<35} {str(BLUR_SIGMA_RANGE):>10}")
    print(f"{'JPEG quality range:':<35} {str(JPEG_QUALITY_RANGE):>10}")
    
    print("\n📁 OUTPUT DIRECTORIES")
    print("-" * 70)
    print(f"Clean SVGs:        {SVG_CLEAN}")
    print(f"Perfect rasters:   {RASTER_PERFECT}")
    print(f"Degraded rasters:  {RASTER_DEGRADED}")
    print(f"Metadata:          {METADATA_FILE}")
    print(f"Visualizations:    {OUTPUTS / 'visualizations'}")
    
    print("\n✅ PIPELINE STATUS")
    print("-" * 70)
    
    stages = [
        ("Phase 0: Setup", True),
        ("Phase 1: SVG Cleaning", n_clean > 0),
        ("Phase 2: Rasterization", n_rasterized > 0),
        ("Phase 3: Augmentation", n_augmented > 0),
        ("Phase 4: Data Splits", len(train_ids) > 0),
    ]
    
    for stage, complete in stages:
        status = "✅" if complete else "❌"
        print(f"{status} {stage}")
    
    print("\n🎯 NEXT STEPS")
    print("-" * 70)
    print("1. Implement Potrace baseline for comparison")
    print("2. Implement benchmarking metrics (L2, SSIM, complexity)")
    print("3. Build optimization-based vectorizer")
    print("4. Scale up dataset (currently using 1000 of 308k available SVGs)")
    print("5. Train ML models (optional)")
    
    print("\n" + "=" * 70)
    print(" " * 25 + "DATASET READY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compute_statistics()
