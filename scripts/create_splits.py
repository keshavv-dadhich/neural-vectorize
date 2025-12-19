"""
PHASE 4: Create Train/Val/Test Splits
=======================================

Split dataset by SVG ID (not by raster variants).

CRITICAL: We split by SVG, not by raster images.
This prevents data leakage where variants of the same SVG
appear in both train and test sets.

Split: 80% train / 10% val / 10% test
"""

import sys
from pathlib import Path
import random

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import load_metadata, save_id_list


def create_splits():
    """Create train/val/test splits by SVG ID."""
    
    print("=" * 60)
    print("PHASE 4: CREATE DATA SPLITS")
    print("=" * 60)
    
    # Load metadata
    try:
        metadata_list = load_metadata(METADATA_FILE)
    except Exception as e:
        print(f"\n❌ Could not load metadata: {e}")
        print("   Please run previous pipeline stages first.")
        return 1
    
    # Extract all SVG IDs
    svg_ids = [m['id'] for m in metadata_list if 'id' in m]
    
    if not svg_ids:
        print("\n❌ No SVG IDs found in metadata")
        return 1
    
    print(f"\n📊 Total SVGs: {len(svg_ids)}")
    print(f"📐 Split ratios: {TRAIN_SPLIT:.0%} / {VAL_SPLIT:.0%} / {TEST_SPLIT:.0%}\n")
    
    # Shuffle for random split
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(svg_ids)
    
    # Calculate split indices
    n_total = len(svg_ids)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    # Rest goes to test
    
    # Split
    train_ids = svg_ids[:n_train]
    val_ids = svg_ids[n_train:n_train + n_val]
    test_ids = svg_ids[n_train + n_val:]
    
    # Save splits
    save_id_list(train_ids, TRAIN_IDS)
    save_id_list(val_ids, VAL_IDS)
    save_id_list(test_ids, TEST_IDS)
    
    # Summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"🚂 Train: {len(train_ids)} SVGs ({len(train_ids) * NUM_VARIANTS} rasters)")
    print(f"📊 Val:   {len(val_ids)} SVGs ({len(val_ids) * NUM_VARIANTS} rasters)")
    print(f"🧪 Test:  {len(test_ids)} SVGs ({len(test_ids) * NUM_VARIANTS} rasters)")
    print(f"\n📁 Split files saved:")
    print(f"   - {TRAIN_IDS}")
    print(f"   - {VAL_IDS}")
    print(f"   - {TEST_IDS}")
    print("\n✅ Phase 4 complete!")
    print("\n🎉 DATASET CREATION COMPLETE!")
    print("\nYou now have a paired raster → vector dataset with:")
    print(f"  • {len(svg_ids)} clean ground-truth SVGs")
    print(f"  • {len(svg_ids) * NUM_VARIANTS} degraded raster images")
    print(f"  • Proper train/val/test splits")
    print(f"  • Complete metadata")
    print("\nNext steps:")
    print("  1. Implement Potrace baseline: python baselines/potrace_runner.py")
    print("  2. Implement benchmarking: python scripts/benchmark.py")
    print("  3. Visualize samples: python scripts/visualize.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(create_splits())
