"""
PHASE 3: Augmentation Pipeline
================================

This script creates degraded variants of perfect raster images.

Degradations simulate real-world conditions:
- Gaussian blur
- Noise (salt & pepper)
- JPEG compression artifacts
- Downscale/upscale
- Optional thresholding

Each perfect raster → N degraded variants for training.
"""

import sys
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import random
import io

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import load_metadata, save_metadata, ProgressTracker


class RasterAugmenter:
    """Handles degradation/augmentation of raster images."""
    
    def __init__(self):
        self.num_variants = NUM_VARIANTS
        self.blur_range = BLUR_SIGMA_RANGE
        self.downscale_size = DOWNSCALE_SIZE
        self.jpeg_range = JPEG_QUALITY_RANGE
        self.noise_max = NOISE_PROB_MAX
    
    def apply_blur(self, image: Image.Image, sigma: float) -> Image.Image:
        """Apply Gaussian blur."""
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    def apply_noise(self, image: Image.Image, prob: float) -> Image.Image:
        """Apply salt and pepper noise."""
        img_array = np.array(image)
        
        # Salt
        salt_mask = np.random.random(img_array.shape) < prob / 2
        img_array[salt_mask] = 255
        
        # Pepper
        pepper_mask = np.random.random(img_array.shape) < prob / 2
        img_array[pepper_mask] = 0
        
        return Image.fromarray(img_array)
    
    def apply_jpeg_compression(self, image: Image.Image, quality: int) -> Image.Image:
        """Apply JPEG compression artifacts."""
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('L')
    
    def apply_downscale_upscale(self, image: Image.Image) -> Image.Image:
        """Downscale then upscale (simulates low resolution source)."""
        orig_size = image.size
        downscaled = image.resize(
            (self.downscale_size, self.downscale_size),
            Image.Resampling.BILINEAR
        )
        upscaled = downscaled.resize(orig_size, Image.Resampling.BILINEAR)
        return upscaled
    
    def apply_threshold(self, image: Image.Image, threshold: int = 127) -> Image.Image:
        """Apply binary thresholding."""
        img_array = np.array(image)
        img_array = (img_array > threshold).astype(np.uint8) * 255
        return Image.fromarray(img_array)
    
    def generate_variant(self, image: Image.Image, variant_id: int) -> Image.Image:
        """
        Generate a single degraded variant with random augmentations.
        
        Args:
            image: Perfect raster image
            variant_id: ID for this variant (affects randomization)
        
        Returns:
            Degraded image
        """
        # Set seed for reproducibility based on variant ID
        random.seed(variant_id)
        np.random.seed(variant_id)
        
        result = image.copy()
        applied_augmentations = []
        
        # Randomly choose which augmentations to apply
        
        # Blur (70% chance)
        if random.random() < 0.7:
            sigma = random.uniform(*self.blur_range)
            result = self.apply_blur(result, sigma)
            applied_augmentations.append(f'blur_{sigma:.2f}')
        
        # Downscale/upscale (50% chance)
        if random.random() < 0.5:
            result = self.apply_downscale_upscale(result)
            applied_augmentations.append('downup')
        
        # JPEG compression (60% chance)
        if random.random() < 0.6:
            quality = random.randint(*self.jpeg_range)
            result = self.apply_jpeg_compression(result, quality)
            applied_augmentations.append(f'jpeg_{quality}')
        
        # Noise (40% chance)
        if random.random() < 0.4:
            prob = random.uniform(0, self.noise_max)
            result = self.apply_noise(result, prob)
            applied_augmentations.append(f'noise_{prob:.4f}')
        
        # Threshold (20% chance, only if enabled)
        if APPLY_THRESHOLD and random.random() < 0.2:
            result = self.apply_threshold(result)
            applied_augmentations.append('threshold')
        
        return result, applied_augmentations
    
    def augment_image(self, image_path: Path, svg_id: str) -> list:
        """
        Create multiple degraded variants of a single image.
        
        Args:
            image_path: Path to perfect raster
            svg_id: ID of the SVG (for output naming)
        
        Returns:
            List of metadata dicts for each variant
        """
        # Load image
        image = Image.open(image_path)
        
        variants_metadata = []
        
        for i in range(1, self.num_variants + 1):
            # Generate degraded variant
            degraded, augmentations = self.generate_variant(image, variant_id=i)
            
            # Save
            output_path = RASTER_DEGRADED / f"{svg_id}_{i:02d}.png"
            degraded.save(output_path, 'PNG')
            
            # Metadata
            variants_metadata.append({
                'variant_id': i,
                'augmentations': augmentations,
                'output_file': output_path.name
            })
        
        return variants_metadata


def main():
    """Run augmentation pipeline on all perfect rasters."""
    
    print("=" * 60)
    print("PHASE 3: AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Ensure output directory exists
    RASTER_DEGRADED.mkdir(parents=True, exist_ok=True)
    
    # Find all perfect rasters
    raster_files = list(RASTER_PERFECT.glob('*_base.png'))
    
    if not raster_files:
        print(f"\n❌ No raster files found in {RASTER_PERFECT}")
        print("   Please run rasterize.py first.")
        return 1
    
    print(f"\n📊 Found {len(raster_files)} raster images")
    print(f"📁 Output: {RASTER_DEGRADED}")
    print(f"🔄 Variants per image: {NUM_VARIANTS}")
    print(f"📈 Total output: {len(raster_files) * NUM_VARIANTS} images\n")
    
    # Initialize augmenter
    augmenter = RasterAugmenter()
    
    # Load existing metadata
    try:
        metadata_list = load_metadata(METADATA_FILE)
        metadata_dict = {m['id']: m for m in metadata_list}
    except:
        print("⚠️  Could not load metadata, creating new")
        metadata_dict = {}
    
    # Process each raster
    tracker = ProgressTracker(len(raster_files), "Augmenting")
    
    successful = 0
    failed = 0
    total_variants = 0
    
    for raster_path in raster_files:
        try:
            # Extract SVG ID from filename (remove _base.png suffix)
            svg_id = raster_path.stem.replace('_base', '')
            
            # Generate variants
            variants_meta = augmenter.augment_image(raster_path, svg_id)
            
            successful += 1
            total_variants += len(variants_meta)
            
            # Update metadata
            if svg_id in metadata_dict:
                metadata_dict[svg_id]['augmented'] = True
                metadata_dict[svg_id]['num_variants'] = len(variants_meta)
                metadata_dict[svg_id]['variants'] = variants_meta
        
        except Exception as e:
            failed += 1
            # Uncomment for debugging:
            # print(f"\n⚠️  Error augmenting {raster_path.name}: {e}")
        
        tracker.update()
    
    tracker.finish()
    
    # Save updated metadata
    print(f"\n📝 Updating metadata...")
    metadata_list = list(metadata_dict.values())
    save_metadata(metadata_list, METADATA_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("n" + "=" * 60)
    print(f"✅ Successfully augmented: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total base images: {len(raster_files)}")
    print(f"🔄 Total variants created: {total_variants}")
    print(f"\n📁 Degraded rasters saved to: {RASTER_DEGRADED}")
    print(f"📄 Metadata updated: {METADATA_FILE}")
    print("\n✅ Phase 3 complete!")
    print("\nNext step: Create data splits")
    print("  python scripts/create_splits.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
