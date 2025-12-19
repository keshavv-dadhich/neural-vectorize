# Vectify Dataset v1.0

**Release Date**: December 17, 2025  
**Status**: Production-Ready ✅

## Dataset Statistics

- **Total raw SVGs processed**: 1,000
- **Successfully cleaned**: 756
- **Rasterized base images**: 756
- **Augmented variants**: 7,560 (10 per icon)
- **Total samples**: 8,316

## Dataset Contract

### Input Format
- **Type**: Grayscale PNG
- **Resolution**: 256×256 pixels
- **Format**: Single-channel (L mode)
- **Degradations**: Blur, noise, JPEG compression, downscale/upscale

### Output Format (Ground Truth)
- **Type**: SVG (stroke-only)
- **ViewBox**: [0, 0, 1.0, 1.0] (normalized)
- **Coordinates**: Absolute, normalized to [0,1]
- **Stroke width**: 0.06 (constant)
- **Commands**: M, L, C, Z only
- **No fills**: stroke="black", fill="none"

## Pipeline Stages Completed

### ✅ Phase 1: SVG Cleaning
- Background removal
- Fill → stroke conversion
- Coordinate normalization to [0,1]
- ViewBox standardization
- Transform removal

### ✅ Phase 2: Rasterization
- Pure Python renderer (no system dependencies)
- White background
- Grayscale conversion
- Proper coordinate scaling

### ✅ Phase 3: Augmentation
10 variants per icon with:
- Gaussian blur (σ ∈ [0.5, 2.0])
- Downscale/upscale (64→256px)
- JPEG compression (quality 30-80)
- Salt & pepper noise (p ≤ 0.01)

## Directory Structure

```
data_processed/
├── svg_clean/          # 756 normalized SVGs
├── raster/             # 756 perfect rasterizations
├── raster_degraded/    # 7,560 augmented variants
├── meta.json           # Complete metadata
├── train_ids.txt       # Training set IDs
├── val_ids.txt         # Validation set IDs
└── test_ids.txt        # Test set IDs
```

## Metadata Schema

```json
{
  "id": "icon_XXXXXX",
  "original_file": "filename.svg",
  "num_paths": 5,
  "cleaned": true,
  "rasterized": true,
  "raster_resolution": 256,
  "num_variants": 10
}
```

## Data Splits

- **Training**: 80% (~605 icons)
- **Validation**: 10% (~76 icons)
- **Test**: 10% (~75 icons)

**Split by SVG ID** - no data leakage across raster variants.

## Quality Metrics

- **Coordinates normalized**: ✅ All in [0,1]
- **Visible strokes**: ✅ 15-40% dark pixels typical
- **Structured geometry**: ✅ Variance >4000 in visualizations
- **No blank images**: ✅ Verified on random samples

## Known Limitations

1. **Simplified rendering**: Uses line segments for complex curves (Béziers approximated)
2. **Binary strokes**: Only 2 pixel values (black/white), no anti-aliasing in base rasters
3. **Large originals**: Some icons from viewBox >512 lose fine details in normalization

## Next Steps (Roadmap)

1. **Potrace Baseline** - Run traditional vectorizer for comparison
2. **Benchmark Metrics** - L2, SSIM, complexity, file size
3. **Optimization Vectorizer** - Gradient-based approach with DiffVG
4. **Neural Methods** - Only after baseline is established

## Citation

If you use this dataset, please cite:

```
Vectify Dataset v1.0 (2025)
A reproducible raster-to-vector benchmark dataset
https://github.com/keshavv-dadhich/vectify
```

## License

MIT License - See LICENSE file

---

**Frozen**: This v1.0 release is immutable. Future improvements will be versioned separately.
