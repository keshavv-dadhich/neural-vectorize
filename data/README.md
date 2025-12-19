# Data Directory

This directory contains datasets and intermediate files for NeuralVectorize training and evaluation.

## Directory Structure

```
data/
├── raster/              # Original raster (PNG) images for vectorization
├── raster_degraded/     # Degraded raster images (with noise/blur) for training
├── svg_raw/             # Raw SVG files (ground truth)
├── svg_clean/           # Cleaned/preprocessed SVG files
├── dataset_info.json    # Dataset metadata and statistics
└── splits.txt           # Train/validation/test split information
```

## Data Format

### raster/
- **Format:** PNG images (256×256 pixels)
- **Purpose:** Clean input images for vectorization
- **Source:** Rendered from ground truth SVGs

### raster_degraded/
- **Format:** PNG images (256×256 pixels)
- **Purpose:** Training inputs with realistic degradation
- **Degradation:** Gaussian noise, blur, compression artifacts, rotation

### svg_raw/
- **Format:** SVG files
- **Purpose:** Original vector graphics (ground truth)
- **Source:** SVG Repo, Flaticon, custom collections

### svg_clean/
- **Format:** SVG files
- **Purpose:** Preprocessed/normalized SVGs for training
- **Processing:** Standardized viewBox, simplified paths, unified colors

## Dataset Statistics

- **Total samples:** 2,000+ SVGs
- **Categories:** 100+ (business, UI, nature, transport, etc.)
- **Path complexity:** 5-150 paths per icon
- **Resolution:** 256×256px standardized
- **Training set:** 770 samples (77 base icons × 10 augmentations)
- **Test set:** 15 samples for ablation studies

## Usage

### Manual Upload Instructions

Upload your files to the appropriate subdirectories:

1. **svg_raw/**: Place your original SVG files here
2. **raster/**: Place clean PNG renderings here
3. **raster_degraded/**: Place degraded PNG images for training here
4. **svg_clean/**: Place preprocessed SVG files here

### Generate Training Data

```bash
# Create degraded rasters from SVGs
python scripts/create_raster_degradation.py

# Generate training trajectories
python scripts/create_training_data.py
```

### Data Splits

See `splits.txt` for train/validation/test split information. The split is deterministic (seed=42) for reproducibility.

## Dataset Sources

- **SVG Repo**: https://www.svgrepo.com/ (2000+ icons)
- **Flaticon**: https://www.flaticon.com/ (curated collections)
- **Custom**: Company logos, web interfaces, design assets

## File Naming Convention

```
{category}_{id}_{variant}.{ext}

Examples:
- business_001_original.svg
- business_001_clean.svg
- business_001_raster.png
- business_001_degraded.png
```

## License

Dataset compiled from public domain and CC-licensed sources. See individual file licenses in source repositories.
