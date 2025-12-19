#!/bin/bash

# Vectify - Full Pipeline Runner
# ================================
# Runs the complete dataset generation pipeline

set -e  # Exit on error

echo "=========================================="
echo "  VECTIFY DATASET GENERATION PIPELINE"
echo "=========================================="
echo ""

# Check if raw SVGs exist
if [ ! -d "data_raw/svgs_raw" ] || [ -z "$(ls -A data_raw/svgs_raw)" ]; then
    echo "❌ Error: No raw SVG files found in data_raw/svgs_raw/"
    echo "   Please add SVG files and run again."
    exit 1
fi

echo "📊 Found SVG files. Starting pipeline..."
echo ""

# Phase 1: Clean SVGs
echo "🔹 Phase 1: Cleaning SVGs..."
python3 scripts/clean_svg.py
if [ $? -ne 0 ]; then
    echo "❌ SVG cleaning failed"
    exit 1
fi
echo ""

# Phase 2: Rasterize
echo "🔹 Phase 2: Rasterizing to PNG..."
python3 scripts/rasterize.py
if [ $? -ne 0 ]; then
    echo "❌ Rasterization failed"
    exit 1
fi
echo ""

# Phase 3: Augment
echo "🔹 Phase 3: Creating degraded variants..."
python3 scripts/augment.py
if [ $? -ne 0 ]; then
    echo "❌ Augmentation failed"
    exit 1
fi
echo ""

# Phase 4: Create splits
echo "🔹 Phase 4: Creating train/val/test splits..."
python3 scripts/create_splits.py
if [ $? -ne 0 ]; then
    echo "❌ Split creation failed"
    exit 1
fi
echo ""

# Phase 5: Visualize
echo "🔹 Phase 5: Generating visualizations..."
python3 scripts/visualize.py
if [ $? -ne 0 ]; then
    echo "⚠️  Visualization failed (non-critical)"
fi
echo ""

# Show statistics
echo "🔹 Dataset Statistics:"
python3 scripts/dataset_stats.py

echo ""
echo "=========================================="
echo "  ✅ PIPELINE COMPLETE!"
echo "=========================================="
