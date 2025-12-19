"""
Central configuration for the Vectify project.
All paths, constants, and hyperparameters defined here.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
OUTPUTS = PROJECT_ROOT / "outputs"
BASELINES = PROJECT_ROOT / "baselines"

# Raw data
SVGS_RAW = DATA_RAW / "svgs_raw"

# Processed data
SVG_CLEAN = DATA_PROCESSED / "svg_clean"
RASTER_PERFECT = DATA_PROCESSED / "raster"
RASTER_DEGRADED = DATA_PROCESSED / "raster_degraded"

# Metadata
METADATA_FILE = DATA_PROCESSED / "meta.json"
TRAIN_IDS = DATA_PROCESSED / "train_ids.txt"
VAL_IDS = DATA_PROCESSED / "val_ids.txt"
TEST_IDS = DATA_PROCESSED / "test_ids.txt"

# ============================================================================
# SVG PROCESSING CONSTANTS
# ============================================================================

# Normalization
VIEWBOX_SIZE = 1.0  # Normalize to [0,1] × [0,1]
STROKE_WIDTH = 0.06  # Constant stroke width in normalized space (increased for visibility)
EPSILON = 1e-6  # Tolerance for geometric comparisons

# Simplification
MIN_SEGMENT_LENGTH = 0.01  # Remove segments shorter than this
COLLINEAR_THRESHOLD = 0.05  # Merge lines within this angle (radians)

# ============================================================================
# RASTERIZATION CONSTANTS
# ============================================================================

# Image dimensions
RASTER_SIZE = 256  # Output image size (square)
DPI = 96  # Standard DPI for SVG rendering

# ============================================================================
# AUGMENTATION PARAMETERS
# ============================================================================

# Number of degraded variants per SVG
NUM_VARIANTS = 10

# Gaussian blur
BLUR_SIGMA_RANGE = (0.5, 2.0)

# Downscale/upscale
DOWNSCALE_SIZE = 64

# JPEG compression
JPEG_QUALITY_RANGE = (30, 80)

# Salt & pepper noise
NOISE_PROB_MAX = 0.01

# Thresholding (optional)
APPLY_THRESHOLD = False
THRESHOLD_VALUE = 127

# ============================================================================
# DATASET SPLIT
# ============================================================================

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ============================================================================
# OPTIMIZATION CONSTANTS (for future use)
# ============================================================================

# Loss weights
WEIGHT_RECON = 1.0
WEIGHT_COVER = 0.5
WEIGHT_CURVE = 0.1
WEIGHT_SNAP = 0.1
WEIGHT_WIDTH = 0.05
WEIGHT_COMPLEX = 0.01

# Optimization
NUM_OPT_STEPS = 500
NUM_SIMPLIFY_STEPS = 100
LEARNING_RATE = 0.01

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    dirs = [
        DATA_RAW, DATA_PROCESSED, OUTPUTS, BASELINES,
        SVGS_RAW, SVG_CLEAN, RASTER_PERFECT, RASTER_DEGRADED,
        OUTPUTS / "visualizations", OUTPUTS / "metrics"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ All directories created/verified")

if __name__ == "__main__":
    ensure_dirs()
    print(f"\n📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Raw SVGs: {SVGS_RAW}")
    print(f"📁 Clean SVGs: {SVG_CLEAN}")
    print(f"📁 Rasters: {RASTER_PERFECT}")
