"""
Potrace Baseline Runner
=======================

Runs Potrace on degraded raster images to establish a baseline
for vectorization quality comparison.

Potrace: http://potrace.sourceforge.net/
Install: brew install potrace (macOS) or apt-get install potrace (Linux)
"""

import sys
from pathlib import Path
import subprocess
import shutil

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import ProgressTracker, load_id_list


def check_potrace_installed() -> bool:
    """Check if Potrace is installed and accessible."""
    return shutil.which('potrace') is not None


def run_potrace(input_png: Path, output_svg: Path) -> bool:
    """
    Run Potrace on a single PNG image.
    
    Converts grayscale to binary first (Potrace requirement).
    
    Args:
        input_png: Path to input PNG
        output_svg: Path for output SVG
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        import tempfile
        
        # Load image and convert to binary (1-bit)
        img = Image.open(input_png).convert('L')
        
        # Threshold to binary: <128 = black (foreground), >=128 = white (background)
        binary = img.point(lambda x: 0 if x < 128 else 255, mode='1')
        
        # Save temporary BMP (Potrace prefers BMP)
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp:
            tmp_path = tmp.name
            binary.save(tmp_path, 'BMP')
        
        # Run potrace on binary BMP
        result = subprocess.run(
            [
                'potrace',
                '-s',  # SVG output
                '-o', str(output_svg),  # Output file
                tmp_path
            ],
            capture_output=True,
            timeout=10,
            text=True
        )
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        return result.returncode == 0 and output_svg.exists()
    
    except Exception as e:
        # Uncomment for debugging:
        # print(f"\n⚠️  Error running potrace on {input_png.name}: {e}")
        return False


def main():
    """Run Potrace baseline on test set degraded rasters."""
    
    print("=" * 60)
    print("POTRACE BASELINE RUNNER")
    print("=" * 60)
    
    # Check if Potrace is installed
    if not check_potrace_installed():
        print("\n❌ Potrace is not installed!")
        print("\nInstall with:")
        print("  macOS:  brew install potrace")
        print("  Linux:  sudo apt-get install potrace")
        print("  Windows: Download from http://potrace.sourceforge.net/")
        return 1
    
    print("\n✅ Potrace found")
    
    # Create output directory
    potrace_output = BASELINES / "potrace"
    potrace_output.mkdir(parents=True, exist_ok=True)
    
    # Load test set IDs
    try:
        test_ids = load_id_list(TEST_IDS)
        print(f"📊 Loaded {len(test_ids)} test samples")
    except:
        print(f"\n❌ Could not load test IDs from {TEST_IDS}")
        print("   Please run create_splits.py first.")
        return 1
    
    # Find degraded rasters for test set
    # We'll run Potrace on variant 01 of each test sample
    raster_files = []
    for svg_id in test_ids:
        raster_path = RASTER_DEGRADED / f"{svg_id}_01.png"
        if raster_path.exists():
            raster_files.append((svg_id, raster_path))
    
    if not raster_files:
        print(f"\n❌ No degraded rasters found for test set")
        return 1
    
    print(f"📁 Found {len(raster_files)} test rasters")
    print(f"📁 Output: {potrace_output}\n")
    
    # Process each raster
    tracker = ProgressTracker(len(raster_files), "Running Potrace")
    
    successful = 0
    failed = 0
    
    for svg_id, raster_path in raster_files:
        output_svg = potrace_output / f"{svg_id}.svg"
        
        if run_potrace(raster_path, output_svg):
            successful += 1
        else:
            failed += 1
        
        tracker.update()
    
    tracker.finish()
    
    # Summary
    print("\n" + "=" * 60)
    print("POTRACE BASELINE SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully vectorized: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total processed: {len(raster_files)}")
    print(f"\n📁 Potrace SVGs saved to: {potrace_output}")
    print("\n✅ Baseline complete!")
    print("\nNext step: Run benchmark")
    print("  python scripts/benchmark.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
