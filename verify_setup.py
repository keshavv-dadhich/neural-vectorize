"""
Verify that Phase 0 setup is complete and correct.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import *

def check_directories():
    """Check all required directories exist."""
    required_dirs = [
        SVGS_RAW, SVG_CLEAN, RASTER_PERFECT, RASTER_DEGRADED,
        OUTPUTS / "visualizations", OUTPUTS / "metrics", BASELINES
    ]
    
    missing = []
    for d in required_dirs:
        if not d.exists():
            missing.append(d)
    
    if missing:
        print("❌ Missing directories:")
        for d in missing:
            print(f"   - {d}")
        return False
    
    print("✅ All directories present")
    return True


def check_files():
    """Check all required files exist."""
    required_files = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "config.py",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "scripts" / "utils.py"
    ]
    
    missing = []
    for f in required_files:
        if not f.exists():
            missing.append(f)
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print("✅ All core files present")
    return True


def check_imports():
    """Test if critical imports work."""
    try:
        import cairosvg
        import svgpathtools
        from PIL import Image
        import cv2
        import numpy as np
        print("✅ All dependencies importable")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")
        return False


def main():
    """Run all verification checks."""
    print("\n🔍 Verifying Phase 0 Setup...\n")
    
    checks = [
        ("Directory Structure", check_directories),
        ("Core Files", check_files),
        ("Dependencies", check_imports)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking: {name}")
        results.append(check_func())
    
    print("\n" + "="*50)
    if all(results):
        print("✅ Phase 0 setup is COMPLETE")
        print("\nYou can now proceed to Phase 1:")
        print("  1. Add raw SVG files to data_raw/svgs_raw/")
        print("  2. Run: python scripts/clean_svg.py")
        return 0
    else:
        print("❌ Phase 0 setup is INCOMPLETE")
        print("\nPlease fix the issues above and run again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
