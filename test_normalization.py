"""
Test coordinate normalization on a single SVG.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.clean_svg import SVGCleaner
from config import *

# Test on first raw SVG
raw_svg = next(SVGS_RAW.glob('*.svg'))
output_svg = Path('test_normalized.svg')

print(f"Testing normalization on: {raw_svg.name}")
print(f"Output: {output_svg}\n")

cleaner = SVGCleaner()

try:
    metadata = cleaner.clean_svg(raw_svg, output_svg)
    print(f"✅ Cleaning successful!")
    print(f"Metadata: {metadata}\n")
    
    # Check the output
    content = output_svg.read_text()
    
    # Check viewBox
    if 'viewBox="0 0 1.0 1.0"' in content:
        print("✅ ViewBox normalized to [0,1]")
    
    # Check if coordinates look normalized (should be < 2.0)
    import re
    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', content)
    coords = [float(n) for n in numbers if '.' in n or int(float(n)) < 100]
    
    if coords:
        print(f"✅ Sample coordinates: {coords[:5]}")
        print(f"   Max coordinate: {max(coords):.3f}")
        print(f"   Min coordinate: {min(coords):.3f}")
        
        if max(coords) <= 2.0 and min(coords) >= -1.0:
            print("✅ Coordinates appear normalized!")
        else:
            print("⚠️  Coordinates might not be fully normalized")
    
    print(f"\n✅ Success! Check {output_svg} in a browser")
    print(f"   open {output_svg}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
