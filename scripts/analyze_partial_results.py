"""
Analyze partial results from ongoing benchmark.

Shows current progress and preliminary metrics.
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import json

import config

RASTER_PERFECT = config.RASTER_PERFECT
BASELINES = config.BASELINES


def analyze_partial_results(
    results_dir: Path = Path('baselines/advanced_full')
):
    """Analyze completed samples from ongoing benchmark."""
    
    # Find completed SVGs
    svg_files = list(results_dir.glob('*.svg'))
    
    if not svg_files:
        print("No results found yet!")
        return
    
    print(f"Analyzing {len(svg_files)} completed samples...")
    print(f"Results directory: {results_dir}")
    print()
    
    # Compute metrics for completed samples
    results = []
    
    for svg_path in svg_files:
        try:
            # Extract ID from filename
            sample_id = svg_path.stem
            
            # Load target raster
            raster_path = RASTER_PERFECT / f"{sample_id}_base.png"
            if not raster_path.exists():
                continue
            
            target = np.array(Image.open(raster_path).convert('L')) / 255.0
            
            # Simple rasterization for L2
            from PIL import ImageDraw
            import xml.etree.ElementTree as ET
            import re
            
            img = Image.new('L', (256, 256), 255)
            draw = ImageDraw.Draw(img)
            
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            num_segments = 0
            for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
                d = path_elem.get('d', '')
                coords = []
                tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
                for token in tokens:
                    match = re.search(r'([\d.]+),([\d.]+)', token)
                    if match:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        coords.append((x, y))
                
                if len(coords) >= 2:
                    for i in range(len(coords) - 1):
                        draw.line([coords[i], coords[i+1]], fill=0, width=2)
                    num_segments += len(coords) - 1
            
            predicted = np.array(img) / 255.0
            l2 = np.mean((predicted - target) ** 2)
            
            # File size
            file_size = svg_path.stat().st_size
            
            results.append({
                'id': sample_id,
                'l2': float(l2),
                'segments': num_segments,
                'file_size': file_size
            })
            
        except Exception as e:
            print(f"Error processing {svg_path.name}: {e}")
            continue
    
    if not results:
        print("No valid results to analyze!")
        return
    
    # Compute statistics
    l2_values = [r['l2'] for r in results]
    segment_values = [r['segments'] for r in results]
    size_values = [r['file_size'] for r in results]
    
    print("=" * 70)
    print("PARTIAL RESULTS (Preliminary)")
    print("=" * 70)
    print(f"\nCompleted samples: {len(results)}/77 ({len(results)/77*100:.1f}%)")
    print()
    print("Metrics:")
    print(f"  L2 Error:")
    print(f"    Mean: {np.mean(l2_values):.4f}")
    print(f"    Std:  {np.std(l2_values):.4f}")
    print(f"    Min:  {np.min(l2_values):.4f}")
    print(f"    Max:  {np.max(l2_values):.4f}")
    print()
    print(f"  Segments:")
    print(f"    Mean: {np.mean(segment_values):.1f}")
    print(f"    Std:  {np.std(segment_values):.1f}")
    print(f"    Min:  {np.min(segment_values):.0f}")
    print(f"    Max:  {np.max(segment_values):.0f}")
    print()
    print(f"  File Size:")
    print(f"    Mean: {np.mean(size_values)/1024:.1f} KB")
    print(f"    Min:  {np.min(size_values)/1024:.1f} KB")
    print(f"    Max:  {np.max(size_values)/1024:.1f} KB")
    
    # Compare with baselines if available
    print()
    print("=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    
    # Check for Potrace baseline
    potrace_dir = BASELINES / 'potrace'
    if potrace_dir.exists():
        potrace_l2s = []
        potrace_segments = []
        
        for result in results:
            potrace_svg = potrace_dir / f"{result['id']}.svg"
            if potrace_svg.exists():
                try:
                    # Quick L2 computation
                    img = Image.new('L', (256, 256), 255)
                    draw = ImageDraw.Draw(img)
                    
                    tree = ET.parse(potrace_svg)
                    root = tree.getroot()
                    
                    num_segs = 0
                    for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
                        d = path_elem.get('d', '')
                        coords = []
                        tokens = re.findall(r'[MLC]\s*[\d.]+,[\d.]+', d)
                        for token in tokens:
                            match = re.search(r'([\d.]+),([\d.]+)', token)
                            if match:
                                x = float(match.group(1))
                                y = float(match.group(2))
                                coords.append((x, y))
                        
                        if len(coords) >= 2:
                            for i in range(len(coords) - 1):
                                draw.line([coords[i], coords[i+1]], fill=0, width=2)
                            num_segs += len(coords) - 1
                    
                    predicted = np.array(img) / 255.0
                    raster_path = RASTER_PERFECT / f"{result['id']}_base.png"
                    target = np.array(Image.open(raster_path).convert('L')) / 255.0
                    l2 = np.mean((predicted - target) ** 2)
                    
                    potrace_l2s.append(l2)
                    potrace_segments.append(num_segs)
                except:
                    pass
        
        if potrace_l2s:
            print(f"\nPotrace ({len(potrace_l2s)} samples):")
            print(f"  L2: {np.mean(potrace_l2s):.4f} ± {np.std(potrace_l2s):.4f}")
            print(f"  Segments: {np.mean(potrace_segments):.1f}")
            
            print(f"\nAdvanced (Edge-Aware):")
            print(f"  L2: {np.mean(l2_values):.4f} ± {np.std(l2_values):.4f}")
            print(f"  Segments: {np.mean(segment_values):.1f}")
            
            improvement = (1 - np.mean(l2_values) / np.mean(potrace_l2s)) * 100
            print(f"\n✅ L2 improvement over Potrace: {improvement:.1f}%")
            
            seg_reduction = (1 - np.mean(segment_values) / np.mean(potrace_segments)) * 100
            print(f"✅ Segment reduction: {seg_reduction:.1f}%")
    
    # Save partial results
    partial_summary = {
        'num_completed': len(results),
        'total_samples': 77,
        'progress_pct': len(results) / 77 * 100,
        'metrics': {
            'l2_mean': float(np.mean(l2_values)),
            'l2_std': float(np.std(l2_values)),
            'segments_mean': float(np.mean(segment_values)),
            'segments_std': float(np.std(segment_values)),
        },
        'results': results
    }
    
    output_path = results_dir / 'partial_results.json'
    with open(output_path, 'w') as f:
        json.dump(partial_summary, f, indent=2)
    
    print(f"\nPartial results saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='baselines/advanced_full')
    args = parser.parse_args()
    
    analyze_partial_results(Path(args.dir))
