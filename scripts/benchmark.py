"""
Benchmark Metrics Script
========================

Evaluates vectorization quality by comparing predicted SVGs against:
1. Input raster (reconstruction error)
2. Ground truth SVG (complexity, topology)

Metrics computed:
- L2 reconstruction error
- SSIM (Structural Similarity)
- Number of paths
- Number of segments (line + curve commands)
- SVG file size
"""

import sys
from pathlib import Path
import json
from typing import Dict, List
import numpy as np
from PIL import Image

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from scripts.utils import load_id_list, ProgressTracker


def rasterize_svg_simple(svg_path: Path, size: int = 256) -> np.ndarray:
    """
    Simple SVG rasterization for comparison.
    Returns numpy array of shape (size, size).
    """
    # Reuse the rasterization logic from rasterize.py
    import xml.etree.ElementTree as ET
    from PIL import ImageDraw
    import re
    
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Get viewBox
        viewbox = root.get('viewBox', '0 0 1 1')
        try:
            vx, vy, vw, vh = map(float, viewbox.split())
        except:
            vx, vy, vw, vh = 0, 0, 1, 1
        
        # Find coordinate bounds
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for elem in root.iter():
            if 'path' in str(elem.tag).lower():
                d = elem.get('d', '')
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
                coords = [float(n) for n in numbers]
                
                for i in range(0, len(coords)-1, 2):
                    x, y = coords[i], coords[i+1]
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        if min_x == float('inf'):
            min_x, min_y = 0, 0
            max_x, max_y = 100, 100
        
        actual_width = max_x - min_x
        actual_height = max_y - min_y
        padding = 0.1 * max(actual_width, actual_height)
        min_x -= padding
        min_y -= padding
        actual_width += 2 * padding
        actual_height += 2 * padding
        
        # Create image
        img = Image.new('L', (size, size), color=255)
        draw = ImageDraw.Draw(img)
        
        scale = min(size / actual_width, size / actual_height)
        
        # Render paths
        for elem in root.iter():
            if 'path' in str(elem.tag).lower():
                d = elem.get('d', '')
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
                if len(numbers) < 4:
                    continue
                
                coords = [float(n) for n in numbers]
                points = []
                
                for i in range(0, len(coords)-1, 2):
                    x = (coords[i] - min_x) * scale
                    y = (coords[i+1] - min_y) * scale
                    points.append((int(x), int(y)))
                
                if len(points) >= 2:
                    draw.line(points, fill=0, width=2)
        
        return np.array(img) / 255.0  # Normalize to [0,1]
    
    except Exception as e:
        # Return blank on error
        return np.ones((size, size))


def compute_l2_error(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute L2 (MSE) between two images."""
    return np.mean((img1 - img2) ** 2)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM (Structural Similarity Index).
    Simplified implementation.
    """
    from scipy import signal
    
    # Constants
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    # Gaussian kernel
    kernel = np.outer(signal.gaussian(11, 1.5), signal.gaussian(11, 1.5))
    kernel /= kernel.sum()
    
    # Mean
    mu1 = signal.convolve2d(img1, kernel, mode='valid')
    mu2 = signal.convolve2d(img2, kernel, mode='valid')
    
    # Variance and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = signal.convolve2d(img1 ** 2, kernel, mode='valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 ** 2, kernel, mode='valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, kernel, mode='valid') - mu1_mu2
    
    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


def count_svg_primitives(svg_path: Path) -> Dict[str, int]:
    """
    Count paths and segments in an SVG.
    
    Returns:
        Dict with 'paths', 'segments', 'file_size_kb'
    """
    import xml.etree.ElementTree as ET
    import re
    
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        num_paths = 0
        num_segments = 0
        
        for elem in root.iter():
            if 'path' in str(elem.tag).lower():
                num_paths += 1
                d = elem.get('d', '')
                
                # Count M, L, C, Z commands
                commands = re.findall(r'[MLCZmlcz]', d)
                num_segments += len(commands)
        
        file_size_kb = svg_path.stat().st_size / 1024
        
        return {
            'paths': num_paths,
            'segments': num_segments,
            'file_size_kb': file_size_kb
        }
    
    except Exception:
        return {'paths': 0, 'segments': 0, 'file_size_kb': 0}


def benchmark_method(method_name: str, svg_dir: Path, test_ids: List[str], 
                     raster_dir: Path) -> Dict:
    """
    Benchmark a vectorization method.
    
    Args:
        method_name: Name of the method (e.g., "Potrace", "Ground Truth")
        svg_dir: Directory containing SVGs
        test_ids: List of test set IDs
        raster_dir: Directory containing input rasters
    
    Returns:
        Dict with aggregated metrics
    """
    print(f"\n📊 Benchmarking: {method_name}")
    
    l2_errors = []
    ssim_scores = []
    path_counts = []
    segment_counts = []
    file_sizes = []
    
    tracker = ProgressTracker(len(test_ids), f"  {method_name}")
    
    for svg_id in test_ids:
        # Find SVG
        svg_path = svg_dir / f"{svg_id}.svg"
        if not svg_path.exists():
            tracker.update()
            continue
        
        # Get input raster (variant 01)
        raster_path = raster_dir / f"{svg_id}_01.png"
        if not raster_path.exists():
            tracker.update()
            continue
        
        # Load input raster
        input_img = np.array(Image.open(raster_path).convert('L')) / 255.0
        
        # Rasterize SVG
        pred_img = rasterize_svg_simple(svg_path)
        
        # Compute reconstruction metrics
        l2 = compute_l2_error(input_img, pred_img)
        ssim = compute_ssim(input_img, pred_img)
        
        l2_errors.append(l2)
        ssim_scores.append(ssim)
        
        # Count primitives
        counts = count_svg_primitives(svg_path)
        path_counts.append(counts['paths'])
        segment_counts.append(counts['segments'])
        file_sizes.append(counts['file_size_kb'])
        
        tracker.update()
    
    tracker.finish()
    
    # Aggregate
    return {
        'method': method_name,
        'samples': len(l2_errors),
        'l2_mean': float(np.mean(l2_errors)) if l2_errors else 0,
        'l2_std': float(np.std(l2_errors)) if l2_errors else 0,
        'ssim_mean': float(np.mean(ssim_scores)) if ssim_scores else 0,
        'ssim_std': float(np.std(ssim_scores)) if ssim_scores else 0,
        'paths_mean': float(np.mean(path_counts)) if path_counts else 0,
        'segments_mean': float(np.mean(segment_counts)) if segment_counts else 0,
        'file_size_kb_mean': float(np.mean(file_sizes)) if file_sizes else 0
    }


def main():
    """Run benchmark on all methods."""
    
    print("=" * 60)
    print("BENCHMARK EVALUATION")
    print("=" * 60)
    
    # Load test IDs
    try:
        test_ids = load_id_list(TEST_IDS)
        print(f"\n📊 Test set: {len(test_ids)} samples")
    except:
        print(f"\n❌ Could not load test IDs from {TEST_IDS}")
        return 1
    
    # Benchmark methods
    results = []
    
    # 1. Ground Truth (our cleaned SVGs)
    print("\n" + "="*60)
    print("METHOD 1: Ground Truth (Clean SVGs)")
    print("="*60)
    gt_results = benchmark_method(
        "Ground Truth",
        SVG_CLEAN,
        test_ids,
        RASTER_DEGRADED
    )
    results.append(gt_results)
    
    # 2. Potrace baseline
    potrace_dir = BASELINES / "potrace"
    if potrace_dir.exists():
        print("\n" + "="*60)
        print("METHOD 2: Potrace Baseline")
        print("="*60)
        potrace_results = benchmark_method(
            "Potrace",
            potrace_dir,
            test_ids,
            RASTER_DEGRADED
        )
        results.append(potrace_results)
    else:
        print(f"\n⚠️  Potrace results not found at {potrace_dir}")
        print("   Run: python baselines/potrace_runner.py")
    
    # 3. Optimization baseline (full 77 samples)
    optimization_dir = BASELINES / "optimization_full"
    if optimization_dir.exists():
        print("\n" + "="*60)
        print("METHOD 3: Optimization (Full Scale)")
        print("="*60)
        optimization_results = benchmark_method(
            "Optimization",
            optimization_dir,
            test_ids,
            RASTER_DEGRADED
        )
        results.append(optimization_results)
    else:
        print(f"\n⚠️  Optimization results not found at {optimization_dir}")
        print("   Run: python vectorizers/scale.py --samples 77")
    
    # Save results
    results_file = OUTPUTS / "metrics" / "benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n{'Method':<20} {'L2↓':<10} {'SSIM↑':<10} {'Segs↓':<10} {'Size(KB)↓':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['method']:<20} {r['l2_mean']:<10.4f} {r['ssim_mean']:<10.4f} "
              f"{r['segments_mean']:<10.1f} {r['file_size_kb_mean']:<12.2f}")
    
    print("\n📁 Detailed results saved to:")
    print(f"   {results_file}")
    
    print("\n✅ Benchmark complete!")
    
    return 0


if __name__ == "__main__":
    # Install scipy if needed
    try:
        import scipy
    except ImportError:
        print("Installing scipy for SSIM computation...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'scipy', '--quiet'])
    
    sys.exit(main())
