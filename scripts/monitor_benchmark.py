"""
Monitor benchmark progress and generate report when complete.

Checks every 60 seconds and creates final report when all 77 samples are done.
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import time
import json
from datetime import datetime


def monitor_benchmark(
    results_dir: Path = Path('baselines/advanced_full'),
    check_interval: int = 60,
    total_samples: int = 77
):
    """Monitor benchmark and report when complete."""
    
    print(f"Monitoring benchmark at: {results_dir}")
    print(f"Target: {total_samples} samples")
    print(f"Checking every {check_interval} seconds...")
    print()
    
    last_count = 0
    start_time = time.time()
    
    while True:
        # Count completed SVGs
        svg_files = list(results_dir.glob('*.svg'))
        current_count = len(svg_files)
        
        if current_count != last_count:
            elapsed = time.time() - start_time
            progress_pct = (current_count / total_samples) * 100
            
            if current_count > last_count:
                # Calculate ETA
                samples_per_sec = current_count / elapsed if elapsed > 0 else 0
                remaining_samples = total_samples - current_count
                eta_sec = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                eta_min = eta_sec / 60
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Progress: {current_count}/{total_samples} ({progress_pct:.1f}%) | "
                      f"ETA: {eta_min:.0f} min")
            
            last_count = current_count
        
        # Check if complete
        if current_count >= total_samples:
            print()
            print("=" * 70)
            print("✅ BENCHMARK COMPLETE!")
            print("=" * 70)
            print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
            print(f"Completed samples: {current_count}/{total_samples}")
            print()
            print("Generating final report...")
            
            # Run analysis
            import subprocess
            result = subprocess.run(
                ['python3', 'scripts/benchmark_full.py', '--compare'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Analysis output:")
                print(result.stdout)
                if result.stderr:
                    print("Errors:")
                    print(result.stderr)
            
            # Check for results.json
            results_file = results_dir / 'results.json'
            if results_file.exists():
                print(f"\nDetailed results saved to: {results_file}")
                
                # Print summary
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                if 'metrics' in data:
                    print("\nFinal Metrics:")
                    print(f"  L2: {data['metrics']['l2_mean']:.4f} ± {data['metrics']['l2_std']:.4f}")
                    print(f"  SSIM: {data['metrics']['ssim_mean']:.4f} ± {data['metrics']['ssim_std']:.4f}")
                    print(f"  Segments: {data['metrics']['segments_mean']:.1f} ± {data['metrics']['segments_std']:.1f}")
                    print(f"  Time/sample: {data['metrics']['time_per_sample']:.1f}s")
            
            break
        
        # Wait before next check
        time.sleep(check_interval)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='baselines/advanced_full')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--total', type=int, default=77, help='Total expected samples')
    args = parser.parse_args()
    
    try:
        monitor_benchmark(
            results_dir=Path(args.dir),
            check_interval=args.interval,
            total_samples=args.total
        )
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
