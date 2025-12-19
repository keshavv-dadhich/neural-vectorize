"""
Ablation study: Compare impact of different loss terms.

Tests 5 configurations:
1. Raster only (baseline)
2. Raster + Edge
3. Raster + Edge + Curvature
4. Raster + Edge + Curvature + Intersection
5. Full (all 5 terms)
"""

import sys
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import json
import time

import config
from vectorizers.optimize_v3 import AdvancedVectorizer

RASTER_PERFECT = config.RASTER_PERFECT
TEST_IDS = config.TEST_IDS
BASELINES = config.BASELINES


# Define configurations to test
CONFIGS = {
    'raster_only': {
        'lambda_raster': 1.0,
        'lambda_edge': 0.0,
        'lambda_curvature': 0.0,
        'lambda_intersection': 0.0,
        'lambda_complexity': 0.0
    },
    'raster_edge': {
        'lambda_raster': 1.0,
        'lambda_edge': 0.5,
        'lambda_curvature': 0.0,
        'lambda_intersection': 0.0,
        'lambda_complexity': 0.0
    },
    'raster_edge_curv': {
        'lambda_raster': 1.0,
        'lambda_edge': 0.5,
        'lambda_curvature': 0.1,
        'lambda_intersection': 0.0,
        'lambda_complexity': 0.0
    },
    'raster_edge_curv_inter': {
        'lambda_raster': 1.0,
        'lambda_edge': 0.5,
        'lambda_curvature': 0.1,
        'lambda_intersection': 0.3,
        'lambda_complexity': 0.0
    },
    'full': {
        'lambda_raster': 1.0,
        'lambda_edge': 0.5,
        'lambda_curvature': 0.1,
        'lambda_intersection': 0.3,
        'lambda_complexity': 0.005
    }
}


def run_ablation_study(
    num_samples: int = 10,
    num_steps: int = 150,
    output_dir: Path = Path('baselines/ablation_loss_terms')
):
    """
    Run ablation study on loss terms.
    
    Args:
        num_samples: Number of test samples to use
        num_steps: Optimization steps per configuration
        output_dir: Where to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test IDs
    test_ids = TEST_IDS.read_text().strip().split('\n')[:num_samples]
    
    print(f"Running ablation study on {len(test_ids)} samples")
    print(f"Configurations: {list(CONFIGS.keys())}")
    print(f"Steps per config: {num_steps}")
    print(f"Output directory: {output_dir}")
    print()
    
    all_results = {}
    
    # Test each configuration
    for config_name, config_params in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Testing configuration: {config_name}")
        print(f"{'='*70}")
        print(f"Parameters: {config_params}")
        
        # Initialize vectorizer with this configuration
        vectorizer = AdvancedVectorizer(
            image_size=256,
            **config_params
        )
        
        config_results = []
        config_failed = []
        
        for test_id in tqdm(test_ids, desc=f"{config_name}"):
            try:
                # Load target
                raster_path = RASTER_PERFECT / f"{test_id}_base.png"
                if not raster_path.exists():
                    config_failed.append(test_id)
                    continue
                
                target = np.array(Image.open(raster_path).convert('L')) / 255.0
                
                # Use optimization_full as initialization
                init_svg_path = BASELINES / 'optimization_full' / f"{test_id}.svg"
                if not init_svg_path.exists():
                    init_svg_path = BASELINES / 'potrace' / f"{test_id}.svg"
                    if not init_svg_path.exists():
                        config_failed.append(test_id)
                        continue
                
                # Optimize
                output_svg_path = output_dir / config_name / f"{test_id}.svg"
                output_svg_path.parent.mkdir(parents=True, exist_ok=True)
                
                start_time = time.time()
                vectorizer.optimize(
                    init_svg_path,
                    target,
                    output_svg_path,
                    num_steps=num_steps,
                    learning_rate=0.01,
                    verbose=False
                )
                elapsed = time.time() - start_time
                
                # Compute L2 metric
                from PIL import ImageDraw
                import xml.etree.ElementTree as ET
                import re
                
                # Simple rasterization for L2
                img = Image.new('L', (256, 256), 255)
                draw = ImageDraw.Draw(img)
                
                tree = ET.parse(output_svg_path)
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
                
                config_results.append({
                    'id': test_id,
                    'l2': float(l2),
                    'segments': num_segments,
                    'time': elapsed
                })
                
            except Exception as e:
                print(f"Error processing {test_id} with {config_name}: {str(e)}")
                config_failed.append(test_id)
                continue
        
        # Compute aggregates for this configuration
        if config_results:
            avg_l2 = np.mean([r['l2'] for r in config_results])
            avg_segments = np.mean([r['segments'] for r in config_results])
            avg_time = np.mean([r['time'] for r in config_results])
            
            all_results[config_name] = {
                'config': config_params,
                'num_samples': len(config_results),
                'num_failed': len(config_failed),
                'metrics': {
                    'l2_mean': float(avg_l2),
                    'l2_std': float(np.std([r['l2'] for r in config_results])),
                    'segments_mean': float(avg_segments),
                    'segments_std': float(np.std([r['segments'] for r in config_results])),
                    'time_mean': float(avg_time)
                },
                'results': config_results
            }
            
            print(f"\n{config_name} results:")
            print(f"  L2: {avg_l2:.4f} ± {np.std([r['l2'] for r in config_results]):.4f}")
            print(f"  Segments: {avg_segments:.1f} ± {np.std([r['segments'] for r in config_results]):.1f}")
            print(f"  Time: {avg_time:.1f}s")
    
    # Save all results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print(f"\n\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Configuration':<30} {'L2':<15} {'Segments':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    for config_name in CONFIGS.keys():
        if config_name in all_results:
            metrics = all_results[config_name]['metrics']
            print(f"{config_name:<30} "
                  f"{metrics['l2_mean']:<15.4f} "
                  f"{metrics['segments_mean']:<15.1f} "
                  f"{metrics['time_mean']:<10.1f}")
    
    # Analyze edge alignment impact
    if 'raster_only' in all_results and 'raster_edge' in all_results:
        baseline_l2 = all_results['raster_only']['metrics']['l2_mean']
        edge_l2 = all_results['raster_edge']['metrics']['l2_mean']
        improvement = (1 - edge_l2 / baseline_l2) * 100
        
        print(f"\n✅ Key Finding: Adding edge alignment improves L2 by {improvement:.1f}%")
    
    print(f"\nResults saved to: {output_dir / 'ablation_results.json'}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--steps', type=int, default=150, help='Optimization steps')
    parser.add_argument('--output', type=str, default='baselines/ablation_loss_terms')
    args = parser.parse_args()
    
    run_ablation_study(
        num_samples=args.samples,
        num_steps=args.steps,
        output_dir=Path(args.output)
    )
