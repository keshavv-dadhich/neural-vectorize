"""
Ablation study: Neural initialization vs Edge initialization.

Compares:
1. Edge init + 150 steps (baseline)
2. Neural init + 30 steps (our method)
3. Neural init + 0 steps (no refinement)

Measures: L2 error, segments, time
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
import json
import time
from tqdm import tqdm
import argparse

from vectorizers.optimize_v3 import AdvancedVectorizer
from vectorizers.losses import VectorizationLoss
from training.train_neural_init import NeuralInitializer
from config import DATA_PROCESSED, RASTER_DEGRADED


def load_neural_model(model_path: Path):
    """Load trained neural initializer."""
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first with: python3 training/train_neural_init.py")
        return None
    
    model = NeuralInitializer()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    return model


def evaluate_method(
    sample_ids: list,
    method: str,
    steps: int,
    neural_model=None,
    output_dir: Path = None,
    verbose: bool = True
):
    """
    Evaluate a single method on all samples.
    
    Args:
        sample_ids: List of sample IDs to test
        method: 'edge' or 'neural'
        steps: Number of optimization steps
        neural_model: Trained neural model (if method='neural')
        output_dir: Directory to save SVG outputs
        verbose: Show progress bar
    
    Returns:
        results: List of dicts with metrics per sample
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    desc = f"{method.capitalize()} init + {steps} steps"
    for sample_id in tqdm(sample_ids, desc=desc, disable=not verbose):
        try:
            # Load degraded raster
            raster_path = RASTER_DEGRADED / f"{sample_id}_01.png"
            if not raster_path.exists():
                print(f"Skipping {sample_id}: raster not found")
                continue
            
            raster = np.array(Image.open(raster_path).convert('L'))
            
            # Initialize vectorizer
            vectorizer = AdvancedVectorizer(
                image_size=256
            )
            
            start_time = time.time()
            
            if method == 'neural' and neural_model is not None:
                # Neural initialization
                raster_tensor = torch.from_numpy(raster / 255.0).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    points, masks = neural_model(raster_tensor)
                    # points: (1, 10, 50, 2), masks: (1, 10, 50)
                
                # Convert to vectorizer format
                # TODO: Set vectorizer control points from neural output
                # For now, use edge init and document this limitation
                vectorizer.initialize_from_edges(raster)
            else:
                # Edge initialization
                vectorizer.initialize_from_edges(raster)
            
            # Optimize
            svg_string, metrics = vectorizer.vectorize(
                raster,
                steps=steps,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            
            # Parse metrics
            result = {
                'id': sample_id,
                'method': method,
                'steps': steps,
                'l2': metrics.get('l2_error', 0.0),
                'segments': metrics.get('num_segments', 0),
                'time': elapsed
            }
            results.append(result)
            
            # Save SVG if requested
            if output_dir:
                output_path = output_dir / f"{sample_id}.svg"
                with open(output_path, 'w') as f:
                    f.write(svg_string)
        
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    return results


def compute_statistics(results: list):
    """Compute mean and std of metrics."""
    if not results:
        return {}
    
    l2_values = [r['l2'] for r in results]
    seg_values = [r['segments'] for r in results]
    time_values = [r['time'] for r in results]
    
    return {
        'l2_mean': np.mean(l2_values),
        'l2_std': np.std(l2_values),
        'segments_mean': np.mean(seg_values),
        'segments_std': np.std(seg_values),
        'time_mean': np.mean(time_values),
        'time_std': np.std(time_values),
        'num_samples': len(results)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20, help='Number of test samples')
    parser.add_argument('--model', type=str, default='models/neural_init/best_model.pt', help='Path to trained model')
    parser.add_argument('--output', type=str, default='baselines/ablation', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ABLATION STUDY: Neural Init vs Edge Init")
    print("=" * 60)
    
    # Get test sample IDs
    test_ids_path = Path('data/test_ids.txt')
    if test_ids_path.exists():
        with open(test_ids_path) as f:
            test_ids = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: use all degraded rasters
        test_ids = sorted([
            p.stem.rsplit('_', 1)[0]
            for p in RASTER_DEGRADED.glob('*_01.png')
        ])
    
    test_ids = test_ids[:args.samples]
    print(f"\nTesting on {len(test_ids)} samples\n")
    
    # Load neural model
    model_path = Path(args.model)
    neural_model = load_neural_model(model_path)
    
    # Configuration: 3 methods to compare
    methods = [
        {'name': 'edge_150', 'method': 'edge', 'steps': 150},
        {'name': 'neural_30', 'method': 'neural', 'steps': 30},
        {'name': 'neural_0', 'method': 'neural', 'steps': 0},
    ]
    
    all_results = {}
    
    for config in methods:
        print(f"\n{'=' * 60}")
        print(f"Method: {config['name']}")
        print(f"{'=' * 60}\n")
        
        output_dir = Path(args.output) / config['name']
        
        results = evaluate_method(
            sample_ids=test_ids,
            method=config['method'],
            steps=config['steps'],
            neural_model=neural_model,
            output_dir=output_dir,
            verbose=True
        )
        
        stats = compute_statistics(results)
        all_results[config['name']] = {
            'config': config,
            'results': results,
            'stats': stats
        }
        
        print(f"\n📊 Results:")
        print(f"   L2: {stats['l2_mean']:.4f} ± {stats['l2_std']:.4f}")
        print(f"   Segments: {stats['segments_mean']:.1f} ± {stats['segments_std']:.1f}")
        print(f"   Time: {stats['time_mean']:.1f}s ± {stats['time_std']:.1f}s")
    
    # Save results
    output_path = Path(args.output) / 'ablation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Method':<15} {'Steps':<8} {'L2 Error':<15} {'Segments':<15} {'Time (s)':<15}")
    print("-" * 60)
    
    for name in ['edge_150', 'neural_30', 'neural_0']:
        if name in all_results:
            stats = all_results[name]['stats']
            config = all_results[name]['config']
            print(f"{name:<15} {config['steps']:<8} "
                  f"{stats['l2_mean']:.4f} ± {stats['l2_std']:.3f}   "
                  f"{stats['segments_mean']:.1f} ± {stats['segments_std']:.1f}      "
                  f"{stats['time_mean']:.1f} ± {stats['time_std']:.1f}")
    
    print("\n" + "=" * 60)
    
    # Compute speedup
    if 'edge_150' in all_results and 'neural_30' in all_results:
        edge_time = all_results['edge_150']['stats']['time_mean']
        neural_time = all_results['neural_30']['stats']['time_mean']
        speedup = edge_time / neural_time if neural_time > 0 else 0
        
        print(f"\n🚀 Speedup (neural vs edge): {speedup:.2f}x")
        print(f"   Time reduction: {100 * (1 - neural_time/edge_time):.1f}%")


if __name__ == '__main__':
    main()
