"""
Complete optimization-based vectorizer runner.

Combines all phases:
- Phase A: Initialize from edges
- Phase B: Optimize with gradient descent
- Phase C: Simplify (merge collinear segments, drop tiny paths)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vectorizers.initialize import EdgeInitializer
from vectorizers.optimize import DifferentiableVectorizer
from vectorizers.simplify import simplify_svg
from config import STROKE_WIDTH


def vectorize(
    raster_path: Path,
    output_svg_path: Path,
    num_opt_steps: int = 300,
    simplify_after: bool = True,
    verbose: bool = True
) -> Path:
    """
    Complete raster-to-vector pipeline.
    
    Args:
        raster_path: Input PNG
        output_svg_path: Output SVG path
        num_opt_steps: Number of optimization iterations
        simplify_after: Whether to apply post-optimization simplification
        verbose: Print progress
        
    Returns:
        Path to final SVG
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"VECTORIZING: {raster_path.name}")
        print(f"{'='*60}\n")
    
    # Phase A: Initialize from edges
    if verbose:
        print("Phase A: Edge detection → Bézier initialization...")
    
    initializer = EdgeInitializer(
        canny_low=50,
        canny_high=150,
        epsilon_factor=0.002,
        min_contour_length=20
    )
    
    temp_init_path = output_svg_path.parent / f"{output_svg_path.stem}_init.svg"
    initializer.initialize_from_raster(
        raster_path,
        temp_init_path,
        stroke_width=STROKE_WIDTH
    )
    
    if verbose:
        print(f"  ✓ Initialized SVG: {temp_init_path.name}\n")
    
    # Phase B: Optimize
    if verbose:
        print("Phase B: Differentiable optimization...")
    
    vectorizer = DifferentiableVectorizer(image_size=256, device='cpu')
    
    temp_opt_path = output_svg_path.parent / f"{output_svg_path.stem}_opt.svg"
    vectorizer.optimize(
        svg_path=temp_init_path,
        target_image_path=raster_path,
        output_svg_path=temp_opt_path,
        num_steps=num_opt_steps,
        lr=0.001,
        lambda_complexity=0.001,
        verbose=verbose
    )
    
    if verbose:
        print(f"  ✓ Optimized SVG: {temp_opt_path.name}\n")
    
    # Phase C: Simplify
    if simplify_after:
        if verbose:
            print("Phase C: Post-optimization simplification...")
        
        simplify_svg(
            temp_opt_path,
            output_svg_path,
            merge_threshold=0.01,  # Merge segments within 1% of image size
            min_path_length=0.02,  # Drop paths shorter than 2% of image size
            verbose=verbose
        )
        
        if verbose:
            print(f"  ✓ Simplified SVG: {output_svg_path.name}\n")
        
        # Clean up temp files
        temp_init_path.unlink()
        temp_opt_path.unlink()
    else:
        # Just rename optimized to final
        temp_opt_path.rename(output_svg_path)
        temp_init_path.unlink()
    
    if verbose:
        print(f"{'='*60}")
        print(f"✅ COMPLETE: {output_svg_path.name}")
        print(f"{'='*60}\n")
    
    return output_svg_path


def test_complete_pipeline():
    """Test the complete vectorization pipeline."""
    from config import RASTER_PERFECT
    
    # Get test raster
    test_rasters = list(RASTER_PERFECT.glob('*.png'))[:3]  # Test on 3 images
    
    output_dir = Path('outputs/vectorized_test')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for raster in test_rasters:
        output_svg = output_dir / f"{raster.stem}.svg"
        vectorize(
            raster,
            output_svg,
            num_opt_steps=200,  # Faster for testing
            simplify_after=True,
            verbose=True
        )


if __name__ == '__main__':
    test_complete_pipeline()
