"""
Test neural architecture components to ensure they work correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

# Import neural components
from training.train_neural_init import RasterEncoder, ControlPointDecoder, NeuralInitializer


def test_encoder():
    """Test RasterEncoder forward pass."""
    print("Testing RasterEncoder...")
    
    encoder = RasterEncoder(latent_dim=512)
    
    # Create dummy input
    dummy_input = torch.randn(2, 1, 256, 256)  # Batch of 2
    
    # Forward pass
    latent = encoder(dummy_input)
    
    assert latent.shape == (2, 512), f"Expected (2, 512), got {latent.shape}"
    print(f"✅ Encoder output shape: {latent.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"✅ Encoder parameters: {num_params:,}")
    
    return encoder


def test_decoder():
    """Test ControlPointDecoder forward pass."""
    print("\nTesting ControlPointDecoder...")
    
    decoder = ControlPointDecoder(latent_dim=512, max_paths=10, max_points_per_path=50)
    
    # Create dummy latent
    dummy_latent = torch.randn(2, 512)
    
    # Forward pass
    points, masks = decoder(dummy_latent)
    
    assert points.shape == (2, 10, 50, 2), f"Expected (2, 10, 50, 2), got {points.shape}"
    assert masks.shape == (2, 10, 50), f"Expected (2, 10, 50), got {masks.shape}"
    
    print(f"✅ Decoder points shape: {points.shape}")
    print(f"✅ Decoder masks shape: {masks.shape}")
    
    # Check value ranges
    assert points.min() >= 0 and points.max() <= 256, "Points should be in [0, 256]"
    assert masks.min() >= 0 and masks.max() <= 1, "Masks should be in [0, 1]"
    
    print(f"✅ Points range: [{points.min():.2f}, {points.max():.2f}]")
    print(f"✅ Masks range: [{masks.min():.2f}, {masks.max():.2f}]")
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"✅ Decoder parameters: {num_params:,}")
    
    return decoder


def test_full_model():
    """Test full NeuralInitializer end-to-end."""
    print("\nTesting NeuralInitializer (full model)...")
    
    model = NeuralInitializer(latent_dim=512, max_paths=10, max_points_per_path=50)
    
    # Create dummy raster input
    dummy_raster = torch.randn(4, 1, 256, 256)  # Batch of 4
    
    # Forward pass
    points, masks = model(dummy_raster)
    
    assert points.shape == (4, 10, 50, 2), f"Expected (4, 10, 50, 2), got {points.shape}"
    assert masks.shape == (4, 10, 50), f"Expected (4, 10, 50), got {masks.shape}"
    
    print(f"✅ Full model points shape: {points.shape}")
    print(f"✅ Full model masks shape: {masks.shape}")
    
    # Count total parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total model parameters: {num_params:,}")
    print(f"   (Target: ~13M params)")
    
    # Test gradient flow
    loss = points.sum() + masks.sum()
    loss.backward()
    
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Some parameters don't have gradients!"
    print(f"✅ Gradient flow working")
    
    return model


def test_memory_usage():
    """Estimate memory usage during training."""
    print("\nEstimating memory usage...")
    
    model = NeuralInitializer()
    
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Model parameters: {param_memory:.2f} MB")
    
    # Batch size 8, input 256x256
    batch_size = 8
    input_memory = batch_size * 1 * 256 * 256 * 4 / 1024**2  # 4 bytes per float32
    print(f"Input batch (size {batch_size}): {input_memory:.2f} MB")
    
    # Output memory
    output_memory = batch_size * (10 * 50 * 2 + 10 * 50) * 4 / 1024**2
    print(f"Output batch: {output_memory:.2f} MB")
    
    # Rough gradient estimate (2x parameters)
    grad_memory = param_memory * 2
    print(f"Gradients (estimate): {grad_memory:.2f} MB")
    
    total = param_memory + input_memory + output_memory + grad_memory
    print(f"✅ Total estimated: {total:.2f} MB (~{total/1024:.2f} GB)")
    print(f"   Should fit on GPU with 4+ GB VRAM")


def test_inference_speed():
    """Test inference speed."""
    print("\nTesting inference speed...")
    
    model = NeuralInitializer()
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Time 100 inferences
    import time
    num_runs = 100
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    elapsed = time.time() - start
    
    per_sample = elapsed / num_runs * 1000  # ms
    print(f"✅ Inference time: {per_sample:.2f} ms per sample")
    print(f"   Throughput: {1000/per_sample:.1f} samples/second")


if __name__ == '__main__':
    print("="*70)
    print("NEURAL ARCHITECTURE TESTS")
    print("="*70)
    
    try:
        test_encoder()
        test_decoder()
        test_full_model()
        test_memory_usage()
        test_inference_speed()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nNeural architecture is ready for training.")
        print("Next steps:")
        print("1. Generate oracle training data (currently running)")
        print("2. Create training dataset: python3 scripts/create_training_data.py")
        print("3. Train model: python3 training/train_neural_init.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
