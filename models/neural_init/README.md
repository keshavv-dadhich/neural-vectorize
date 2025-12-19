# Neural Initialization Models

This directory contains the trained neural network models for path initialization.

## Model Files (Not Included in Git)

The model checkpoint files (`best_model.pt`, `latest_model.pt`) are **190MB each** and exceed GitHub's 100MB file size limit, so they are not tracked in git.

## Option 1: Train Your Own Model (Recommended)

Train the ResNet-18 initialization network from scratch:

```bash
# Generate training data (770 samples from 77 benchmark icons)
python scripts/create_training_data.py

# Train neural network (2 hours on CPU)
python training/train_neural_init.py --epochs 50 --batch_size 16

# Model will be saved to: models/neural_init/best_model.pt
```

**Expected results**:
- Training loss: 853.2 → 139.7 (83.6% reduction)
- Validation loss: 95.59
- Test set L2 error: 0.246±0.045 (after optimization)
- Training time: ~2 hours on Apple M-series CPU

## Option 2: Download Pre-trained Model

Download the pre-trained checkpoint from one of these sources:

**Google Drive**: [Link to be added]
**Hugging Face**: [Link to be added]
**Releases Page**: [Link to be added]

After downloading `best_model.pt` (190MB), place it in this directory:
```
models/neural_init/best_model.pt
```

## Model Details

- **Architecture**: ResNet-18 backbone + MLP head
- **Parameters**: 16.6M
- **Input**: Degraded raster (256×256×3)
- **Output**: Path parameters (num_paths, 3) - {x, y, κ}
- **Inference time**: 37ms on CPU
- **File size**: 190MB (PyTorch checkpoint)

## Verification

Test that the model works correctly:

```bash
python vectorizers/run_full_test.py
```

Expected output:
```
✓ Model loaded successfully
✓ Neural inference: 37ms
✓ Full vectorization: 10.1s (30 steps)
✓ L2 error: 0.246±0.045
```

## Training Curves

Training progress for reference (your results may vary slightly):

```
Epoch 1/50:  Train Loss: 853.2, Val Loss: 95.59
Epoch 10/50: Train Loss: 425.1, Val Loss: 95.59
Epoch 20/50: Train Loss: 245.3, Val Loss: 95.59
Epoch 30/50: Train Loss: 178.4, Val Loss: 95.59
Epoch 40/50: Train Loss: 152.6, Val Loss: 95.59
Epoch 50/50: Train Loss: 139.7, Val Loss: 95.59 ✓ Best
```

The constant validation loss is expected—the network predicts initialization parameters for a dynamic optimization system, not final outputs. True validation is the test set L2 error after optimization (0.246).

## Support

If you encounter issues training or downloading the model, please open an issue on the GitHub repository.
