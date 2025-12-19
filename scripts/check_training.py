"""Quick check of neural training progress."""

import re
from pathlib import Path

log_file = Path('logs/neural_training.log')

if not log_file.exists():
    print("❌ Training log not found")
    exit(1)

with open(log_file) as f:
    lines = f.readlines()

# Find epochs
epochs = []
for line in lines:
    if 'Epoch' in line and '/' in line and '%' not in line:
        epochs.append(line.strip())

# Find validation losses
val_losses = []
for line in lines:
    if 'Val   - Point' in line:
        val_losses.append(line.strip())

# Find best model saves
best_saves = []
for line in lines:
    if 'Saved best model' in line:
        best_saves.append(line.strip())

print("=" * 60)
print("NEURAL TRAINING PROGRESS")
print("=" * 60)

if epochs:
    print(f"\nCurrent epoch: {epochs[-1]}")
else:
    print("\nNo epoch info yet")

if val_losses:
    print(f"\nRecent validation losses (last 5):")
    for vl in val_losses[-5:]:
        print(f"  {vl}")
else:
    print("\nNo validation data yet")

if best_saves:
    print(f"\nBest model saves: {len(best_saves)}")
    print(f"  Last: {best_saves[-1]}")
else:
    print("\nNo best model saved yet")

print("\n" + "=" * 60)
