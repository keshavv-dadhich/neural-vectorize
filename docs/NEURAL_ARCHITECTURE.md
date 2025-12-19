# Neural Initialization Architecture Design

**Goal**: Learn to approximate optimization oracle in 30 steps (vs 150)

**Expected Result**: 80% time reduction, <10% accuracy loss

---

## Architecture Overview

```
Degraded Raster (256×256) 
    ↓
[ResNet-18 Encoder]  # Pretrained on ImageNet
    ↓
Latent Vector (512-d)
    ↓
[MLP Decoder]
    ↓
Control Points (10 paths × 50 points × 2 coords)
+ Validity Masks (10 paths × 50 points)
```

---

## Detailed Design

### Input: Degraded Raster
- **Shape**: (B, 1, 256, 256)
- **Format**: Grayscale, normalized [0, 1]
- **Source**: Augmented training set (605 samples × 10 variants = 6,050 images)

### Encoder: ResNet-18
```python
class RasterEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet-18, replace first conv for grayscale
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Modify first conv: 3 → 1 channel
        self.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        
    def forward(self, x):
        # x: (B, 1, 256, 256)
        latent = self.features(x)  # (B, 512, 1, 1)
        return latent.squeeze()    # (B, 512)
```

**Why ResNet-18?**
- Pretrained features transfer well to edge detection
- Lightweight: 11M params, fast forward pass
- Well-validated architecture

### Decoder: MLP
```python
class ControlPointDecoder(nn.Module):
    def __init__(self, latent_dim=512, num_paths=10, points_per_path=50):
        super().__init__()
        self.num_paths = num_paths
        self.points_per_path = points_per_path
        
        # MLP layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output heads
        self.points_head = nn.Linear(2048, num_paths * points_per_path * 2)
        self.mask_head = nn.Linear(2048, num_paths * points_per_path)
        
    def forward(self, latent):
        # latent: (B, 512)
        features = self.fc(latent)  # (B, 2048)
        
        # Predict control points
        points = self.points_head(features)  # (B, num_paths * points_per_path * 2)
        points = points.view(-1, self.num_paths, self.points_per_path, 2)
        points = torch.sigmoid(points)  # Normalize to [0, 1]
        
        # Predict validity masks
        masks = self.mask_head(features)  # (B, num_paths * points_per_path)
        masks = masks.view(-1, self.num_paths, self.points_per_path)
        masks = torch.sigmoid(masks)  # Probability each point is valid
        
        return points, masks
```

**Why MLP?**
- Simple, interpretable
- Direct mapping: latent → control points
- Validity masks handle variable path lengths

### Full Model
```python
class NeuralInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RasterEncoder()
        self.decoder = ControlPointDecoder()
        
    def forward(self, raster):
        # raster: (B, 1, 256, 256)
        latent = self.encoder(raster)  # (B, 512)
        points, masks = self.decoder(latent)
        return points, masks
```

**Total Parameters**: ~13M (11M encoder + 2M decoder)

---

## Training Setup

### Dataset
```python
class VectorizationDataset(Dataset):
    def __init__(self, split='train'):
        self.rasters = []  # Degraded rasters (605 × 10 = 6,050)
        self.targets = []  # Oracle-optimized control points
        
    def __getitem__(self, idx):
        raster = self.rasters[idx]  # (1, 256, 256)
        
        # Load oracle SVG, extract control points
        svg_path = f"baselines/advanced_pilot/{self.ids[idx]}.svg"
        points, masks = self.parse_svg_to_tensors(svg_path)
        
        return raster, points, masks
```

**Key Decision**: Train on **oracle outputs**, not ground truth!
- Oracle = optimized with multi-term loss
- Ground truth = original clean SVGs
- **Why oracle?** Because that's what we want to approximate

### Loss Function
```python
def loss_fn(pred_points, pred_masks, target_points, target_masks):
    # MSE on control points (only valid points)
    point_loss = F.mse_loss(
        pred_points * target_masks.unsqueeze(-1),
        target_points * target_masks.unsqueeze(-1)
    )
    
    # BCE on validity masks
    mask_loss = F.binary_cross_entropy(pred_masks, target_masks)
    
    return point_loss + 0.1 * mask_loss
```

### Training Loop
```python
model = NeuralInitializer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

for epoch in range(50):
    for raster, points, masks in train_loader:
        pred_points, pred_masks = model(raster)
        loss = loss_fn(pred_points, pred_masks, points, masks)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validate
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)
    
    # Save best
    if val_loss < best_loss:
        torch.save(model.state_dict(), 'best_model.pt')
```

**Hyperparameters**:
- Epochs: 50
- Batch size: 8 (fits in 8GB GPU)
- Learning rate: 1e-4 (Adam)
- Scheduler: ReduceLROnPlateau (patience=5)
- Early stopping: patience=10

---

## Inference Pipeline

### Neural Initialization → Optimization Refinement

```python
def vectorize_with_neural_init(raster_path, output_svg_path):
    # Step 1: Neural initialization (fast)
    model = NeuralInitializer()
    model.load_state_dict(torch.load('best_model.pt'))
    
    raster = load_raster(raster_path)
    pred_points, pred_masks = model(raster)
    
    # Convert to SVG
    init_svg = tensors_to_svg(pred_points, pred_masks)
    
    # Step 2: Optimization refinement (30 steps vs 150)
    vectorizer = AdvancedVectorizer()
    vectorizer.optimize(
        svg_path=init_svg,
        target_image=raster,
        output_path=output_svg_path,
        num_steps=30  # Reduced from 150!
    )
```

---

## Expected Results

### Ablation: Edge Init vs Neural Init

| Init Method | Opt Steps | Final L2 | Time | Visual Quality |
|-------------|-----------|----------|------|----------------|
| **Edge init** | 150 | 0.070 | 150s | ✅ Oracle |
| **Neural init** | 30 | ~0.075 | 30s | ✅ Near-oracle |

**Target**:
- Time: 150s → 30s (80% reduction)
- Accuracy: L2=0.070 → 0.075 (<10% degradation)
- Visual: Still edge-aligned, smooth, interpretable

### Why 30 Steps?

Neural init provides **warm start** closer to optimum:
- Edge init: Random → requires 150 steps
- Neural init: Near-optimal → requires only 30 steps refinement

**Hypothesis**: Neural network learns to approximate first 120 steps of optimization.

---

## Implementation Plan

### Week 1: Data Preparation
- [ ] Run optimization oracle on all 605 training samples
- [ ] Extract control points from oracle SVGs
- [ ] Create PyTorch dataset (raster → points, masks)
- [ ] Validate data loader

### Week 2: Training
- [ ] Implement ResNet-18 encoder
- [ ] Implement MLP decoder
- [ ] Training loop with validation
- [ ] Hyperparameter tuning
- [ ] Save best model

### Week 3: Evaluation
- [ ] Inference on 77 test samples
- [ ] Benchmark: Neural init + 30-step opt
- [ ] Compare to edge init + 150-step opt
- [ ] Visual quality assessment

### Week 4: Analysis & Write-up
- [ ] Ablation studies (10, 20, 30, 50 steps)
- [ ] Failure analysis
- [ ] Paper draft

---

## Success Criteria

**Minimal (workshop paper)**:
- ✅ 30-step neural init matches 150-step edge init
- ✅ <10% accuracy loss (L2: 0.070 → 0.075)
- ✅ 80% time reduction (150s → 30s)

**Strong (conference paper)**:
- 20-step neural init matches edge init
- <5% accuracy loss
- Generalizes to out-of-distribution shapes

---

## Risk Mitigation

**Risk 1**: Neural init doesn't converge  
**Mitigation**: Pretrained ResNet features, large batch sizes, gradient clipping

**Risk 2**: 30 steps not enough  
**Mitigation**: Ablate 10/20/30/50 steps, find sweet spot

**Risk 3**: Oracle outputs too diverse  
**Mitigation**: Start with 100 similar samples (e.g., all crowns), then scale

---

## Future Extensions

1. **Attention Mechanisms**: Cross-attention between encoder and decoder
2. **Autoregressive Decoding**: Generate paths sequentially (like SketchRNN)
3. **Diffusion Models**: Iterative refinement of control points
4. **Multi-Resolution**: Predict coarse → fine control points

---

_Document Status: Architecture Designed_  
_Next Step: Implement Dataset Creation (Week 1)_  
_Target: 30-step neural initialization by end of Month 1_
