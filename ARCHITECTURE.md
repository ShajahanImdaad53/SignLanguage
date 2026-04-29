# 🏗️ System Architecture

Detailed technical documentation of the SLT Hybrid SSL system.

## High-Level Overview

```
Raw Sign Videos
       ↓
[Frame Extraction]  → Extract frames @ 25 FPS
       ↓
[Keypoint Extraction] → MediaPipe hands + pose
       ↓
     Frames + Keypoints
       ↓
[Hybrid Model]
   ├─→ CNN Backbone (ResNet50)
   ├─→ Keypoint Encoder (MLP)
   ├─→ Fusion Layer (concat/attention)
   └─→ Temporal Transformer
       ↓
[SSL Pretraining] → Masked Autoencoder / Contrastive
       ↓
[Supervised Fine-tuning] → Sign Classification
       ↓
Sign Gloss (Text Output)
```

## Architecture Deep Dive

### 1. Data Flow

```
data/
├── raw_videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── interim/
│   ├── frames/
│   │   ├── video_001/
│   │   │   ├── frame_000000.jpg
│   │   │   ├── frame_000001.jpg
│   │   │   └── ...
│   │   └── video_002/
│   ├── keypoints/
│   │   ├── video_001.npy  (T × 258)
│   │   ├── video_002.npy
│   │   └── ...
│   └── features/
│       ├── video_001.npy  (T × D)
│       └── ...
└── splits/
    ├── train.csv  (video_id, video_path, keypoints_path, label)
    ├── val.csv
    ├── test.csv
    └── label_map.json  {gloss_str: class_id}
```

### 2. Neural Network Architecture

```
Input: frames (B, T, C, H, W) + keypoints (B, T, D)

┌─────────────────────────────────────────────────────┐
│                  Visual Stream                       │
├─────────────────────────────────────────────────────┤
│  frames (B, T, 3, 224, 224)                         │
│       ↓                                              │
│  VideoBackbone (ResNet50)                           │
│  Merge B,T → single forward → split back            │
│       ↓                                              │
│  visual_feats (B, T, 2048)                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                 Keypoint Stream                      │
├─────────────────────────────────────────────────────┤
│  keypoints (B, T, 258)                              │
│       ↓                                              │
│  KeypointEncoder (MLP: 258 → 256)                  │
│       ↓                                              │
│  kp_feats (B, T, 256)                               │
└─────────────────────────────────────────────────────┘

       ↓
┌─────────────────────────────────────────────────────┐
│             Fusion Layer                             │
├─────────────────────────────────────────────────────┤
│  Concat: [visual_feats, kp_feats] → (B, T, 2304)  │
│  Project: Linear(2304 → 256)                        │
│       ↓                                              │
│  fused (B, T, 256)                                  │
└─────────────────────────────────────────────────────┘

       ↓
┌─────────────────────────────────────────────────────┐
│        Temporal Transformer Encoder                  │
├─────────────────────────────────────────────────────┤
│  + Positional Encoding (sinusoidal)                 │
│  + 4 Transformer Blocks:                            │
│    - MultiheadAttention (8 heads, 256 dim)          │
│    - FeedForward (256 → 1024 → 256)                 │
│    - LayerNorm (pre-norm)                           │
│       ↓                                              │
│  encoded (B, T, 256)                                │
└─────────────────────────────────────────────────────┘

       ↓ (Fine-tuning mode)
┌─────────────────────────────────────────────────────┐
│            Pooling & Classification                  │
├─────────────────────────────────────────────────────┤
│  Mean Pool over time: (B, T, 256) → (B, 256)       │
│  Classification Head: Linear(256 → vocab_size)     │
│       ↓                                              │
│  logits (B, vocab_size)                             │
└─────────────────────────────────────────────────────┘
```

### 3. Self-Supervised Learning (SSL)

#### 3a. Masked Autoencoder (MAE)

```
Original Frames
    ↓
Extract CNN features (target): (B, T, 2048)
    ↓
Create random mask: (B, T) bool
    ↓
Apply mask → zero out masked positions
    ↓
Pass through encoder (shared)
    ↓
Decoder (MLP):
  Linear(256 → hidden_dim)
  GELU
  Linear(hidden_dim → 2048)  [reconstruct CNN features]
    ↓
MSE Loss = ||reconstructed[mask] - target[mask]||²
```

**Training Objective:**
Reconstruct visual features for masked-out frames.

#### 3b. Contrastive Learning (SimCLR-style)

```
Original Clip (B, T, 3, H, W)
    ↓
View 1: Temporal shift + augmentation
View 2: Temporal shift + augmentation
    ↓
Encode both through shared encoder
    ↓
Pool over time → (B, hidden_dim)
    ↓
Normalize (L2)
    ↓
NT-Xent Loss (InfoNCE):
  - Positive: (view1, view1), (view2, view2)
  - Negatives: all other pairs in batch
  - Temperature τ = 0.07 (controls sharpness)
    ↓
Loss = cross_entropy(similarities, labels)
```

**Training Objective:**
Make embeddings of the same clip similar, different clips dissimilar.

### 4. Training Modes

#### Mode 1: SSL Pretraining

```yaml
ssl:
  method: "masked_autoencoder"
  masking_ratio: 0.75
  epochs: 30
  lr: 0.0001
```

- **Backbone:** Frozen (CNN features are targets)
- **Trainable:** Transformer encoder + SSL decoder
- **Data:** Unlabelled videos only (no labels needed)
- **Loss:** MSE or contrastive

#### Mode 2: Supervised Fine-tuning

```yaml
training:
  epochs: 50
  lr: 0.0003
  warmup_epochs: 5
```

- **Backbone:** Unfrozen (can update CNN weights)
- **Trainable:** All layers
- **Data:** Labelled train/val splits
- **Loss:** Cross-entropy with label smoothing

### 5. Key Components

#### VideoBackbone
- Takes frames: (B, T, 3, 224, 224)
- Processes each frame independently through CNN
- Returns per-frame features: (B, T, feature_dim)

**Why per-frame?** Efficient batch processing; CNN is not designed for temporal data.

#### KeypointEncoder
- Takes MediaPipe landmarks: (B, T, 258)
  - 21 left hand landmarks × 3 (x,y,z)
  - 21 right hand landmarks × 3
  - 33 body/pose landmarks × 3
- Projects to hidden dim via 2-layer MLP
- Output: (B, T, hidden_dim)

#### FusionLayer
Three fusion strategies:

1. **Concat + Project:**
   - [visual, keypoint] → Dense → hidden_dim

2. **Gated:**
   - gate = sigmoid(Dense([visual, keypoint]))
   - output = gate * visual + (1 - gate) * keypoint

3. **Attention:**
   - visual as query, keypoint as key/value
   - output = MultiheadAttention(visual, keypoint, keypoint)

#### TemporalTransformer
- Positional encoding (sinusoidal)
- 4-layer transformer encoder
- 8 attention heads
- 256-dim hidden
- Pre-LayerNorm for stability

### 6. Config Hierarchy

```
base_config.yaml (all defaults)
    ↓
├─ data_config.yaml (override data settings)
├─ model_config.yaml (override model architecture)
├─ train_config.yaml (override training hyperparams)
└─ ssl_config.yaml (override SSL settings)
    ↓
experiments/exp01_lr_search.yaml (experiment-specific)
```

Any child config overrides parent on conflicting keys.

### 7. Data Loading Pipeline

```
CSV (train.csv)
    ↓
SLTDataset.__getitem__(idx)
    ├─ load_frames(video_id) → (T, H, W, 3) uint8
    ├─ apply transforms → (T, C, H, W) float32
    ├─ load_keypoints(path) → (T, 258) float32
    └─ return (frames, keypoints, label)
    ↓
DataLoader (batch_size=16, shuffle=True)
    ├─ Stack batch: frames (B, T, C, H, W)
    ├─ Stack batch: keypoints (B, T, D)
    └─ Stack labels: (B,)
```

### 8. Training Loop

```
for epoch in range(epochs):
    ├─ model.train()
    ├─ for batch in train_loader:
    │   ├─ frames, keypoints, labels = batch
    │   ├─ logits = model(frames, keypoints, mode="finetune")
    │   ├─ loss = CrossEntropyLoss(logits, labels)
    │   ├─ loss.backward()
    │   ├─ clip gradients
    │   └─ optimizer.step()
    │
    ├─ model.eval()
    ├─ for batch in val_loader:
    │   ├─ logits = model(...)
    │   ├─ compute accuracy, loss
    │   └─ track best model
    │
    └─ scheduler.step()
        (learning rate decay)
```

### 9. Inference Pipeline

```
new_video.mp4
    ↓
extract_frames() → (T, H, W, 3)
    ↓
extract_keypoints() → (T, 258)
    ↓
normalize & batch:
  frames → (1, T, C, H, W)
  keypoints → (1, T, 258)
    ↓
model.forward(..., mode="finetune")
    ↓
logits (1, vocab_size)
    ↓
argmax → class_id
    ↓
label_map[class_id] → gloss string
```

### 10. Performance Metrics

**Training Metrics:**
- Loss (training, validation)
- Accuracy (validation)
- Learning rate schedule

**Evaluation Metrics:**
- Top-1 Accuracy
- Top-5 Accuracy (if vocab > 5)
- Per-class accuracy
- Confusion matrix
- Precision, recall, F1-score

## Hardware Requirements

| Component | Min | Recommended | Ideal |
|-----------|-----|-------------|-------|
| GPU | 4GB | 8GB | 16GB+ |
| CPU Cores | 4 | 8 | 16+ |
| RAM | 16GB | 32GB | 64GB+ |
| Disk | 100GB | 500GB | 1TB+ |

## Computational Complexity

| Operation | Time | Memory |
|-----------|------|--------|
| Frame extract | ~10ms/frame | ~100MB |
| Keypoint extract | ~50ms/frame | ~50MB |
| Model inference | ~100-200ms | ~2GB |
| Full pipeline | ~5-10s | ~3GB |

## Distributed Training

For multi-GPU training:

```python
model = torch.nn.DataParallel(model)
# or DistributedDataParallel(model)
```

## Deployment Considerations

- **Inference latency:** ~100-200ms per video (CPU) or 50-100ms (GPU)
- **Throughput:** ~10-20 videos/sec on single GPU
- **Model size:** ~200MB checkpoint
- **Quantization:** 4-8x smaller with minimal accuracy loss

---

See individual module docstrings for implementation details.
