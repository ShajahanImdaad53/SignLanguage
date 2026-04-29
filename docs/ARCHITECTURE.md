# SLT Hybrid SSL — Architecture Documentation

Complete technical documentation of the Sign Language Translation system.

## System Overview

```
Raw Video
    ↓
[Frame Extraction] → Video frames (224×224, 25fps)
    ↓
[MediaPipe Extraction] → Hand + Body landmarks (258D vector per frame)
    ↓
[Fusion Layer] → Combined visual + keypoint features
    ↓
[Temporal Transformer] → Contextualised sequence representations
    ↓
[Classifier] → Sign gloss predictions
```

## 1. Data Pipeline

### 1.1 Video Download & Processing

**File:** `pipelines/download_and_process_video.py`

- Downloads from YouTube (via yt-dlp) or uses local files
- Extracts frames at 25 FPS (standard for sign language)
- Resizes to 224×224 (ImageNet standard)
- Saves as JPEG per frame

**Output:**
```
data/interim/frames/
├── video_001/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── video_002/
    └── ...
```

### 1.2 Keypoint Extraction

**File:** `pipelines/extract_keypoints.py`

Uses MediaPipe to extract hand and body landmarks:
- **Left hand:** 21 landmarks × 3 (x, y, z) = 63 dims
- **Right hand:** 21 landmarks × 3 = 63 dims
- **Body pose:** 33 landmarks × 3 = 99 dims
- **Total:** 225 dims, padded to 258 dims

**Output:**
```
data/interim/keypoints/
├── video_001.npy  (T, 258) float32
├── video_002.npy
└── ...
```

### 1.3 Data Splits

**File:** `pipelines/create_splits.py`

Creates stratified train/val/test splits:
- **Train:** 75% (learning)
- **Validation:** 12.5% (hyperparameter tuning)
- **Test:** 12.5% (final evaluation)

**Output:**
```
data/splits/
├── train.csv
├── val.csv
├── test.csv
└── label_map.json  {gloss_word: class_idx}
```

## 2. Model Architecture

### 2.1 Visual Backbone

**File:** `src/models/backbone.py`

```python
class VideoBackbone(ResNet50):
    input:  (B, T, C, H, W)  — video frames
    output: (B, T, 2048)     — per-frame CNN features
```

**Details:**
- Pretrained on ImageNet
- Frozen during SSL pretraining
- Unfrozen during supervised fine-tuning
- Efficient: frames processed in parallel batch

### 2.2 Keypoint Encoder

```python
class KeypointEncoder(nn.Module):
    input:  (B, T, 258)  — MediaPipe landmarks
    output: (B, T, 256)  — encoded keypoints
```

Simple 2-layer MLP with LayerNorm and GELU activation.

### 2.3 Fusion Layer

```python
class FusionLayer(nn.Module):
    visual:    (B, T, 2048) → (B, T, 256)
    keypoints: (B, T, 256)  → (B, T, 256)
    output:    (B, T, 256)  — fused features
```

**Three fusion strategies:**

1. **Concat** (default): `[visual, keypoints] → FC → hidden`
2. **Gated**: `gate(concat) * visual + (1 - gate) * keypoints`
3. **Attention**: `Attention(visual, keypoints, keypoints) + visual`

### 2.4 Temporal Transformer

```python
class TemporalTransformer(nn.Module):
    input:  (B, T, hidden_dim)  — fused features
    output: (B, T, hidden_dim)  — contextualised
```

**Architecture:**
- Sinusoidal positional encoding
- Multi-head self-attention (8 heads default)
- Feed-forward layers (4× expansion)
- Pre-layer norm (more stable)
- 4 transformer layers

### 2.5 Classification Head

```python
classifier = nn.Sequential(
    LayerNorm(hidden_dim),
    Linear(hidden_dim, vocab_size)
)
```

Mean-pools over time, then classifies to sign gloss.

## 3. Self-Supervised Learning

### 3.1 Masked Autoencoder (MAE)

**File:** `src/ssl_pretrain.py` — `_step_mae()`

```
Input frames: (B, T, C, H, W)
    ↓ [Mask 75% of frames]
Masked frames
    ↓ [Encoder → Transformer → Decoder]
Reconstructed features: (B, T, D)
    ↓ [MSE loss on masked positions only]
Loss = ||reconstructed[mask] - original[mask]||²
```

**Why this works:**
- Forces encoder to learn meaningful representations
- Decoder learns inverse mapping
- Masked positions are "blind spots" model must predict

### 3.2 Contrastive Learning (SimCLR)

**File:** `src/ssl_pretrain.py` — `_step_contrastive()`

```
View 1: Original frames (B, T, C, H, W)
View 2: Temporally shuffled frames
    ↓ [Each through encoder]
    ↓ [Pool over time]
Embeddings: z1, z2 (B, D)
    ↓ [L2 normalize]
    ↓ [NT-Xent loss: pull same, push different]
Loss = cross_entropy(sim_matrix, positive_pairs)
```

**Why this works:**
- Model learns rotation-invariant features
- Positive pairs (same video, augmented) attract
- Negative pairs (different videos) repel

## 4. Training Pipeline

### 4.1 SSL Pretraining

**File:** `src/ssl_pretrain.py`

```python
for epoch in range(30):
    for batch in train_loader:
        frames, keypoints, _ = batch
        
        # Freeze visual backbone, train transformer + decoder
        if method == "masked_autoencoder":
            mask = random_temporal_mask(B, T, 0.75)
            reconstructed = model(frames, keypoints, mask=mask, mode="pretrain")
            loss = MSE(reconstructed[mask], features[mask])
        else:  # contrastive
            z1 = model.encode(frames, keypoints)
            z2 = model.encode(frames_aug, keypoints_aug)
            loss = contrastive_loss(z1, z2)
        
        backward(loss)
```

**Outputs:**
- `models/ssl_pretrained.pt` — encoder weights

### 4.2 Supervised Fine-tuning

**File:** `src/trainer.py`

```python
# Load SSL weights
model.load_state_dict(torch.load("models/ssl_pretrained.pt"))
model.backbone.unfreeze()  # End-to-end training

for epoch in range(50):
    for batch in train_loader:
        frames, keypoints, labels = batch
        
        logits = model(frames, keypoints, mode="finetune")
        loss = CrossEntropyLoss(logits, labels)
        
        backward(loss)
        
        # Early stopping if no improvement for 10 epochs
        if val_acc not improved:
            patience_counter += 1
        if patience_counter >= 10:
            break
```

**Outputs:**
- `models/best_model.pt` — best checkpoint
- `logs/training_history.json` — metrics per epoch

## 5. Evaluation Metrics

**File:** `src/evaluate.py` and `src/evaluate_test.py`

### Metrics Computed

1. **Top-1 Accuracy**: % predictions exactly correct
2. **Top-5 Accuracy**: % true label in top-5 predictions
3. **Precision/Recall/F1**: Per-class metrics
4. **Confusion Matrix**: Class-wise confusions
5. **Macro/Weighted Averages**: Aggregate metrics

### Loss Functions

- **CrossEntropyLoss** with label smoothing (0.1)
  - Reduces overfitting by softening hard targets

## 6. Inference

### 6.1 Single Video Translation

**File:** `src/inference.py`

```python
translator = SignLanguageTranslator("models/best_model.pt", cfg)
result = translator.translate_video("path/to/video.mp4")
# Returns: {
#   "prediction": "hello",
#   "confidence": 0.95,
#   "top5": [...]
# }
```

### 6.2 Real-time Webcam

```python
translator.translate_webcam(duration=30)  # 30 seconds of webcam
```

### 6.3 Batch Inference

**File:** `pipelines/batch_inference.py`

```bash
python pipelines/batch_inference.py \
    --input_dir data/raw_videos/ \
    --output results.json \
    --workers 4
```

Parallel processing of multiple videos with progress tracking.

## 7. Export & Deployment

### 7.1 ONNX Export

**File:** `src/export_onnx.py`

Converts PyTorch → ONNX (interoperable, deployment-ready):

```bash
python src/export_onnx.py \
    --checkpoint models/best_model.pt \
    --output models/slt_model.onnx
```

**Advantages:**
- Run on CPU, GPU, TPU, mobile
- No PyTorch dependency
- Hardware optimization
- Framework agnostic

### 7.2 Docker Deployment

**File:** `Dockerfile`

```bash
docker build -t slt-hybrid-ssl .
docker run --gpus all -it slt-hybrid-ssl python src/inference.py --video video.mp4
```

## 8. Configuration System

### Hierarchy

```
base_config.yaml (all defaults)
    ↓ (overridden by)
task_config.yaml (data/train/ssl)
    ↓ (overridden by)
experiment_config.yaml (per-run)
    ↓ (overridden by)
CLI arguments
```

### Key Configs

**data_config.yaml:**
```yaml
video:
  fps: 25
  frame_size: [224, 224]
  max_frames: 64
augmentation:
  enabled: true
  random_crop: true
  horizontal_flip: true
```

**model_config.yaml:**
```yaml
model:
  backbone: "resnet50"
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  fusion: "concat"
```

**train_config.yaml:**
```yaml
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.0003
  early_stopping_patience: 10
  warmup_epochs: 5
```

## 9. Hyperparameter Tuning

### Grid Search

**File:** `automation/hyperparameter_search.py`

```bash
python automation/hyperparameter_search.py \
    --base configs/base_config.yaml \
    --workers 4
```

Searches over:
- Learning rates: [0.0001, 0.0003, 0.001]
- Batch sizes: [8, 16, 32]
- Hidden dims: [128, 256]
- Num layers: [2, 4]

Returns best hyperparameters with results.

## 10. Monitoring & Logging

### Training Agent

**File:** `agents/training_agent.py`

- Tracks best validation accuracy
- Early stopping logic
- Checkpoint management
- W&B integration (optional)

### Training Monitor

**File:** `automation/monitor_training.py`

Real-time monitoring:
```bash
python automation/monitor_training.py --log_dir logs/ --interval 60
```

Detects anomalies:
- NaN loss
- Loss explosion
- Low validation accuracy
- Training divergence

## 11. Performance Characteristics

### Memory Usage
- Per-video: ~200 MB (frames + keypoints)
- Model weights: ~200 MB (ResNet50 + Transformer)
- Batch size 16 on GPU: ~8 GB VRAM

### Inference Speed
- Single video (64 frames): ~2-5 seconds on GPU, ~30s on CPU
- Throughput: ~12-30 videos/sec on single GPU

### Model Size
- Saved checkpoint: ~200 MB
- ONNX export: ~180 MB
- Quantized (int8): ~50 MB

## 12. Extensibility

### Add New Backbone
```python
# In src/models/backbone.py
BACKBONE_DIMS["vit_base"] = 768
# Add ViT extraction logic
```

### Add New Fusion Method
```python
# In src/models/slt_model.py
elif fusion == "my_fusion":
    # Custom fusion logic
```

### Add Custom Loss Function
```python
# In src/trainer.py
criterion = MyCustomLoss()
loss = criterion(logits, labels)
```

---

**See code comments for detailed implementation notes.**
