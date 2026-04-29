# Troubleshooting Guide

Common issues and how to fix them.

## Installation Issues

### "ModuleNotFoundError: No module named 'mediapipe'"

```bash
pip install mediapipe
```

### "CUDA out of memory"

The model doesn't fit in your GPU. Solutions:

1. **Reduce batch size** (fastest fix):
```yaml
# configs/train_config.yaml
training:
  batch_size: 8  # reduced from 16
```

2. **Reduce hidden dimensions**:
```yaml
# configs/model_config.yaml
model:
  hidden_dim: 128  # reduced from 256
```

3. **Use smaller backbone**:
```yaml
model:
  backbone: "resnet18"  # instead of resnet50
```

4. **Enable gradient checkpointing** (if available):
```python
# In src/trainer.py
model.backbone.gradient_checkpointing_enable()
```

### "AttributeError: 'DotDict' object has no attribute 'xxx'"

Missing config key. Check:
```bash
cat configs/base_config.yaml | grep xxx
# If not there, add it to the config
```

---

## Data Pipeline Issues

### "FileNotFoundError: Video not found"

Make sure your `data/manifest.csv` has correct paths:
```csv
video_id,video_path,label,gloss
video_001,/full/path/to/video.mp4,0,hello
# Use absolute paths, not relative
```

### "No frames extracted from video"

Possible causes:

1. **Invalid video file**: Test with `ffmpeg`:
```bash
ffmpeg -i path/to/video.mp4 -f null -
```

2. **Video codec not supported**: Convert to H.264:
```bash
ffmpeg -i input.mp4 -vcodec libx264 output.mp4
```

3. **Empty video**: Check duration:
```bash
ffprobe -v error -show_format -show_streams path/to/video.mp4 | grep duration
```

### "Keypoints extraction failed"

MediaPipe not detecting landmarks. Try:

1. **Lower confidence threshold**:
```yaml
# configs/data_config.yaml
keypoints:
  confidence_threshold: 0.3  # from 0.5
```

2. **Check video quality**: Ensure hands are visible and well-lit

3. **Check frame size**: MediaPipe works best with 224×224

---

## Training Issues

### "NaN loss detected"

Loss went to NaN. Solutions:

1. **Reduce learning rate**:
```yaml
training:
  learning_rate: 0.0001  # reduced from 0.0003
```

2. **Add gradient clipping** (should already be enabled):
```python
# In src/trainer.py
gradient_clip = cfg.training.gradient_clip  # 1.0
```

3. **Check for bad data**: Look for extreme values in train.csv

4. **Use gradient checkpointing** to stabilize:
```python
model.backbone.gradient_checkpointing_enable()
```

### "Training loss not decreasing"

Model not learning. Possible causes:

1. **Learning rate too low**:
```yaml
training:
  learning_rate: 0.001  # increase from 0.0003
```

2. **Data imbalance**: Check class distribution:
```python
import pandas as pd
df = pd.read_csv("data/splits/train.csv")
print(df["label"].value_counts())
# If very imbalanced, use class weights or oversampling
```

3. **Backbone frozen**: Check if frozen:
```python
# In src/trainer.py
model.backbone.unfreeze()
```

4. **Wrong labels**: Verify label mapping:
```bash
cat data/splits/label_map.json
# Compare with your actual glosses
```

### "Validation accuracy stuck at random (~10% for 10 classes)"

Model predicting random class. Likely causes:

1. **Labels not loading correctly**: Check CSV format
2. **Model not trained on labeled data**: Ensure train.csv exists
3. **Class mismatch**: Vocabulary size mismatch

```python
# Debug in src/trainer.py
print(f"Num classes from logits: {logits.shape[1]}")
print(f"Expected from config: {cfg.model.vocab_size}")
```

### "Early stopping triggered too early"

Model stops before convergence. Fix:

1. **Increase patience**:
```yaml
training:
  early_stopping_patience: 20  # from 10
```

2. **Reduce warmup**:
```yaml
training:
  warmup_epochs: 2  # from 5
```

3. **Check val/train loss ratio**: If val loss >> train loss, model is overfitting:

```python
# Increase regularization
training:
  weight_decay: 0.1  # from 0.01
  dropout: 0.3       # increase model.dropout
```

---

## Inference Issues

### "RuntimeError: CUDA out of memory during inference"

Inference shouldn't use much memory. Try:

```python
# In src/inference.py
with torch.no_grad():
    torch.cuda.empty_cache()  # Clear cache
    output = model(frames, keypoints)
```

Or switch to CPU:
```bash
python src/inference.py --video video.mp4 --device cpu
```

### "Model prediction always the same class"

Model collapsed. Causes:

1. **Checkpoint corrupt**: Retrain from scratch
2. **Label mapping issue**: Check label_map.json
3. **Input not normalized**: Verify mean/std in config

---

## Performance Issues

### "Training is very slow"

Check:

1. **num_workers** too low:
```yaml
training:
  num_workers: 8  # increase from 4
```

2. **pin_memory** disabled:
```yaml
training:
  pin_memory: true
```

3. **Disk I/O bottleneck**: Check data read speed:
```bash
# Time how long to read one epoch
time python -c "
from src.dataset import build_dataloaders
from src.utils.config import load_config
cfg = load_config('configs/data_config.yaml', base_path='configs/base_config.yaml')
loaders = build_dataloaders(cfg)
for _ in loaders['train']:
    pass
"
```

If slow, use SSD for data or reduce dataset size.

### "GPU not fully utilized"

Check:

1. **Batch size too small**: Increase to 32+ if GPU has memory
2. **Mixed precision disabled**: Enable:
```yaml
training:
  mixed_precision: true
```

3. **Data loading slow**: Check num_workers and pin_memory above

---

## Evaluation Issues

### "test.csv not found"

Create test set:
```bash
python pipelines/create_splits.py --manifest data/manifest.csv
```

### "Evaluation metrics show 0% accuracy"

Likely a label mismatch. Debug:

```python
# In src/evaluate_test.py
# Add debug output:
print(f"Predictions shape: {preds.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique predictions: {np.unique(preds)}")
print(f"Unique labels: {np.unique(labels)}")
```

---

## Deployment Issues

### "ONNX model doesn't load"

Ensure ONNX package installed:
```bash
pip install onnx onnxruntime
```

Test loading:
```python
import onnx
import onnxruntime as rt

model = onnx.load("models/slt_model.onnx")
onnx.checker.check_model(model)  # Should not error

sess = rt.InferenceSession("models/slt_model.onnx")
```

### "ONNX inference gives different results"

Likely numerical precision issue. Fix:

1. **Use float32** (not float16) in export
2. **Normalize inputs identically** in ONNX as in PyTorch
3. **Compare on same GPU/CPU** to isolate issues

---

## Monitoring Issues

### "Training monitor not detecting metrics"

Check file paths:
```bash
ls -la logs/training_history.json
# Should exist after first epoch
```

If missing, training process hasn't written yet. Wait for first epoch to complete.

### "Hyperparameter search stuck"

Likely one experiment hanging. Solutions:

1. **Reduce timeout** in `automation/hyperparameter_search.py`:
```python
timeout=3600  # 1 hour instead of 2
```

2. **Run with fewer workers**:
```bash
python automation/hyperparameter_search.py --workers 1
```

3. **Kill hanging processes**:
```bash
ps aux | grep trainer.py | grep -v grep | awk '{print $2}' | xargs kill
```

---

## General Debugging Tips

### Enable verbose logging:
```python
# At top of any script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check config at runtime:
```python
cfg = load_config(...)
print(cfg)  # Print entire config tree
```

### Validate inputs manually:
```python
import torch
frames = torch.randn(1, 64, 3, 224, 224)
keypoints = torch.randn(1, 64, 258)
model(frames, keypoints)  # Should not error
```

### Profile GPU memory:
```bash
python -c "
import torch
from src.models import SLTModel
model = SLTModel(cfg).cuda()
print(torch.cuda.memory_allocated() / 1e9, 'GB')
"
```

### Profile execution time:
```python
import time
start = time.time()
output = model(frames, keypoints)
print(f"Inference took {time.time() - start:.2f}s")
```

---

## Still Stuck?

1. **Check logs**: `tail -f logs/*.log`
2. **Review code comments**: Each file has detailed inline documentation
3. **Check the README**: Common patterns documented there
4. **Check GitHub issues**: May be a known issue with workaround
5. **Create minimal reproduction**: Isolate the problem with a simple script

**Example minimal repro script:**

```python
#!/usr/bin/env python
"""Minimal script to test dataset loading."""
import sys
from src.utils.config import load_config
from src.dataset import build_dataloaders

cfg = load_config("configs/train_config.yaml", base_path="configs/base_config.yaml")
loaders = build_dataloaders(cfg)

train_loader = loaders["train"]
frames, keypoints, labels = next(iter(train_loader))
print(f"✓ Data loaded successfully")
print(f"  Frames: {frames.shape}")
print(f"  Keypoints: {keypoints.shape}")
print(f"  Labels: {labels.shape}")
```

Run: `python debug_minimal.py` — if this fails, data pipeline is broken. If this succeeds, problem is in training code.
