# ⚡ Quick Start Guide

Get your SLT Hybrid SSL system up and running in 10 minutes.

## Step 1: Clone & Setup (2 min)

```bash
git clone <your-repo> slt-hybrid-ssl
cd slt-hybrid-ssl
bash setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies
- Create required directories
- Run smoke tests

## Step 2: Prepare Your Data (2 min)

Edit `data/manifest.csv` with your sign language videos:

```csv
video_id,video_path,label,gloss
video_001,https://youtube.com/watch?v=...,0,hello
video_002,path/to/local/video.mp4,1,thank_you
video_003,path/to/local/video.mp4,2,goodbye
```

**Three ways to specify videos:**
- YouTube URL: `https://youtube.com/watch?v=dQw4w9WgXcQ`
- Local file: `/path/to/video.mp4`
- HTTP URL: `https://example.com/video.mp4`

## Step 3: Run Data Pipeline (3 min)

```bash
make data
```

Or manually:
```bash
python pipelines/run_data_pipeline.py \
    --manifest data/manifest.csv \
    --workers 4
```

This will:
1. Download videos (if URLs)
2. Extract frames at 25 FPS → `data/interim/frames/{video_id}/`
3. Extract MediaPipe keypoints → `data/interim/keypoints/{video_id}.npy`
4. Create train/val/test splits → `data/splits/{train,val,test}.csv`

**Expected output:**
```
[Pipeline] Downloaded & extracted: 5/5
[Pipeline] Keypoints: 5/5
[Pipeline] ✓ COMPLETE
```

## Step 4: SSL Pretraining (2 min setup, hours to run)

```bash
make ssl
```

Or manually:
```bash
python src/ssl_pretrain.py --config configs/ssl_config.yaml
```

This trains a masked autoencoder on unlabelled video:
- Masks 75% of frames
- Model tries to reconstruct visual features
- Creates `models/ssl_pretrained.pt` checkpoint

**To use contrastive learning instead**, edit `configs/ssl_config.yaml`:
```yaml
ssl:
  method: "contrastive"  # Changed from "masked_autoencoder"
```

**Estimated time:** 4-8 hours on single GPU

## Step 5: Fine-tune on Labelled Data (2 min setup, 1-2 hours to run)

```bash
make train
```

Or manually:
```bash
python src/trainer.py --config configs/train_config.yaml
```

This:
1. Loads SSL-pretrained weights
2. Unfreezes backbone
3. Trains end-to-end on labelled data
4. Saves best model → `models/best_model.pt`

**Live monitoring (TensorBoard):**
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006

**Estimated time:** 1-2 hours on single GPU

## Step 6: Evaluate on Test Set

```bash
python src/evaluate_test.py --checkpoint models/best_model.pt
```

This reports:
- Test accuracy
- Per-class metrics
- Confusion matrix (optional)

---

## 🎯 Quick Experiments

### Experiment 1: Different Learning Rate

```bash
# Create experiment config
cp configs/train_config.yaml configs/experiments/exp_lr0001.yaml

# Edit learning rate
sed -i 's/learning_rate: 0.0003/learning_rate: 0.0001/' configs/experiments/exp_lr0001.yaml

# Train
python src/trainer.py --config configs/experiments/exp_lr0001.yaml
```

### Experiment 2: Smaller Model

```bash
# Edit model_config.yaml
sed -i 's/hidden_dim: 256/hidden_dim: 128/' configs/model_config.yaml

# Train
make train
```

### Experiment 3: Different SSL Method

```bash
# Edit ssl_config.yaml
sed -i 's/method: "masked_autoencoder"/method: "contrastive"/' configs/ssl_config.yaml

# Pretrain
make ssl

# Fine-tune
make train
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Dataset loading
- Model forward pass
- Config loading
- Transforms

---

## 🐛 Troubleshooting

### "No module named mediapipe"
```bash
pip install mediapipe
```

### "CUDA out of memory"
Reduce batch size in `configs/train_config.yaml`:
```yaml
training:
  batch_size: 8  # Changed from 16
```

### "Video download failed"
Try a different video URL or use a local file path instead.

### "No frames found"
Make sure `data/interim/frames/` directory was created and contains subdirectories.

---

## 📊 Monitoring & Logs

### TensorBoard
```bash
tensorboard --logdir logs/
```

### View Training History
```bash
cat logs/training_history.json | python -m json.tool
```

### Check Experiment Config
```bash
cat logs/train_config_snapshot.yaml
```

---

## 🚀 Next Steps

1. **Improve Data Quality**: Collect more labelled sign videos
2. **Hyperparameter Tuning**: Adjust LR, batch size, model dim
3. **Custom Backbone**: Try `efficientnet_b0` or `vit` in model_config.yaml
4. **Deployment**: Export to ONNX for production inference
5. **W&B Tracking**: Set `wandb: true` in train_config.yaml

---

## 📚 Full Documentation

- See `README.md` for overview
- See `src/` for implementation details
- See `configs/*.yaml` for all configuration options
- Run `make help` for all available commands

---

## ⏱️ Typical Timeline

| Step | Time | GPU Memory | Disk Space |
|------|------|-----------|-----------|
| Setup | 5 min | - | 1 GB |
| Data Pipeline | 10-20 min | - | 50-200 GB |
| SSL Pretraining | 4-8 hours | 8 GB | 5 GB |
| Fine-tuning | 1-2 hours | 8 GB | 2 GB |
| **Total** | **~12 hours** | **8 GB** | **60 GB** |

(Estimated for 100 videos on single GPU)

---

Happy training! 🎉
