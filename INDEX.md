# 📑 Complete Project Index

**SLT Hybrid SSL** — Sign Language Translation with Self-Supervised Learning

---

## 🎯 Start Here

### First Time Users
1. **FINAL_README.txt** (19 KB) ← Read this first!
   - Complete overview
   - Quick start guide
   - Checklist before starting
   
2. **QUICKSTART.md** (5 KB)
   - 10-minute setup
   - Essential commands only
   - For impatient users

### Full Setup
3. **README.md** (7.5 KB)
   - Project overview
   - Features & capabilities
   - Full usage guide

### Understanding the System
4. **ARCHITECTURE.md** (12 KB)
   - System design deep-dive
   - Network architecture
   - SSL mechanisms explained

### Production Deployment
5. **DEPLOYMENT.md** (6.7 KB)
   - Docker setup
   - FastAPI server
   - Cloud deployment options

### Complete Reference
6. **PROJECT_SUMMARY.md** (13 KB)
   - File-by-file breakdown
   - Feature checklist
   - FAQ & troubleshooting

### Download & Setup
7. **DOWNLOAD_SETUP.md** (12 KB)
   - Installation options
   - System requirements
   - Troubleshooting guide

---

## 📂 Directory Structure

### `src/` — Core ML Code
```
src/
├── trainer.py                  # Supervised fine-tuning training loop
├── ssl_pretrain.py             # SSL pretraining (MAE + contrastive)
├── inference.py                # Inference on new videos
├── dataset.py                  # PyTorch Dataset & DataLoaders
├── evaluate.py                 # Validation metrics
├── evaluate_test.py            # Test evaluation with detailed metrics
│
├── models/
│   ├── slt_model.py           # Full hybrid SLT architecture
│   ├── backbone.py            # CNN feature extractor (ResNet50)
│   └── transformer.py         # Temporal transformer encoder
│
└── utils/
    ├── config.py              # YAML config merging & loading
    ├── logger.py              # Centralized logging
    └── seed.py                # Reproducibility helpers
```

**What to read:**
- Start: `models/slt_model.py` (architecture overview)
- Then: `trainer.py` (training loop)
- Deep dive: `ssl_pretrain.py` (SSL mechanisms)

### `pipelines/` — Data Processing
```
pipelines/
├── run_data_pipeline.py        # Master orchestrator
├── download_and_process_video.py  # Download + frame extraction
├── extract_keypoints.py        # MediaPipe keypoint extraction
└── create_splits.py            # Train/val/test splitting
```

**What to read:**
- Quick overview: `run_data_pipeline.py`
- Implementation details: individual scripts

### `configs/` — Configuration Files
```
configs/
├── base_config.yaml            # ALL defaults (start here)
├── data_config.yaml            # Data-specific settings
├── model_config.yaml           # Architecture hyperparams
├── train_config.yaml           # Training hyperparams
├── ssl_config.yaml             # SSL settings
└── experiments/
    └── exp01_lr_search.yaml    # Per-experiment overrides
```

**How to use:**
- Copy `base_config.yaml` as template
- Edit specific config for your experiment
- No code changes needed!

### `app/` — Deployment
```
app/
└── main.py                     # FastAPI inference server
```

**What to read:**
- See `DEPLOYMENT.md` first
- Then: `main.py` for endpoint implementation

### `tests/` — Unit Tests
```
tests/
├── test_dataset.py             # Dataset loading tests
└── test_model.py               # Model architecture tests
```

**How to run:**
```bash
pytest tests/ -v
pytest tests/ --cov=src/
```

### `notebooks/` — Jupyter Examples
```
notebooks/
├── 01_data_exploration.ipynb   # Dataset analysis
└── 02_inference_demo.ipynb     # Model usage examples
```

### `data/` — Your Dataset
```
data/
├── manifest.csv                # List your videos here (CREATE THIS)
├── raw_videos/                 # Downloaded videos go here
├── interim/
│   ├── frames/                 # Extracted frames
│   ├── keypoints/              # MediaPipe keypoints (.npy)
│   └── features/               # Precomputed features
└── splits/
    ├── train.csv               # Training set
    ├── val.csv                 # Validation set
    ├── test.csv                # Test set
    └── label_map.json          # Class ID mapping
```

### `models/` — Saved Checkpoints
```
models/
├── ssl_pretrained.pt           # SSL pretraining checkpoint
└── best_model.pt               # Best fine-tuned model
```

### `logs/` — Training Logs
```
logs/
├── training_history.json       # Loss/accuracy history
├── test_results.json           # Test metrics
└── *.log                        # Detailed logs
```

---

## 🎯 Common Tasks

### Task 1: Understand the Architecture
**Read in order:**
1. `ARCHITECTURE.md` (overview)
2. `src/models/slt_model.py` (code)
3. `src/models/backbone.py` (visual features)
4. `src/models/transformer.py` (temporal modeling)

**Time:** 1-2 hours

### Task 2: Prepare Your Data
**Read:**
1. `QUICKSTART.md` (data format)
2. `pipelines/run_data_pipeline.py` (what happens)
3. `data/manifest.csv` (example)

**Do:**
```bash
# Create your manifest
cat > data/manifest.csv << EOF
video_id,video_path,label,gloss
video_001,path/to/video.mp4,0,hello
video_002,path/to/video.mp4,1,thank_you
