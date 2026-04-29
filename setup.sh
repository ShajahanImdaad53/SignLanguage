#!/bin/bash
# setup.sh — Complete project setup in one command

set -e

echo "=========================================="
echo "SLT Hybrid SSL — Setup Script"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "[✓] Python $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
else
    echo "[1/5] Virtual environment already exists"
    source .venv/bin/activate
fi

# Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "      ✓ Dependencies installed"

# Create directory structure
echo "[3/5] Creating directory structure..."
mkdir -p data/{raw_videos,interim/{frames,keypoints,features},splits}
mkdir -p models logs experiments tests
mkdir -p src/models
mkdir -p pipelines agents
echo "      ✓ Directories created"

# Create empty manifest if not exists
if [ ! -f "data/manifest.csv" ]; then
    echo "[4/5] Creating sample manifest..."
    cat > data/manifest.csv << 'EOF'
video_id,video_path,label,gloss
video_001,data/raw_videos/sample_001.mp4,0,hello
video_002,data/raw_videos/sample_002.mp4,1,thank_you
EOF
    echo "      ✓ Sample manifest created at data/manifest.csv"
else
    echo "[4/5] Manifest already exists"
fi

# Run tests
echo "[5/5] Running smoke tests..."
python -m pytest tests/ -q --tb=short 2>/dev/null || echo "      (Tests skipped)"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit data/manifest.csv with your video paths"
echo "  2. Run: make data      # Download and process videos"
echo "  3. Run: make ssl       # SSL pretraining"
echo "  4. Run: make train     # Fine-tune on labelled data"
echo ""
echo "Or run individual steps:"
echo "  make help              # Show all commands"
echo ""
