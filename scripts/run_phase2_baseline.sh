#!/bin/bash
# Phase 2 Baseline Training Launcher
# Automatically checks prerequisites and starts training

set -e

echo "=================================================="
echo "  Phase 2: SAM-RS Few-Shot Baseline Training"
echo "=================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python environment..."
PYTHON_PATH="/opt/miniconda3/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}✗ Python not found at $PYTHON_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $PYTHON_PATH${NC}"

# Check dataset
echo ""
echo "Checking LoveDA dataset..."
DATA_ROOT="./data/LoveDA"
if [ ! -d "$DATA_ROOT/Train/Urban/images_png" ] || [ ! -d "$DATA_ROOT/Val/Urban/images_png" ]; then
    echo -e "${RED}✗ LoveDA dataset not found${NC}"
    echo ""
    echo "Please download the dataset first:"
    echo "  1. Visit: https://zenodo.org/record/5706578"
    echo "  2. Download LoveDA.zip"
    echo "  3. Extract to ./data/LoveDA/"
    echo ""
    echo "Expected structure:"
    echo "  data/LoveDA/"
    echo "    ├── Train/Urban/images_png/"
    echo "    ├── Train/Rural/images_png/"
    echo "    ├── Val/Urban/images_png/"
    echo "    └── Val/Rural/images_png/"
    exit 1
fi
echo -e "${GREEN}✓ LoveDA dataset found${NC}"

# Check SAM checkpoint
echo ""
echo "Checking SAM checkpoint..."
SAM_CHECKPOINT="./checkpoints/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo -e "${YELLOW}⚠ SAM checkpoint not found${NC}"
    echo ""
    echo "Downloading SAM ViT-B checkpoint..."
    mkdir -p ./checkpoints

    if command -v wget &> /dev/null; then
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O "$SAM_CHECKPOINT"
    elif command -v curl &> /dev/null; then
        curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o "$SAM_CHECKPOINT"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please download manually:${NC}"
        echo "  URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        echo "  Save to: $SAM_CHECKPOINT"
        exit 1
    fi

    if [ -f "$SAM_CHECKPOINT" ]; then
        echo -e "${GREEN}✓ SAM checkpoint downloaded${NC}"
    else
        echo -e "${RED}✗ Failed to download SAM checkpoint${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ SAM checkpoint found${NC}"
fi

# Detect device
echo ""
echo "Detecting compute device..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEVICE="mps"
    echo -e "${GREEN}✓ Using Apple Silicon MPS${NC}"
elif command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo -e "${GREEN}✓ Using NVIDIA CUDA${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}⚠ Using CPU (training will be slow)${NC}"
fi

# Create output directories
mkdir -p ./checkpoints
mkdir -p ./runs
mkdir -p ./docs/experiments

# Training configuration
echo ""
echo "Training Configuration:"
echo "  Dataset: LoveDA"
echo "  Model: SAM-ViT-B + LoRA (rank=4)"
echo "  Few-shot: 5-way 5-shot"
echo "  Epochs: 50"
echo "  Batch size: 4"
echo "  Learning rate: 1e-4"
echo "  Device: $DEVICE"
echo ""

# Confirm start
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "=================================================="
echo "  Starting Training..."
echo "=================================================="
echo ""

# Start training
$PYTHON_PATH scripts/train.py \
    --dataRoot "$DATA_ROOT" \
    --samCheckpoint "$SAM_CHECKPOINT" \
    --samModelType vit_b \
    --loraRank 4 \
    --nPrompts 5 \
    --nWay 5 \
    --kShot 5 \
    --numEpochs 50 \
    --batchSize 4 \
    --lr 1e-4 \
    --device "$DEVICE" \
    --evalInterval 5 \
    --saveInterval 10 \
    --checkpointDir ./checkpoints \
    --logDir ./runs

echo ""
echo "=================================================="
echo "  Training Completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. View results: tensorboard --logdir=./runs"
echo "  2. Check best model: ./checkpoints/best_model.pth"
echo "  3. Record experiment results in docs/experiments/"
