#!/usr/bin/env bash
set -e

# Detect CUDA version from nvidia-smi, fall back to CPU
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | tr -d '.')
    if   [ "$CUDA_VER" -ge 124 ] 2>/dev/null; then WHEEL="cu124"
    elif [ "$CUDA_VER" -ge 121 ] 2>/dev/null; then WHEEL="cu121"
    elif [ "$CUDA_VER" -ge 118 ] 2>/dev/null; then WHEEL="cu118"
    else WHEEL="cu118"
    fi
    echo "[install] CUDA $CUDA_VER detected — using $WHEEL wheels"
else
    WHEEL="cpu"
    echo "[install] No GPU detected — installing CPU-only torch"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${WHEEL}"

# Step 1: install torch (latest stable for this CUDA version)
echo "[install] Installing torch..."
pip install torch --index-url "$TORCH_INDEX"

# Step 2: detect the exact torch version that was installed
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "[install] torch ${TORCH_VER} installed"

PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VER}+${WHEEL}.html"

# Step 3: torch-geometric (torch-scatter/torch-sparse not needed for PyG 2.4+)
echo "[install] Installing torch-geometric..."
pip install torch-geometric

# Step 4: everything else
echo "[install] Installing remaining dependencies..."
pip install \
    duckdb==1.5.0 \
    pyarrow==23.0.1 \
    pandas==2.2.3 \
    pyyaml==6.0.2 \
    click==8.1.8 \
    hdbscan==0.8.40 \
    scikit-learn==1.6.1 \
    umap-learn==0.5.7 \
    matplotlib==3.10.1 \
    seaborn==0.13.2 \
    tqdm==4.67.1 \
    rich==14.0.0 \
    scipy==1.15.3 \
    pytest==8.3.5

echo "[install] Done. torch==${TORCH_VER}+${WHEEL}"
