#!/bin/bash

# Image-to-3D Generator Environment Setup
# Based on 404-base-miner architecture

set -e

echo "========================================="
echo "  Image-to-3D Generator Setup"
echo "========================================="

# Check for Conda installation
if [ -z "$(which conda)" ]; then
    echo "[INFO] Conda not found. Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init --all
    
    # Install system tools
    echo "[INFO] Installing system tools..."
    apt update
    apt install -y nano vim npm
    npm install -g pm2@6.0.12
else
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Source Conda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create Conda environment
echo "[INFO] Creating Conda environment from conda_env.yml..."
conda env create -f conda_env.yml
conda activate image-to-3d-gen
conda info --env

# Setup CUDA environment hooks
echo "[INFO] Setting up CUDA environment hooks..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<'SH'
export CUDA_ENV_PREFIX="$CONDA_PREFIX"
# Save originals if not saved
if [ -z "${CUDA_SAVED_CUDA_HOME+x}" ]; then export CUDA_SAVED_CUDA_HOME="${CUDA_HOME:-}"; fi
if [ -z "${CUDA_SAVED_CUDA_PATH+x}" ]; then export CUDA_SAVED_CUDA_PATH="${CUDA_PATH:-}"; fi
if [ -z "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then export CUDA_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"; fi

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda.sh" <<'SH'
if [ -n "${CUDA_SAVED_CUDA_HOME+x}" ]; then export CUDA_HOME="${CUDA_SAVED_CUDA_HOME}"; else unset CUDA_HOME; fi
if [ -n "${CUDA_SAVED_CUDA_PATH+x}" ]; then export CUDA_PATH="${CUDA_SAVED_CUDA_PATH}"; else unset CUDA_PATH; fi
if [ -n "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then export LD_LIBRARY_PATH="${CUDA_SAVED_LD_LIBRARY_PATH}"; fi

if [ -n "$CUDA_ENV_PREFIX" ]; then
    PATH=":${PATH:-}:"; PATH="${PATH//:$CUDA_ENV_PREFIX\/bin:/:}"; PATH="${PATH#:}"; PATH="${PATH%:}"; export PATH
    if [ -z "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then
        LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"; LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:$CUDA_ENV_PREFIX\/lib:/:}"; LD_LIBRARY_PATH="${LD_LIBRARY_PATH#:}"; LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"; export LD_LIBRARY_PATH
    fi
fi
unset CUDA_ENV_PREFIX CUDA_SAVED_CUDA_HOME CUDA_SAVED_CUDA_PATH CUDA_SAVED_LD_LIBRARY_PATH
SH

# Force activate hooks to set CUDA paths
conda deactivate
conda activate image-to-3d-gen

# Set CUDA architecture list for compilation
# 8.9 = Ada Lovelace (RTX 4090, 5090)
# 9.0 = Hopper (H100, H200)
# 10.0 = Blackwell (B100, GB200)
export TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0;12.0"

# Install PyTorch FIRST (required by mip-splatting)
echo "[INFO] Installing PyTorch 2.7.1 with CUDA 12.8..."
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1

# Install mip-splatting (requires PyTorch)
echo "[INFO] Installing mip-splatting..."
mkdir -p /tmp/extensions
if [ ! -d "/tmp/extensions/mip-splatting" ]; then
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
fi
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# Install Python dependencies
echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt

# Install Flash Attention (critical for performance)
echo "[INFO] Installing Flash Attention..."
pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir

# Install FlashInfer (for attention optimization)
echo "[INFO] Installing FlashInfer..."
pip install flashinfer-python==0.5.2 flashinfer-cubin==0.5.2 --no-build-isolation --no-cache-dir

# # HuggingFace Authentication
# if [ -f .env ]; then
#     echo "[INFO] Loading HuggingFace token from .env file..."
#     export $(grep -v '^#' .env | grep HF_TOKEN | xargs)
# fi

# if [ -n "$HF_TOKEN" ]; then
#     echo "[INFO] Logging into HuggingFace with provided token..."
#     huggingface-cli login --token "$HF_TOKEN"
#     echo "[INFO] HuggingFace authentication successful!"
# else
#     echo "[WARN] No HF_TOKEN found. Gated models (like briaai/RMBG-2.0) may not work."
#     echo "[WARN] Create .env file with your token or set HF_TOKEN environment variable."
# fi

# Store the path of the Conda interpreter for PM2
CONDA_INTERPRETER_PATH=$(which python)

# Generate PM2 config file
echo "[INFO] Generating PM2 config file..."
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'image-to-3d-gen',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--port 8094 --model Stable-X/trellis-vggt-v0-2'
  }]
};
EOF

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate image-to-3d-gen"
echo ""
echo "To start the server:"
echo "  pm2 start generation.config.js"
echo ""
echo "To test the API:"
echo "  curl -X POST http://0.0.0.0:8094/generate -F 'prompt_image_file=@image.png' > model.ply"
echo ""

