# Image-to-3D Generator Docker Image
# Based on Trellis VGGT with BiRefNet background removal

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG GITHUB_USER
ENV GITHUB_USER=${GITHUB_USER}

ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}

ENV DEBIAN_FRONTEND=noninteractive

LABEL name="image-to-3d-gen" \
      maintainer="hiepnd11" \
      description="Image-to-3D generation using Trellis VGGT and BiRefNet"

# Install system dependencies
RUN apt update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update -y && \
    apt install -y wget gnupg git

# Add NVIDIA CUDA 12.8 repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# Install build tools and CUDA toolkit
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3.11 \
    python3-pip \
    python3-wheel \
    libjpeg-dev \
    zlib1g-dev \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Setup Python alternatives
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 120

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# GPU architectures: 8.9=Ada Lovelace, 9.0=Hopper, 10.0=Blackwell
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0;12.0"

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker cache)
COPY requirements.txt /app/requirements.txt

# Install Python packages (base image already has PyTorch 2.7.1)
RUN pip install packaging && \
    pip install -r requirements.txt

# Pre-cache DINOv2 for torch.hub to avoid GitHub token/rate-limit at runtime
RUN mkdir -p /root/.cache/torch/hub && \
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git /root/.cache/torch/hub/facebookresearch_dinov2_main

# (Optional but safer for offline) download DINOv2 weights during build
# Uncomment if you want fully offline runtime
# RUN python - <<'PY'\n# import torch\n# repo = '/root/.cache/torch/hub/facebookresearch_dinov2_main'\n# torch.hub.load(repo, 'dinov2_vits14', source='local', pretrained=True)\n# PY

# Copy rest of project files
COPY . /app

# Install mip-splatting (requires PyTorch)
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# Install Flash Attention and FlashInfer
RUN pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir && \
    pip install flashinfer-python==0.5.2 flashinfer-cubin==0.5.2 --no-build-isolation

# # HuggingFace Authentication (login if token provided)
# RUN if [ -n "$HF_TOKEN" ]; then \
#         echo "Logging into HuggingFace..." && \
#         pip install huggingface_hub --upgrade && \
#         huggingface-cli login --token $HF_TOKEN; \
#     else \
#         echo "WARNING: No HF_TOKEN provided. Gated models may not work."; \
#     fi

# Expose port
EXPOSE 10006

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:10006/health || exit 1

# Run server
CMD ["python", "serve.py", "--port", "10006"]

