# Heretic Dockerfile
# This Dockerfile builds a container image for running Heretic with full GPU support

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY config.default.toml ./

# Install Heretic and dependencies
# Install base dependencies first
RUN pip3 install --no-cache-dir \
    accelerate>=1.10.0 \
    datasets>=4.0.0 \
    hf-transfer>=0.1.9 \
    huggingface-hub>=0.34.4 \
    optuna>=4.5.0 \
    pydantic-settings>=2.10.1 \
    questionary>=2.1.1 \
    rich>=14.1.0 \
    transformers>=4.55.2

# Install vLLM separately for faster inference (optional but recommended)
RUN pip3 install --no-cache-dir vllm>=0.11.0 || echo "vLLM installation failed, continuing without it"

# Install Heretic in development mode
RUN pip3 install --no-cache-dir -e .

# Create cache directory
RUN mkdir -p /workspace/.cache/huggingface

# Set environment variable for faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Expose port for potential web interfaces (future use)
EXPOSE 8080

# Default command shows help
CMD ["heretic", "--help"]
