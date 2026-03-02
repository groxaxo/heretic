# Docker Guide for Heretic

This guide explains how to use Heretic with Docker for containerized deployment.

## Prerequisites

- Docker Engine with NVIDIA GPU support ([nvidia-docker](https://github.com/NVIDIA/nvidia-docker))
- NVIDIA GPU with CUDA support
- Sufficient disk space for models (can be 10GB+ per model)

## Quick Start

### Using Docker Run

```bash
# Build the image
docker build -t heretic .

# Run Heretic with GPU support
docker run --gpus all -it heretic heretic MODEL_NAME

# Example with a specific model
docker run --gpus all -it heretic heretic Qwen/Qwen3-4B-Instruct-2507
```

### Using Docker Compose (Recommended)

Docker Compose provides easier volume management and configuration:

```bash
# Run with Docker Compose
docker-compose run heretic heretic MODEL_NAME

# Example
docker-compose run heretic heretic Qwen/Qwen3-4B-Instruct-2507
```

## Volume Mounts

The Docker Compose configuration includes persistent volumes:

### Hugging Face Cache

Models are cached in a Docker volume to avoid re-downloading:
```yaml
volumes:
  - huggingface-cache:/workspace/.cache/huggingface
```

To clear the cache:
```bash
docker volume rm heretic_huggingface-cache
```

### Model Output Directory

Save models to a local directory that's accessible outside the container:
```yaml
volumes:
  - ./models:/workspace/models
```

Save models to `/workspace/models` inside the container to access them in `./models` on your host.

## Configuration

### Using a Custom Config File

Mount your custom configuration:

**Docker Run:**
```bash
docker run --gpus all -it \
  -v $(pwd)/config.toml:/workspace/config.toml \
  heretic heretic MODEL_NAME
```

**Docker Compose:**
```yaml
volumes:
  - ./config.toml:/workspace/config.toml
```

### Environment Variables

Set environment variables for configuration:

**Docker Run:**
```bash
docker run --gpus all -it \
  -e HERETIC_N_TRIALS=100 \
  -e HERETIC_QUANTIZATION=bnb_4bit \
  heretic heretic MODEL_NAME
```

**Docker Compose:**
```yaml
environment:
  - HERETIC_N_TRIALS=100
  - HERETIC_QUANTIZATION=bnb_4bit
```

See `config.default.toml` for all available configuration options.

## Hugging Face Authentication

For private models, provide your Hugging Face token:

**Docker Run:**
```bash
docker run --gpus all -it \
  -e HUGGINGFACE_TOKEN=your_token_here \
  heretic heretic private/model-name
```

**Docker Compose:**
```yaml
environment:
  - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
```

Then set the environment variable on your host:
```bash
export HUGGINGFACE_TOKEN=your_token_here
docker-compose run heretic heretic private/model-name
```

## GPU Configuration

### Using Specific GPUs

Run on specific GPU(s):
```bash
docker run --gpus '"device=0,1"' -it heretic heretic MODEL_NAME
```

### Memory Limits

You can limit per-device memory usage with `max_memory` if needed:
```bash
docker run --gpus all -it \
  -e HERETIC_MAX_MEMORY={"0":"20GB","cpu":"64GB"} \
  heretic heretic MODEL_NAME
```

## Common Issues

### Out of Memory Errors

1. **Enable 4-bit quantization:**
   ```bash
   -e HERETIC_QUANTIZATION=bnb_4bit
   ```

2. **Reduce model size:** Use a smaller model variant when possible.

3. **Limit batch size:**
   ```bash
   -e HERETIC_BATCH_SIZE=1
   ```

### Model Not Found

Ensure you have internet connectivity and, for private models, have set `HUGGINGFACE_TOKEN`.

### GPU Not Available

Verify nvidia-docker is installed:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Advanced Usage

### Interactive Mode

Start an interactive shell in the container:
```bash
docker run --gpus all -it heretic /bin/bash
```

Then run Heretic commands inside:
```bash
heretic MODEL_NAME
```

### Custom Docker Build

Modify the Dockerfile for your needs, then rebuild:
```bash
docker build -t heretic:custom .
```

### Multi-Stage Builds

For smaller images, consider a multi-stage build (requires modifying Dockerfile).

## Cleaning Up

Remove the container and images:
```bash
# Remove stopped containers
docker-compose down

# Remove images
docker rmi heretic

# Remove volumes (WARNING: deletes cached models)
docker volume prune
```

## Performance Tips

1. **Enable 4-bit quantization** (`HERETIC_QUANTIZATION=bnb_4bit`) for lower VRAM use
2. **Keep models cached** using the persistent volume
3. **Use SSD storage** for Docker volumes for faster I/O
4. **Allocate sufficient shared memory** if needed: `--shm-size=16g`

## Example Workflows

### Decensor and Save a Model

```bash
docker-compose run heretic heretic meta-llama/Llama-3.1-8B-Instruct

# Follow the interactive prompts to save to /workspace/models
# The model will be accessible in ./models on your host
```

### Evaluate a Saved Model

```bash
docker-compose run heretic heretic \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --evaluate-model /workspace/models/my-saved-model
```

## Support

For issues specific to Docker deployment, please check:
1. Docker logs: `docker logs <container_id>`
2. GPU availability: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. [Project issues](https://github.com/p-e-w/heretic/issues)
