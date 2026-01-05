# Chatterbox TTS Server Dockerfile
# Base: NVIDIA TensorRT with Python 3
#
# Build:
#   docker build -t chatterbox-tts:latest -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8080:8080 chatterbox-tts:latest

# =============================================================================
# Base Image
# =============================================================================
FROM nvcr.io/nvidia/tensorrt:24.06-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    # Utilities
    curl \
    wget \
    git \
    # Redis tools (for debugging)
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Python Environment
# =============================================================================
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Application Code
# =============================================================================
# Copy server code
COPY server/ ./server/
COPY scripts/ ./scripts/

# Copy configs (if they exist)
COPY configs/ ./configs/ 2>/dev/null || true

# =============================================================================
# Model Files (Optional - can mount at runtime)
# =============================================================================
# Uncomment to include models in image (increases image size significantly)
# COPY models/ ./models/
# COPY engines/ ./engines/

# Create directories for mounted volumes
RUN mkdir -p /app/models /app/engines /app/speakers

# =============================================================================
# Environment Variables
# =============================================================================
# GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Chatterbox
ENV CHATTERBOX_CFG_SCALE=0.5

# Server
ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=info
ENV DEVICE=cuda
ENV USE_TENSORRT=true

# Redis (if using external)
ENV REDIS_URL=redis://redis:6379
ENV ENABLE_SPEAKER_CACHE=true

# =============================================================================
# Health Check
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# =============================================================================
# Expose Port
# =============================================================================
EXPOSE 8080

# =============================================================================
# Entrypoint
# =============================================================================
# Default command - run the server
CMD ["python", "-m", "server.main"]

# Alternative: Use uvicorn directly with more options
# CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
