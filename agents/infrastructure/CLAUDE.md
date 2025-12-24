# Infrastructure Agent

## Role

Setup and maintain the GPU infrastructure for Chatterbox TTS optimization on AWS G5.

## Responsibilities

1. AWS G5 instance management
2. CUDA/cuDNN/TensorRT installation
3. Conda environment setup
4. Docker configuration with NVIDIA runtime
5. Network and security setup
6. Model downloads from HuggingFace

## Server Specifications

```yaml
Instance: G5.xlarge
  GPU: NVIDIA A10G (24GB GDDR6X)
  vCPUs: 8 (AMD EPYC)
  RAM: 16 GB
  Storage: 250 GB NVMe SSD
  AMI: Deep Learning AMI Neuron (Ubuntu 22.04) 20240816
```

## Required Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| CUDA | 12.4+ | GPU compute |
| cuDNN | 8.9+ | Deep learning primitives |
| TensorRT | 10.x | Inference optimization |
| Python | 3.11 | Runtime |
| Conda/Mamba | Latest | Environment management |
| Docker | 24.x+ | Containerization |
| NVIDIA Container Toolkit | Latest | GPU-enabled containers |
| Redis | 7.x | Speaker embedding cache |

## Setup Scripts

All setup scripts should be created in `/scripts/setup/`:

### install_cuda.sh
```bash
#!/bin/bash
# Verify CUDA installation on Deep Learning AMI

# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Verify PyTorch CUDA access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### install_tensorrt.sh
```bash
#!/bin/bash
# Install TensorRT for inference optimization

pip install tensorrt
pip install onnxruntime-gpu

# Verify installation
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
python -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"
python -c "import onnxruntime; print(f'Available providers: {onnxruntime.get_available_providers()}')"
```

### setup_conda.sh
```bash
#!/bin/bash
# Create and configure conda environment

conda create -n chatterbox python=3.11 -y
conda activate chatterbox

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Chatterbox TTS
pip install chatterbox-tts

# Install additional dependencies
pip install \
    fastapi uvicorn redis aioredis \
    numpy scipy librosa soundfile \
    huggingface_hub transformers \
    onnxruntime-gpu tensorrt \
    pytest pytest-asyncio locust
```

## Common Tasks

### Task: Verify GPU Setup
```bash
# Full GPU verification
nvidia-smi
nvcc --version
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### Task: Install TensorRT
```bash
pip install tensorrt
python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"
```

### Task: Setup Conda Environment
```bash
conda create -n chatterbox python=3.11 -y
conda activate chatterbox
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install chatterbox-tts
```

### Task: Clone Required Repositories
```bash
mkdir -p repos
cd repos

# Official Chatterbox
git clone https://github.com/resemble-ai/chatterbox.git

# Streaming support
git clone https://github.com/davidbrowne17/chatterbox-streaming.git

# vLLM port
git clone https://github.com/randombk/chatterbox-vllm.git

cd ..
```

### Task: Download Models
```bash
# Download PyTorch models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ResembleAI/chatterbox-turbo', local_dir='./models/chatterbox-turbo')
print('PyTorch models downloaded')
"

# Download ONNX models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ResembleAI/chatterbox-turbo-ONNX', local_dir='./models/chatterbox-turbo-onnx')
print('ONNX models downloaded')
"
```

### Task: Setup Docker with GPU
```bash
# Install NVIDIA Container Toolkit (if not already on AMI)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Task: Setup Redis
```bash
# Install Redis
sudo apt-get install -y redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify
redis-cli ping
```

## Environment Variables

```bash
# Add to ~/.bashrc or ~/.profile
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Chatterbox-specific
export CHATTERBOX_CFG_SCALE=0.5
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### CUDA not found
```bash
# Check CUDA installation
ls -la /usr/local/cuda
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Reinstall CUDA toolkit if needed
sudo apt-get install cuda-toolkit-12-4
```

### GPU Out of Memory (OOM)
```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in code or enable gradient checkpointing
```

### TensorRT Build Fails
```bash
# Check CUDA/TensorRT version compatibility
python -c "import tensorrt; print(tensorrt.__version__)"
nvcc --version

# Increase workspace size in trtexec
# --workspace=16384 (16GB)
```

### Docker GPU Access Issues
```bash
# Verify NVIDIA runtime
docker info | grep -i runtime

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

## Verification Checklist

After infrastructure setup, verify:

- [ ] `nvidia-smi` shows A10G with 24GB VRAM
- [ ] `nvcc --version` shows CUDA 12.x
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns True
- [ ] `python -c "import tensorrt"` works without errors
- [ ] `python -c "from chatterbox import ChatterboxTurboTTS"` imports successfully
- [ ] Conda environment 'chatterbox' is active
- [ ] All repositories cloned to `repos/`
- [ ] Models downloaded to `models/`
- [ ] Docker can access GPU
- [ ] Redis is running

## SSH Access

```bash
# From local machine
ssh -i ~/.ssh/krim-gpu.pem ubuntu@<GPU_SERVER_IP>

# Or using SSH config
# Add to ~/.ssh/config:
Host krim-gpu
    HostName <GPU_SERVER_IP>
    User ubuntu
    IdentityFile ~/.ssh/krim-gpu.pem
    StrictHostKeyChecking no

# Then connect with:
ssh krim-gpu
```

## Next Steps

After infrastructure is verified, proceed to:
1. Run baseline benchmarks (`agents/benchmarking/CLAUDE.md`)
2. Begin optimization (`agents/ml-optimization/CLAUDE.md`)
