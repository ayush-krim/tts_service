#!/bin/bash
# setup_environment.sh - Complete environment setup for Chatterbox TTS

set -e  # Exit on error

echo "=================================================="
echo "CHATTERBOX TTS ENVIRONMENT SETUP"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

echo ""
echo "Step 1: Creating conda environment..."
echo "--------------------------------------"
conda create -n chatterbox python=3.11 -y

echo ""
echo "Step 2: Activating environment..."
echo "----------------------------------"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate chatterbox

echo ""
echo "Step 3: Installing PyTorch with CUDA..."
echo "----------------------------------------"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 4: Verifying PyTorch CUDA..."
echo "----------------------------------"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    echo 'WARNING: CUDA not available!'
    exit 1
"

echo ""
echo "Step 5: Installing Chatterbox TTS..."
echo "-------------------------------------"
pip install git+https://github.com/resemble-ai/chatterbox.git

echo ""
echo "Step 6: Installing additional dependencies..."
echo "----------------------------------------------"
pip install \
    fastapi uvicorn redis aioredis \
    numpy scipy librosa soundfile \
    huggingface_hub transformers \
    pytest pytest-asyncio

echo ""
echo "Step 7: Installing TensorRT and ONNX Runtime..."
echo "-------------------------------------------------"
pip install tensorrt || echo "WARNING: TensorRT install failed (may need manual install)"
pip install onnxruntime-gpu

echo ""
echo "Step 8: Installing streaming support..."
echo "----------------------------------------"
pip install git+https://github.com/davidbrowne17/chatterbox-streaming.git || echo "WARNING: Streaming install failed"

echo ""
echo "Step 9: Verifying installations..."
echo "-----------------------------------"
python -c "
print('Checking imports...')

# PyTorch
import torch
print(f'  [OK] PyTorch {torch.__version__}')

# Chatterbox
try:
    from chatterbox.tts import ChatterboxTTS
    print('  [OK] Chatterbox TTS')
except ImportError:
    from chatterbox import ChatterboxTTS
    print('  [OK] Chatterbox TTS')

# TensorRT
try:
    import tensorrt
    print(f'  [OK] TensorRT {tensorrt.__version__}')
except ImportError:
    print('  [WARN] TensorRT not available')

# ONNX Runtime
import onnxruntime
print(f'  [OK] ONNX Runtime {onnxruntime.__version__}')
providers = onnxruntime.get_available_providers()
print(f'       Providers: {providers}')

# Streaming
try:
    from chatterbox_streaming import ChatterboxStreamingTTS
    print('  [OK] Chatterbox Streaming')
except ImportError:
    print('  [WARN] Chatterbox Streaming not available')

print('')
print('All critical dependencies installed!')
"

echo ""
echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "To activate the environment in new terminals, run:"
echo "  conda activate chatterbox"
echo ""
