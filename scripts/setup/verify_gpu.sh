#!/bin/bash
# verify_gpu.sh - Verify GPU and CUDA installation on AWS G5

echo "=================================================="
echo "GPU & CUDA VERIFICATION"
echo "=================================================="

echo ""
echo "1. NVIDIA Driver & GPU Info:"
echo "----------------------------"
nvidia-smi

echo ""
echo "2. CUDA Version:"
echo "----------------"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH (this may be OK on Deep Learning AMI)"

echo ""
echo "3. CUDA Libraries:"
echo "------------------"
ls -la /usr/local/cuda/lib64/*.so* 2>/dev/null | head -5 || echo "CUDA libs not in standard location"

echo ""
echo "=================================================="
echo "VERIFICATION COMPLETE"
echo "=================================================="
