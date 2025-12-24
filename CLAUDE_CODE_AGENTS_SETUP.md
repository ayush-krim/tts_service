# Claude Code Agents Setup for Chatterbox TTS Optimization

## Complete Step-by-Step Guide

This document provides everything needed to create and configure Claude Code agents for executing the Chatterbox Turbo optimization plan on Krim AI's infrastructure.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Claude Code Installation](#2-claude-code-installation)
3. [Project Structure](#3-project-structure)
4. [Agent Configuration Files](#4-agent-configuration-files)
5. [MCP Server Setup](#5-mcp-server-setup)
6. [Individual Agent Definitions](#6-individual-agent-definitions)
7. [Execution Workflows](#7-execution-workflows)
8. [Running the Plan](#8-running-the-plan)

---

## 1. Prerequisites

### 1.1 System Requirements

```bash
# Local Machine (where Claude Code runs)
- macOS, Linux, or Windows (WSL2)
- Node.js 18+ 
- Python 3.11+
- Git

# Remote Server (AWS G5.xlarge)
- Ubuntu 22.04 LTS
- NVIDIA A10G GPU (24GB VRAM)
- CUDA 12.4+
- Docker with NVIDIA Container Toolkit
```

### 1.2 Accounts & Access

```yaml
Required:
  - Anthropic API key (for Claude Code)
  - AWS account with G5 instance access
  - HuggingFace account (for model downloads)
  - GitHub account (for cloning repos)

Optional:
  - Redis Cloud account (for speaker caching)
  - Weights & Biases (for experiment tracking)
```

---

## 2. Claude Code Installation

### 2.1 Install Claude Code CLI

```bash
# Install via npm (recommended)
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version

# Authenticate with Anthropic
claude auth login
# This will open browser for authentication
```

### 2.2 Alternative: Install via Homebrew (macOS)

```bash
brew install claude-code
claude auth login
```

### 2.3 Verify Setup

```bash
# Test Claude Code is working
claude chat "Hello, are you ready to help with TTS optimization?"

# Check available commands
claude --help
```

---

## 3. Project Structure

### 3.1 Create Project Directory

```bash
# Create main project directory
mkdir -p ~/krim-ai/chatterbox-optimization
cd ~/krim-ai/chatterbox-optimization

# Create subdirectories
mkdir -p \
  .claude \
  agents \
  scripts \
  configs \
  benchmarks \
  models \
  engines \
  server \
  docker \
  docs
```

### 3.2 Complete Directory Structure

```
~/krim-ai/chatterbox-optimization/
├── .claude/                     # Claude Code configuration
│   ├── settings.json           # Global settings
│   ├── mcp.json                # MCP server configs
│   └── prompts/                # Saved prompts
│       ├── infrastructure.md
│       ├── optimization.md
│       ├── benchmarking.md
│       └── deployment.md
│
├── agents/                      # Agent definition files
│   ├── CLAUDE.md               # Main project agent (root)
│   ├── infrastructure/
│   │   └── CLAUDE.md
│   ├── ml-optimization/
│   │   └── CLAUDE.md
│   ├── benchmarking/
│   │   └── CLAUDE.md
│   └── deployment/
│       └── CLAUDE.md
│
├── scripts/                     # Execution scripts
│   ├── setup/
│   │   ├── install_cuda.sh
│   │   ├── install_tensorrt.sh
│   │   └── setup_conda.sh
│   ├── benchmark/
│   │   ├── baseline.py
│   │   ├── streaming.py
│   │   ├── onnx_benchmark.py
│   │   └── vllm_benchmark.py
│   ├── optimization/
│   │   ├── export_onnx.py
│   │   ├── convert_tensorrt.py
│   │   └── optimize_s3gen.py
│   └── server/
│       ├── main.py
│       └── telephony.py
│
├── configs/                     # Configuration files
│   ├── server_config.yaml
│   ├── model_config.yaml
│   └── benchmark_config.yaml
│
├── benchmarks/                  # Benchmark results
│   └── results/
│
├── models/                      # Downloaded models
│   ├── chatterbox-turbo/
│   └── chatterbox-turbo-onnx/
│
├── engines/                     # TensorRT engines
│   ├── t3_fp16.plan
│   ├── s3gen_fp16.plan
│   └── vocoder_fp16.plan
│
├── server/                      # Production server code
│   ├── app/
│   ├── tests/
│   └── requirements.txt
│
├── docker/                      # Docker configs
│   ├── Dockerfile
│   ├── Dockerfile.tensorrt
│   └── docker-compose.yml
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   └── DEPLOYMENT.md
│
├── CLAUDE.md                    # Root agent definition
├── requirements.txt
└── README.md
```

---

## 4. Agent Configuration Files

### 4.1 Root CLAUDE.md (Main Project Agent)

Create this file at the root of your project:

```bash
cat > ~/krim-ai/chatterbox-optimization/CLAUDE.md << 'EOF'
# Chatterbox TTS Optimization Project

## Project Overview

This project optimizes Resemble AI's Chatterbox Turbo TTS model for sub-200ms latency deployment on Krim AI's telephony infrastructure handling 30,000+ daily calls.

## Project Context

- **Target Hardware**: AWS G5.xlarge (NVIDIA A10G, 24GB VRAM)
- **Target Latency**: < 400ms first chunk, RTF < 0.5
- **Daily Volume**: 30,000+ calls
- **Languages**: English (Hindi via Multilingual variant if needed)

## Key Architecture

```
Chatterbox Turbo Pipeline:
1. Text → T3 (GPT-2 350M) → Speech Tokens (25Hz)
2. Speech Tokens → S3Gen (Flow Matching + Vocoder) → Audio (24kHz)

Bottleneck: S3Gen takes 70% of inference time
```

## Optimization Strategy

1. **Phase 1**: Baseline benchmarks on A10G
2. **Phase 2**: Streaming implementation (reduce perceived latency)
3. **Phase 3**: ONNX Runtime + TensorRT acceleration
4. **Phase 4**: vLLM integration for T3 backbone
5. **Phase 5**: TensorRT optimization for S3Gen (critical!)
6. **Phase 6**: Production FastAPI server
7. **Phase 7**: Docker containerization
8. **Phase 8**: Telephony integration (G.711, RTP)

## File Structure

- `/scripts/` - All Python/Bash scripts
- `/configs/` - YAML configurations
- `/benchmarks/` - Performance test results
- `/models/` - Downloaded model files
- `/engines/` - TensorRT engine files
- `/server/` - FastAPI production server
- `/docker/` - Container configurations

## Code Standards

- Python 3.11 with type hints
- Use async/await for I/O operations
- All benchmarks must be reproducible
- Log all GPU memory usage
- Save benchmark results as JSON

## Important Commands

```bash
# SSH to GPU server
ssh -i ~/.ssh/krim-gpu.pem ubuntu@<GPU_SERVER_IP>

# Activate conda environment
conda activate chatterbox

# Run benchmarks
python scripts/benchmark/baseline.py

# Start server
python server/main.py
```

## Current Status

Track progress in `/docs/PROGRESS.md`

## Agent Hierarchy

This root agent can delegate to specialized agents in `/agents/`:
- `infrastructure/` - AWS/GPU setup
- `ml-optimization/` - Model optimization
- `benchmarking/` - Performance testing
- `deployment/` - Docker/production

When working on specific domains, refer to the relevant agent's CLAUDE.md for specialized instructions.
EOF
```

### 4.2 Infrastructure Agent CLAUDE.md

```bash
mkdir -p ~/krim-ai/chatterbox-optimization/agents/infrastructure
cat > ~/krim-ai/chatterbox-optimization/agents/infrastructure/CLAUDE.md << 'EOF'
# Infrastructure Agent

## Role

Setup and maintain the GPU infrastructure for Chatterbox TTS optimization.

## Responsibilities

1. AWS G5 instance management
2. CUDA/cuDNN/TensorRT installation
3. Conda environment setup
4. Docker configuration
5. Network and security setup

## Server Details

```yaml
Instance: G5.xlarge
  GPU: NVIDIA A10G (24GB GDDR6)
  vCPUs: 8 (AMD EPYC)
  RAM: 16 GB
  Storage: 250 GB NVMe SSD
  AMI: Deep Learning AMI (Ubuntu 22.04)
```

## Required Software Stack

```bash
# Verify these are installed and configured:
- CUDA 12.4+
- cuDNN 8.9+
- TensorRT 10.x
- Python 3.11
- Conda/Mamba
- Docker with NVIDIA runtime
- Redis (for caching)
```

## Setup Scripts Location

All setup scripts are in `/scripts/setup/`:
- `install_cuda.sh` - CUDA installation
- `install_tensorrt.sh` - TensorRT setup
- `setup_conda.sh` - Conda environment

## Common Tasks

### Task: Verify GPU Setup
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

### Task: Install TensorRT
```bash
pip install tensorrt
python -c "import tensorrt; print(tensorrt.__version__)"
```

### Task: Setup Conda Environment
```bash
conda create -n chatterbox python=3.11 -y
conda activate chatterbox
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install chatterbox-tts
```

## Troubleshooting

- If CUDA not found: Check `LD_LIBRARY_PATH`
- If GPU OOM: Reduce batch size or enable gradient checkpointing
- If TensorRT build fails: Check CUDA/TensorRT version compatibility

## Outputs

After setup, verify:
- [ ] `nvidia-smi` shows A10G with 24GB
- [ ] `nvcc --version` shows CUDA 12.x
- [ ] `python -c "import tensorrt"` works
- [ ] Chatterbox model loads successfully
EOF
```

### 4.3 ML Optimization Agent CLAUDE.md

```bash
mkdir -p ~/krim-ai/chatterbox-optimization/agents/ml-optimization
cat > ~/krim-ai/chatterbox-optimization/agents/ml-optimization/CLAUDE.md << 'EOF'
# ML Optimization Agent

## Role

Optimize Chatterbox Turbo model for low-latency inference using ONNX, TensorRT, and vLLM.

## Critical Context

```
BOTTLENECK ANALYSIS:
- T3 (Token Generation): 15% of time
- S3Gen (Audio Synthesis): 70% of time  ◄── OPTIMIZE THIS FIRST
- Other (I/O, watermark): 15% of time
```

## Optimization Priorities

1. **S3Gen TensorRT** (HIGHEST PRIORITY)
   - Export to ONNX
   - Convert to TensorRT FP16
   - Expected: 40-50% latency reduction

2. **T3 vLLM Integration**
   - Use chatterbox-vllm port
   - Requires vLLM 0.9.2 exactly
   - Expected: 4-10x token generation speedup

3. **ONNX Runtime Optimization**
   - Use official ONNX models from HuggingFace
   - Enable TensorRT Execution Provider
   - Cache TensorRT engines

## Model Files

```
PyTorch (ResembleAI/chatterbox-turbo):
├── t3_turbo_v1.safetensors     (1.92 GB) - GPT-2 backbone
├── s3gen.safetensors           (1.06 GB) - Flow matching + vocoder
└── ve.safetensors              (5.7 MB)  - Voice encoder

ONNX (ResembleAI/chatterbox-turbo-ONNX):
├── language_model_fp16.onnx    (635 MB)  ◄── Use this
├── conditional_decoder_fp16.onnx (384 MB) ◄── Use this
└── speech_encoder_fp16.onnx    (522 MB)  ◄── Use this
```

## Scripts

Located in `/scripts/optimization/`:
- `export_onnx.py` - Export PyTorch to ONNX
- `convert_tensorrt.py` - ONNX to TensorRT
- `optimize_s3gen.py` - S3Gen specific optimization

## Key Commands

### Export to ONNX
```python
import torch
from chatterbox import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
# Export T3, S3Gen, Vocoder separately
```

### Convert to TensorRT
```bash
trtexec \
  --onnx=./models/s3gen.onnx \
  --saveEngine=./engines/s3gen_fp16.plan \
  --fp16 \
  --workspace=8192
```

### vLLM Setup
```bash
pip install vllm==0.9.2
export CHATTERBOX_CFG_SCALE=0.5
```

## Validation Checklist

After optimization, verify:
- [ ] ONNX model loads without errors
- [ ] TensorRT engine builds successfully
- [ ] Output audio quality is acceptable (no artifacts)
- [ ] Latency improved by at least 30%

## Known Issues

1. **vLLM CFG limitation**: Cannot tune per-request (environment variable only)
2. **vLLM version lock**: Must use exactly 0.9.2
3. **S3Gen export**: May need to trace submodules separately
4. **TensorRT dynamic shapes**: Use min/opt/max shape profiles
EOF
```

### 4.4 Benchmarking Agent CLAUDE.md

```bash
mkdir -p ~/krim-ai/chatterbox-optimization/agents/benchmarking
cat > ~/krim-ai/chatterbox-optimization/agents/benchmarking/CLAUDE.md << 'EOF'
# Benchmarking Agent

## Role

Measure, track, and report performance metrics for Chatterbox TTS optimization.

## Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| TTFS (Time to First Sound) | Latency until first audio chunk | < 400ms |
| RTF (Real-Time Factor) | Generation time / Audio duration | < 0.5 |
| P95 Latency | 95th percentile latency | < 500ms |
| P99 Latency | 99th percentile latency | < 750ms |
| Throughput | Concurrent streams per GPU | > 3 |
| VRAM Usage | GPU memory consumption | < 20GB |

## Benchmark Scripts

Located in `/scripts/benchmark/`:

### baseline.py
Measures vanilla PyTorch inference without optimization.

### streaming.py
Measures time-to-first-chunk with streaming output.

### onnx_benchmark.py
Compares ONNX Runtime vs PyTorch.

### vllm_benchmark.py
Benchmarks vLLM-accelerated inference.

## Standard Test Texts

Always use these for consistent comparison:

```python
TEST_TEXTS = [
    # Ultra short (< 10 words)
    "Hello there.",
    
    # Short (10-20 words)
    "Hi, this is Sarah from customer service. How can I help you today?",
    
    # Medium (20-40 words)
    "Thank you for calling. I can see your account here. Let me check the details of your recent transaction and explain what happened with your payment.",
    
    # Long (40+ words)
    "I understand you're concerned about the charges on your account. Let me walk you through each item on your bill. First, there's your monthly subscription fee, then we have the usage charges from last month, and finally a small processing fee that applies to all accounts.",
]
```

## Benchmark Protocol

1. **Warmup**: 5 runs (discard results)
2. **Measurement**: 20 runs minimum
3. **Report**: Mean, P50, P95, P99
4. **GPU Sync**: Always call `torch.cuda.synchronize()`

## Output Format

Save all benchmarks as JSON:

```json
{
  "timestamp": "2024-12-23T10:00:00Z",
  "configuration": {
    "gpu": "NVIDIA A10G",
    "cuda_version": "12.4",
    "model": "chatterbox-turbo",
    "optimization": "baseline"
  },
  "results": {
    "short_text": {
      "mean_ms": 450,
      "p50_ms": 440,
      "p95_ms": 520,
      "p99_ms": 580,
      "rtf": 0.65
    }
  }
}
```

## Comparison Template

After each optimization phase, create comparison:

```
┌─────────────────────────────────────────────────────────────────┐
│ BENCHMARK COMPARISON: [Phase Name]                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Metric          Before      After       Improvement            │
│  ─────────────────────────────────────────────────────────────  │
│  Mean Latency    650ms       420ms       -35.4%                 │
│  P95 Latency     780ms       510ms       -34.6%                 │
│  RTF             0.75        0.48        -36.0%                 │
│  VRAM Usage      12GB        10GB        -16.7%                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Results Storage

All results go to `/benchmarks/results/` with naming:
- `baseline_YYYYMMDD_HHMMSS.json`
- `streaming_YYYYMMDD_HHMMSS.json`
- `onnx_YYYYMMDD_HHMMSS.json`
- `vllm_YYYYMMDD_HHMMSS.json`
- `final_YYYYMMDD_HHMMSS.json`
EOF
```

### 4.5 Deployment Agent CLAUDE.md

```bash
mkdir -p ~/krim-ai/chatterbox-optimization/agents/deployment
cat > ~/krim-ai/chatterbox-optimization/agents/deployment/CLAUDE.md << 'EOF'
# Deployment Agent

## Role

Package, containerize, and deploy the optimized Chatterbox TTS service.

## Deployment Stack

```yaml
Application:
  - FastAPI (async REST API)
  - Uvicorn (ASGI server)
  - Redis (speaker embedding cache)

Container:
  - Docker with NVIDIA runtime
  - Base: nvcr.io/nvidia/tensorrt:24.06-py3

Infrastructure:
  - AWS G5.xlarge (or multi-GPU cluster)
  - Application Load Balancer
  - CloudWatch monitoring
```

## API Endpoints

```
POST /v1/synthesize
  - Input: text, speaker_id, streaming (bool)
  - Output: audio/wav or chunked stream

POST /v1/synthesize/stream
  - Input: text, speaker_id
  - Output: Server-Sent Events with audio chunks

GET /v1/speakers
  - List available speaker profiles

POST /v1/speakers
  - Upload new speaker reference audio

GET /health
  - Health check

GET /metrics
  - Prometheus metrics
```

## Docker Configuration

### Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorrt:24.06-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
WORKDIR /app
COPY . .

# Pre-download models
RUN python -c "from chatterbox import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')"

ENV CUDA_VISIBLE_DEVICES=0
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Telephony Integration

For Krim AI's telephony system:

### Audio Format Conversion
```python
# Convert 24kHz -> 8kHz (G.711 μ-law)
import g711

def convert_for_telephony(audio_24k: np.ndarray) -> bytes:
    # Resample to 8kHz
    audio_8k = librosa.resample(audio_24k, orig_sr=24000, target_sr=8000)
    
    # Convert to 16-bit PCM
    pcm16 = (audio_8k * 32767).astype(np.int16)
    
    # Encode to μ-law
    return g711.encode_ulaw(pcm16.tobytes())
```

### RTP Streaming
```python
# Stream 20ms RTP packets
PACKET_SIZE = 160  # 20ms @ 8kHz
PAYLOAD_TYPE = 0   # PCMU

async def stream_rtp(audio_bytes: bytes, dest: tuple):
    seq = 0
    timestamp = 0
    
    for i in range(0, len(audio_bytes), PACKET_SIZE):
        chunk = audio_bytes[i:i+PACKET_SIZE]
        rtp_packet = build_rtp_packet(seq, timestamp, PAYLOAD_TYPE, chunk)
        await send_udp(rtp_packet, dest)
        seq += 1
        timestamp += PACKET_SIZE
        await asyncio.sleep(0.020)  # 20ms
```

## Scaling Configuration

### For 30,000 daily calls:
```yaml
# Assuming 3-min avg call, distributed over 8 hours
# Peak: ~200 concurrent calls

gpu_cluster:
  t3_workers: 2      # Token generation
  s3gen_workers: 4   # Audio synthesis (bottleneck)
  
redis:
  speaker_cache_mb: 512
  ttl_seconds: 3600
  
load_balancer:
  algorithm: least_connections
  health_check: /health
  timeout_ms: 30000
```

## Deployment Checklist

- [ ] Docker image builds successfully
- [ ] Container starts and serves /health
- [ ] GPU is accessible inside container
- [ ] TensorRT engines load correctly
- [ ] Latency meets targets under load
- [ ] Telephony codec conversion works
- [ ] Redis caching reduces repeat speaker latency
- [ ] Monitoring and alerts configured
EOF
```

---

## 5. MCP Server Setup

### 5.1 Create MCP Configuration

```bash
mkdir -p ~/krim-ai/chatterbox-optimization/.claude
cat > ~/krim-ai/chatterbox-optimization/.claude/mcp.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@anthropic-ai/mcp-server-filesystem",
        "/home/ubuntu/krim-ai/chatterbox-optimization"
      ],
      "description": "Access project files and directories"
    },
    "bash": {
      "command": "npx",
      "args": [
        "-y",
        "@anthropic-ai/mcp-server-bash"
      ],
      "description": "Execute shell commands for GPU operations, benchmarks, and deployment"
    },
    "ssh": {
      "command": "npx",
      "args": [
        "-y",
        "@anthropic-ai/mcp-server-ssh"
      ],
      "env": {
        "SSH_HOST": "your-gpu-server-ip",
        "SSH_USER": "ubuntu",
        "SSH_KEY_PATH": "~/.ssh/krim-gpu.pem"
      },
      "description": "SSH into GPU server for remote operations"
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@anthropic-ai/mcp-server-github"
      ],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      },
      "description": "Clone repos and manage code"
    }
  }
}
EOF
```

### 5.2 Install MCP Servers

```bash
# Install required MCP servers globally
npm install -g @anthropic-ai/mcp-server-filesystem
npm install -g @anthropic-ai/mcp-server-bash
npm install -g @anthropic-ai/mcp-server-ssh
npm install -g @anthropic-ai/mcp-server-github

# Verify installations
npx @anthropic-ai/mcp-server-filesystem --version
npx @anthropic-ai/mcp-server-bash --version
```

### 5.3 Configure SSH Access

```bash
# Create SSH config for GPU server
cat >> ~/.ssh/config << 'EOF'

Host krim-gpu
    HostName your-gpu-server-ip
    User ubuntu
    IdentityFile ~/.ssh/krim-gpu.pem
    StrictHostKeyChecking no
EOF

# Test connection
ssh krim-gpu "nvidia-smi"
```

---

## 6. Individual Agent Definitions

### 6.1 Claude Code Settings

```bash
cat > ~/krim-ai/chatterbox-optimization/.claude/settings.json << 'EOF'
{
  "model": "claude-sonnet-4-20250514",
  "maxTokens": 16000,
  "temperature": 0,
  "autoApprove": [
    "read",
    "list",
    "search"
  ],
  "permissions": {
    "allowedDirectories": [
      "/home/ubuntu/krim-ai/chatterbox-optimization",
      "/tmp"
    ],
    "allowedCommands": [
      "python",
      "pip",
      "conda",
      "nvidia-smi",
      "nvcc",
      "trtexec",
      "docker",
      "git",
      "curl",
      "wget"
    ],
    "blockedCommands": [
      "rm -rf /",
      "sudo rm",
      "mkfs",
      "dd"
    ]
  },
  "context": {
    "includeGitHistory": true,
    "maxFileSize": "10MB"
  }
}
EOF
```

---

## 7. Execution Workflows

### 7.1 Week 1: Infrastructure Setup

Create a prompt file for Claude Code:

```bash
cat > ~/krim-ai/chatterbox-optimization/.claude/prompts/week1_infrastructure.md << 'EOF'
# Week 1: Infrastructure Setup

## Objective
Set up the GPU server environment for Chatterbox TTS optimization.

## Tasks

### Task 1.1: Verify GPU Server
SSH into the GPU server and verify:
1. GPU is NVIDIA A10G with 24GB VRAM
2. CUDA is installed (12.x)
3. Driver version is compatible

Commands to run:
```bash
nvidia-smi
nvcc --version
```

### Task 1.2: Create Conda Environment
```bash
conda create -n chatterbox python=3.11 -y
conda activate chatterbox
```

### Task 1.3: Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Task 1.4: Install Chatterbox
```bash
pip install chatterbox-tts
python -c "from chatterbox import ChatterboxTurboTTS; print('OK')"
```

### Task 1.5: Install TensorRT
```bash
pip install tensorrt onnxruntime-gpu
python -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')"
```

### Task 1.6: Install Additional Dependencies
```bash
pip install \
    fastapi uvicorn redis aioredis \
    numpy scipy librosa soundfile \
    huggingface_hub transformers
```

### Task 1.7: Clone Required Repositories
```bash
cd ~/krim-ai/chatterbox-optimization

# Official Chatterbox
git clone https://github.com/resemble-ai/chatterbox.git repos/chatterbox

# Streaming support
git clone https://github.com/davidbrowne17/chatterbox-streaming.git repos/chatterbox-streaming

# vLLM port
git clone https://github.com/randombk/chatterbox-vllm.git repos/chatterbox-vllm
```

### Task 1.8: Download Models
```bash
# Download PyTorch models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ResembleAI/chatterbox-turbo', local_dir='./models/chatterbox-turbo')
"

# Download ONNX models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ResembleAI/chatterbox-turbo-ONNX', local_dir='./models/chatterbox-turbo-onnx')
"
```

## Expected Outputs
- [ ] nvidia-smi shows A10G with 24GB
- [ ] Conda environment 'chatterbox' created
- [ ] PyTorch with CUDA working
- [ ] Chatterbox-tts installed
- [ ] TensorRT installed
- [ ] All repos cloned
- [ ] Models downloaded

## Next Steps
Proceed to Week 1 benchmarking after setup is verified.
EOF
```

### 7.2 Week 2: Baseline Benchmarking

```bash
cat > ~/krim-ai/chatterbox-optimization/.claude/prompts/week2_benchmarking.md << 'EOF'
# Week 2: Baseline Benchmarking

## Objective
Establish baseline performance metrics on A10G GPU.

## Tasks

### Task 2.1: Create Baseline Benchmark Script
Create `/scripts/benchmark/baseline.py` with:
- Multiple test text lengths
- 20 measurement runs
- Warmup phase
- GPU synchronization
- JSON output

### Task 2.2: Run Baseline Benchmark
```bash
cd ~/krim-ai/chatterbox-optimization
python scripts/benchmark/baseline.py
```

Expected output:
```
GPU: NVIDIA A10G
CUDA: 12.4

Text Length  | Mean (ms) | P95 (ms) | RTF
-------------|-----------|----------|------
Short (20w)  | 550       | 620      | 0.65
Medium (40w) | 720       | 810      | 0.68
Long (60w)   | 950       | 1100     | 0.70
```

### Task 2.3: Create Streaming Benchmark
Create `/scripts/benchmark/streaming.py` to measure:
- Time to first chunk (TTFC)
- Chunk latency distribution
- End-to-end latency

### Task 2.4: Run Streaming Benchmark
```bash
pip install git+https://github.com/davidbrowne17/chatterbox-streaming.git
python scripts/benchmark/streaming.py
```

### Task 2.5: Document Results
Save results to `/benchmarks/results/baseline_YYYYMMDD.json`

Create comparison table in `/docs/BENCHMARKS.md`

## Deliverables
- [ ] baseline.py script
- [ ] streaming.py script  
- [ ] Baseline results JSON
- [ ] Streaming results JSON
- [ ] BENCHMARKS.md documentation
EOF
```

### 7.3 Weeks 3-4: ONNX/TensorRT Optimization

```bash
cat > ~/krim-ai/chatterbox-optimization/.claude/prompts/week3_4_optimization.md << 'EOF'
# Weeks 3-4: ONNX/TensorRT Optimization

## Objective
Accelerate inference using ONNX Runtime and TensorRT.

## Tasks

### Task 3.1: Use Official ONNX Models
The official ONNX models are already available:
```python
from huggingface_hub import snapshot_download
model_path = snapshot_download("ResembleAI/chatterbox-turbo-ONNX")
```

### Task 3.2: Create ONNX Inference Wrapper
Create `/scripts/optimization/onnx_inference.py`:
- Load FP16 ONNX models
- Configure TensorRT Execution Provider
- Enable engine caching
- Implement warmup

### Task 3.3: Benchmark ONNX vs PyTorch
Run comparison:
```bash
python scripts/benchmark/onnx_benchmark.py
```

Expected improvement: 20-30% latency reduction

### Task 3.4: TensorRT Engine Building
```bash
# Build TensorRT engines for each model
trtexec \
    --onnx=./models/chatterbox-turbo-onnx/onnx/conditional_decoder_fp16.onnx \
    --saveEngine=./engines/s3gen_fp16.plan \
    --fp16 \
    --workspace=8192
```

### Task 3.5: vLLM Integration
```bash
cd repos/chatterbox-vllm
pip install vllm==0.9.2
pip install -e .

export CHATTERBOX_CFG_SCALE=0.5
python benchmark_vllm.py
```

### Task 3.6: Benchmark All Configurations
Compare:
1. Baseline PyTorch
2. ONNX Runtime (CPU EP)
3. ONNX Runtime (TensorRT EP)
4. vLLM

## Deliverables
- [ ] onnx_inference.py wrapper
- [ ] TensorRT engines built
- [ ] vLLM working
- [ ] Comparison benchmark results
- [ ] Updated BENCHMARKS.md
EOF
```

### 7.4 Weeks 5-6: S3Gen Optimization (Critical)

```bash
cat > ~/krim-ai/chatterbox-optimization/.claude/prompts/week5_6_s3gen.md << 'EOF'
# Weeks 5-6: S3Gen Optimization (CRITICAL)

## Objective
Optimize S3Gen which takes 70% of inference time.

## Context
```
S3Gen Components:
├── S3Token2Mel (Flow Matching)
│   ├── UpsampleConformerEncoder (512 dim, 6 blocks)
│   └── CausalConditionalCFM (256 channels, 4 blocks)
└── HiFTGenerator (Vocoder)
    └── Upsampling [8, 5, 3]
```

## Tasks

### Task 5.1: Profile S3Gen
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run S3Gen inference
    pass

print(prof.key_averages().table())
```

### Task 5.2: Export S3Gen to ONNX
```python
# See /scripts/optimization/export_s3gen.py
# Must handle:
# - Dynamic sequence lengths
# - Speaker embedding conditioning
# - Reference mel input
```

### Task 5.3: Build Optimized TensorRT Engine
```bash
trtexec \
    --onnx=./s3gen_onnx/s3gen.onnx \
    --saveEngine=./engines/s3gen_fp16.plan \
    --fp16 \
    --minShapes=speech_tokens:1x10,speaker_embedding:1x256 \
    --optShapes=speech_tokens:1x200,speaker_embedding:1x256 \
    --maxShapes=speech_tokens:4x500,speaker_embedding:4x256 \
    --workspace=8192 \
    --verbose
```

### Task 5.4: Integrate TensorRT S3Gen
Create hybrid pipeline:
- T3: vLLM (or ONNX)
- S3Gen: TensorRT engine

### Task 5.5: Benchmark Optimized S3Gen
Compare:
- PyTorch S3Gen
- ONNX S3Gen
- TensorRT S3Gen

Expected improvement: 40-50% latency reduction

## Troubleshooting

### Issue: ONNX export fails
Try exporting submodules separately:
- S3Token2Mel.encoder
- S3Token2Mel.cfm
- HiFTGenerator

### Issue: TensorRT shape errors
Use explicit shape profiles with min/opt/max

### Issue: Audio quality degradation
Check for:
- Precision issues (try FP32)
- Dynamic shape handling
- Correct input normalization

## Deliverables
- [ ] S3Gen profiling results
- [ ] S3Gen ONNX export
- [ ] S3Gen TensorRT engine
- [ ] Hybrid pipeline implementation
- [ ] Benchmark showing 40%+ improvement
EOF
```

### 7.5 Weeks 7-8: Production Deployment

```bash
cat > ~/krim-ai/chatterbox-optimization/.claude/prompts/week7_8_deployment.md << 'EOF'
# Weeks 7-8: Production Deployment

## Objective
Deploy optimized Chatterbox TTS for production telephony.

## Tasks

### Task 7.1: FastAPI Server
Create `/server/main.py` with:
- `/v1/synthesize` - Standard TTS
- `/v1/synthesize/stream` - Streaming TTS
- `/health` - Health check
- `/metrics` - Prometheus metrics

### Task 7.2: Telephony Integration
Create `/server/telephony.py` with:
- 24kHz → 8kHz resampling
- G.711 μ-law encoding
- RTP packet streaming

### Task 7.3: Docker Container
Create `/docker/Dockerfile`:
- Base: nvcr.io/nvidia/tensorrt:24.06-py3
- Pre-downloaded models
- TensorRT engines
- Health check

### Task 7.4: Docker Compose
Create `/docker/docker-compose.yml`:
- TTS service with GPU
- Redis for caching
- Proper networking

### Task 7.5: Load Testing
```bash
# Install load testing tool
pip install locust

# Run load test
locust -f tests/load_test.py --host http://localhost:8080
```

Target: 200 concurrent requests

### Task 7.6: Monitoring Setup
- Prometheus metrics
- Grafana dashboard
- CloudWatch alarms

### Task 7.7: Final Benchmarks
Run comprehensive benchmarks:
- Latency under load
- Throughput limits
- Memory usage over time
- Error rates

### Task 7.8: Documentation
Update all documentation:
- ARCHITECTURE.md
- DEPLOYMENT.md
- API.md
- TROUBLESHOOTING.md

## Deliverables
- [ ] FastAPI server
- [ ] Telephony integration
- [ ] Docker configuration
- [ ] Load test results
- [ ] Monitoring setup
- [ ] Final documentation
EOF
```

---

## 8. Running the Plan

### 8.1 Start Claude Code in Project

```bash
cd ~/krim-ai/chatterbox-optimization

# Start Claude Code with project context
claude chat

# Or start with specific prompt
claude chat --prompt .claude/prompts/week1_infrastructure.md
```

### 8.2 Example Session: Week 1

```bash
# Terminal 1: Start Claude Code
cd ~/krim-ai/chatterbox-optimization
claude chat

# In Claude Code session:
> Read the project CLAUDE.md and understand the project structure

> Now read .claude/prompts/week1_infrastructure.md and execute Task 1.1

> SSH into the GPU server and run nvidia-smi to verify the GPU

> Create the conda environment as specified in Task 1.2

> Continue with the remaining tasks...
```

### 8.3 Batch Execution Mode

For automated execution:

```bash
# Create execution script
cat > run_week1.sh << 'EOF'
#!/bin/bash
cd ~/krim-ai/chatterbox-optimization

# Run Week 1 tasks via Claude Code
claude run "
1. Read the project context from CLAUDE.md
2. Execute all tasks in .claude/prompts/week1_infrastructure.md
3. Save results to /docs/week1_results.md
4. Report any errors encountered
"
EOF

chmod +x run_week1.sh
./run_week1.sh
```

### 8.4 Interactive Workflow

For complex tasks requiring judgment:

```bash
cd ~/krim-ai/chatterbox-optimization

# Start interactive session
claude chat

# Example interaction:
You: "I want to run the S3Gen optimization from Week 5. 
     First, check if the baseline benchmarks exist.
     If not, run baseline first.
     Then proceed with S3Gen export to ONNX."

Claude: [Reads CLAUDE.md for context]
        [Checks /benchmarks/results/ for existing files]
        [Either runs baseline or proceeds with S3Gen]
        [Reports progress and any issues]
```

### 8.5 Agent Switching

When working on specific domains:

```bash
# In Claude Code session:

> Switch context to the ML Optimization agent by reading agents/ml-optimization/CLAUDE.md

> Now I need to export S3Gen to ONNX. What are the steps?

# Claude will use the specialized agent's instructions
```

### 8.6 Progress Tracking

Create a progress file:

```bash
cat > ~/krim-ai/chatterbox-optimization/docs/PROGRESS.md << 'EOF'
# Chatterbox Optimization Progress

## Week 1: Infrastructure ⏳
- [ ] GPU server verified
- [ ] Conda environment created
- [ ] PyTorch + CUDA installed
- [ ] Chatterbox installed
- [ ] TensorRT installed
- [ ] Repos cloned
- [ ] Models downloaded

## Week 2: Baseline Benchmarks ⏳
- [ ] baseline.py created
- [ ] streaming.py created
- [ ] Baseline results collected
- [ ] Streaming results collected
- [ ] BENCHMARKS.md updated

## Weeks 3-4: ONNX/TensorRT ⏳
- [ ] ONNX inference wrapper
- [ ] TensorRT engines built
- [ ] vLLM integration
- [ ] Comparison benchmarks

## Weeks 5-6: S3Gen Optimization ⏳
- [ ] S3Gen profiled
- [ ] S3Gen ONNX export
- [ ] S3Gen TensorRT engine
- [ ] 40%+ improvement verified

## Weeks 7-8: Production ⏳
- [ ] FastAPI server
- [ ] Telephony integration
- [ ] Docker deployment
- [ ] Load testing passed
- [ ] Monitoring setup
- [ ] Documentation complete

## Current Status
Last updated: YYYY-MM-DD
Current phase: Week 1
Blockers: None
EOF
```

---

## Quick Start Checklist

```bash
# 1. Install Claude Code
npm install -g @anthropic-ai/claude-code
claude auth login

# 2. Create project structure
mkdir -p ~/krim-ai/chatterbox-optimization
cd ~/krim-ai/chatterbox-optimization

# 3. Copy all CLAUDE.md files from this guide

# 4. Setup MCP servers
# Copy .claude/mcp.json from this guide

# 5. Start Claude Code
claude chat

# 6. Begin Week 1
> Read CLAUDE.md and .claude/prompts/week1_infrastructure.md
> Execute Task 1.1
```

---

## Troubleshooting

### Claude Code can't access GPU
- Ensure SSH config is correct
- Check MCP SSH server is running
- Verify GPU server is accessible

### ONNX export fails
- Check PyTorch version compatibility
- Try exporting smaller submodules
- Use opset_version=17

### TensorRT build fails
- Verify CUDA/TensorRT version match
- Increase workspace size
- Check for unsupported operations

### vLLM not working
- Must use exactly version 0.9.2
- Set CHATTERBOX_CFG_SCALE env var
- Check GPU memory availability

---

*Document created for Krim AI - December 2024*
