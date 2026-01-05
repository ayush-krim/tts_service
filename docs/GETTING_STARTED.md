# Getting Started Guide

This guide walks you through setting up and running the Chatterbox TTS optimization project.

## Prerequisites

### Hardware
- AWS G5.xlarge instance (or similar with NVIDIA A10G GPU)
- 24GB VRAM, 8 vCPUs, 16GB RAM minimum

### Software (on GPU server)
- Ubuntu 22.04 LTS
- CUDA 12.x (pre-installed on Deep Learning AMI)
- Conda or Miniconda

### Access
- SSH access to GPU server
- GitHub account with repo access

---

## Step 1: Clone the Repository

On your **GPU server**:

```bash
# SSH to your GPU server
ssh -i ~/.ssh/your-key.pem ubuntu@<GPU_SERVER_IP>

# Clone the repo
git clone https://github.com/ayush-krim/tts_service.git
cd tts_service
```

---

## Step 2: Run Phase 1 Setup

You have two options:

### Option A: Run Everything Automatically

```bash
bash scripts/setup/run_phase1.sh
```

This will:
1. Verify GPU setup
2. Create conda environment
3. Install all dependencies
4. Clone required repositories
5. Download models (~11GB)
6. Run baseline benchmarks
7. Run streaming benchmarks

**Estimated time: 30-60 minutes**

### Option B: Run Step by Step

```bash
# 1. Verify GPU
bash scripts/setup/verify_gpu.sh

# 2. Setup conda environment + dependencies
bash scripts/setup/setup_environment.sh

# 3. IMPORTANT: Activate the environment
conda activate chatterbox

# 4. Clone required repos (chatterbox, streaming, vllm)
bash scripts/setup/clone_repos.sh

# 5. Download models from HuggingFace (~11GB)
python scripts/setup/download_models.py

# 6. Run baseline benchmark (~15 min)
python scripts/benchmark/baseline.py

# 7. Run streaming benchmark (~15 min)
python scripts/benchmark/streaming.py
```

---

## Step 3: Review Results

After benchmarks complete, check results:

```bash
ls -la benchmarks/results/
cat benchmarks/results/baseline_*.json
cat benchmarks/results/streaming_*.json
```

### Expected Baseline Results (A10G)

| Metric | Expected | Target |
|--------|----------|--------|
| Mean Latency | 650-750ms | <400ms |
| P95 Latency | 800-900ms | <500ms |
| RTF | 0.65-0.80 | <0.50 |
| TTFC (streaming) | 500-600ms | <400ms |

---

## Step 4: Next Phases

### Phase 2: ONNX/TensorRT Optimization (Week 3-4)

```bash
# Read the ML optimization agent guide
cat agents/ml-optimization/CLAUDE.md

# Key tasks:
# - Use official ONNX models from HuggingFace
# - Enable TensorRT Execution Provider
# - Install vLLM 0.9.2 for T3 acceleration
```

### Phase 3: S3Gen TensorRT (Week 5-6) - CRITICAL

```bash
# S3Gen is 70% of latency - biggest optimization opportunity
# - Export S3Gen to ONNX
# - Build TensorRT engines
# - Expected: 40-50% latency reduction
```

### Phase 4: Production Deployment (Week 7-8)

```bash
# Read the deployment agent guide
cat agents/deployment/CLAUDE.md

# Key tasks:
# - FastAPI server with streaming
# - Telephony integration (8kHz G.711 μ-law)
# - Docker containerization
# - Load testing
```

---

## Project Structure

```
tts_service/
├── CLAUDE.md                      # Main project context
├── agents/                        # Specialized agent guides
│   ├── infrastructure/CLAUDE.md   # GPU/CUDA setup
│   ├── ml-optimization/CLAUDE.md  # ONNX/TensorRT/vLLM
│   ├── benchmarking/CLAUDE.md     # Performance testing
│   └── deployment/CLAUDE.md       # Docker/FastAPI
├── scripts/
│   ├── setup/                     # Setup scripts
│   ├── benchmark/                 # Benchmark scripts
│   └── optimization/              # (Phase 2-3)
├── models/                        # Downloaded models (gitignored)
├── engines/                       # TensorRT engines (gitignored)
├── benchmarks/results/            # Benchmark outputs
├── server/                        # Production server (Phase 4)
└── docker/                        # Docker configs (Phase 4)
```

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Project context and architecture overview |
| `CHATTERBOX_OPTIMIZATION_PLAN_VERIFIED.md` | Complete verified optimization plan |
| `agents/*/CLAUDE.md` | Specialized guides for each phase |
| `CLAUDE_CODE_QUICK_REFERENCE.md` | Quick reference for Claude Code |

---

## Troubleshooting

### CUDA not available
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Model download fails
```bash
# Check disk space
df -h

# Retry download
python scripts/setup/download_models.py
```

### Benchmark errors
```bash
# Ensure environment is activated
conda activate chatterbox

# Check GPU memory
nvidia-smi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Contact

For questions about this project, refer to:
- Optimization plan: `CHATTERBOX_OPTIMIZATION_PLAN_VERIFIED.md`
- Architecture details: `CLAUDE.md`
