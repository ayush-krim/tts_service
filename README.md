# Chatterbox TTS Optimization Service

Optimized deployment of Resemble AI's **Chatterbox Turbo** TTS model for low-latency telephony applications.

## Target Metrics

| Metric | Baseline (A10G) | Target |
|--------|-----------------|--------|
| First Chunk Latency | 650-750ms | <400ms |
| Real-Time Factor | 0.70-0.80 | <0.50 |
| P95 Latency | 800ms | <500ms |
| VRAM Usage | 12GB | <16GB |

## Architecture

```
Text → T3 (GPT-2 350M) → Speech Tokens → S3Gen → Audio (24kHz)
                                           ↑
                                    70% of latency
```

## Project Structure

```
chatterbox_tts_service/
├── CLAUDE.md                 # Project context for Claude Code
├── agents/                   # Specialized agent definitions
│   ├── infrastructure/       # AWS/GPU setup
│   ├── ml-optimization/      # ONNX/TensorRT/vLLM
│   ├── benchmarking/         # Performance testing
│   └── deployment/           # Docker/FastAPI/telephony
├── scripts/
│   ├── setup/                # Environment setup scripts
│   ├── benchmark/            # Benchmarking scripts
│   └── optimization/         # Model optimization scripts
├── configs/                  # Configuration files
├── benchmarks/results/       # Benchmark outputs
├── models/                   # Downloaded models (gitignored)
├── engines/                  # TensorRT engines (gitignored)
├── server/                   # FastAPI production server
├── docker/                   # Docker configurations
└── docs/                     # Documentation
```

## Quick Start

### Prerequisites

- AWS G5.xlarge instance (NVIDIA A10G, 24GB VRAM)
- Ubuntu 22.04 with CUDA 12.x
- Conda/Miniconda

### Phase 1: Setup & Baseline

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/chatterbox_tts_service.git
cd chatterbox_tts_service

# Run Phase 1 setup (on GPU server)
bash scripts/setup/run_phase1.sh
```

Or run step-by-step:

```bash
# 1. Verify GPU
bash scripts/setup/verify_gpu.sh

# 2. Setup environment
bash scripts/setup/setup_environment.sh
conda activate chatterbox

# 3. Clone required repos
bash scripts/setup/clone_repos.sh

# 4. Download models (~11GB)
python scripts/setup/download_models.py

# 5. Run baseline benchmark
python scripts/benchmark/baseline.py

# 6. Run streaming benchmark
python scripts/benchmark/streaming.py
```

## Optimization Phases

| Phase | Week | Focus | Expected Improvement |
|-------|------|-------|---------------------|
| 1 | 1-2 | Baseline & Streaming | Establish metrics |
| 2 | 3-4 | ONNX/TensorRT/vLLM | 20-30% latency reduction |
| 3 | 5-6 | S3Gen TensorRT | 40-50% latency reduction |
| 4 | 7-8 | Production Server | Docker + telephony |

## Key Technologies

- **Chatterbox Turbo**: GPT-2 backbone (350M), 1-step S3Gen
- **vLLM 0.9.2**: Accelerated token generation
- **TensorRT**: S3Gen optimization (critical bottleneck)
- **ONNX Runtime**: Cross-platform inference
- **FastAPI**: Production-ready async server

## References

- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Chatterbox vLLM](https://github.com/randombk/chatterbox-vllm)
- [Chatterbox Streaming](https://github.com/davidbrowne17/chatterbox-streaming)
- [ONNX Models](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)

## License

This project is for internal use at Krim AI.
