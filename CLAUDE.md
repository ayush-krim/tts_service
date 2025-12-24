# Chatterbox TTS Optimization for Krim AI

## Project Overview

Optimize Resemble AI's **Chatterbox Turbo** TTS model for sub-200ms latency deployment on AWS G5 (A10G GPU) to power Krim AI's telephony infrastructure handling **30,000+ daily voice calls**.

## Architecture

```
CHATTERBOX TURBO PIPELINE
=========================

Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: CONDITIONING                                  │
│  VoiceEncoder (CAMPPlus) → 256-dim Speaker Embedding    │
│  Time: ~10ms                                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: T3 (Text-to-Token)                           │
│  GPT-2 Backbone: 350M params, 30 layers, 16 heads      │
│  Input: Text + Speaker Embedding                        │
│  Output: Speech Tokens @ 25Hz                           │
│  Time: ~30% of total inference                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: S3Gen (Token-to-Audio) ⚠️ BOTTLENECK         │
│  Flow Matching (1-step distilled) + HiFT-GAN Vocoder   │
│  Input: Speech Tokens                                   │
│  Output: Audio @ 24kHz                                  │
│  Time: ~70% of total inference ← PRIMARY TARGET        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: Perth Watermarker (Optional)                 │
│  Adds imperceptible audio watermark                     │
│  Can be disabled for telephony use                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output Audio (24kHz) → Resample to 8kHz for Telephony
```

## Model Files

| Component | File | Size | Purpose |
|-----------|------|------|---------|
| T3 Model | `t3_turbo_v1.safetensors` | 1.92 GB | GPT-2 text-to-token |
| S3Gen | `s3gen.safetensors` | 1.06 GB | Token-to-audio |
| VoiceEncoder | `ve.safetensors` | 5.7 MB | Speaker embedding |

### ONNX Models (HuggingFace)
- `language_model_fp16.onnx` - 635 MB
- `conditional_decoder_fp16.onnx` - 384 MB
- `speech_encoder_fp16.onnx` - 522 MB

## Target Metrics

| Metric | Baseline (A10G) | Target | Stretch Goal |
|--------|-----------------|--------|--------------|
| First Chunk Latency | 650-750ms | < 400ms | < 300ms |
| Real-Time Factor | 0.70-0.80 | < 0.50 | < 0.40 |
| P95 Latency | 800ms | < 500ms | < 400ms |
| VRAM Usage | 12GB | < 16GB | < 10GB |
| Throughput | 10 req/s | 20 req/s | 30 req/s |

## Infrastructure

- **GPU**: AWS G5 (NVIDIA A10G, 24GB VRAM)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) 20240816
- **vCPUs**: 8 (scalable to multi-GPU in production)
- **Target**: Containerized deployment for GPU cluster scaling

## Critical Requirements

1. **vLLM Version**: Must use exactly `0.9.2` for Chatterbox compatibility
2. **CFG Scale**: Set via `CHATTERBOX_CFG_SCALE=0.5` environment variable (not per-request)
3. **S3Gen Optimization**: Primary target - accounts for 70% of latency
4. **Telephony Output**: 8kHz G.711 μ-law encoding for RTP streaming
5. **Language**: English only (Turbo variant)

## Project Structure

```
chatterbox_tts_service/
├── CLAUDE.md                    # This file - project context
├── agents/                      # Specialized agent definitions
│   ├── infrastructure/          # AWS, CUDA, TensorRT setup
│   ├── ml-optimization/         # ONNX, TensorRT, vLLM conversion
│   ├── benchmarking/            # Performance measurement
│   └── deployment/              # Docker, FastAPI, telephony
├── scripts/
│   ├── setup/                   # Environment setup scripts
│   ├── benchmark/               # Benchmarking scripts
│   └── optimization/            # Model optimization scripts
├── configs/                     # Configuration files
├── benchmarks/results/          # Benchmark output data
├── models/                      # Downloaded model files
├── engines/                     # TensorRT engines
├── server/                      # FastAPI production server
├── docker/                      # Docker configurations
└── docs/                        # Documentation
```

## Optimization Phases

### Phase 1: Infrastructure (Week 1-2)
- Setup AWS G5 instance with CUDA 12.x
- Install TensorRT 10.x, PyTorch 2.x
- Clone Chatterbox repos and download models
- Establish baseline benchmarks

### Phase 2: T3 Optimization (Week 3-4)
- Convert T3 to ONNX with dynamic shapes
- Build TensorRT engine with FP16
- Integrate vLLM 0.9.2 for batched inference
- Benchmark: Target 30-40% T3 speedup

### Phase 3: S3Gen Optimization (Week 5-6) ⚠️ CRITICAL
- Profile S3Gen to identify sub-bottlenecks
- Export S3Gen components to ONNX
- Build TensorRT engines for flow matching + vocoder
- Target: 40-50% S3Gen speedup (biggest impact)

### Phase 4: Production (Week 7-8)
- FastAPI streaming server with chunked output
- Telephony integration (24kHz → 8kHz, G.711 μ-law, RTP)
- Docker containerization with GPU support
- Load testing: 200 concurrent requests
- Monitoring and alerting setup

## Key Commands

```bash
# SSH to GPU server
ssh -i ~/.ssh/krim-gpu.pem ubuntu@<GPU_IP>

# Activate environment
conda activate chatterbox

# Run baseline benchmark
python scripts/benchmark/baseline.py

# Start production server
python server/main.py

# Build TensorRT engine
trtexec --onnx=models/t3.onnx --saveEngine=engines/t3.trt --fp16
```

## References

- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Chatterbox vLLM](https://github.com/resemble-ai/chatterbox-vllm)
- [Chatterbox Streaming](https://github.com/resemble-ai/chatterbox-streaming)
- [ONNX Models](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)

## Agent Delegation

When working on specific tasks, read the appropriate agent CLAUDE.md:

| Task | Agent Location |
|------|----------------|
| AWS/CUDA setup | `agents/infrastructure/CLAUDE.md` |
| Model optimization | `agents/ml-optimization/CLAUDE.md` |
| Performance testing | `agents/benchmarking/CLAUDE.md` |
| Production deployment | `agents/deployment/CLAUDE.md` |
