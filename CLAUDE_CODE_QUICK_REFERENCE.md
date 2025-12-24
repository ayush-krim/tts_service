# Claude Code Quick Reference - Chatterbox TTS Optimization

## 🚀 5-Minute Setup

### Step 1: Install Claude Code
```bash
npm install -g @anthropic-ai/claude-code
claude auth login
```

### Step 2: Create Project
```bash
mkdir -p ~/krim-ai/chatterbox-optimization/{.claude,agents,scripts,models,engines}
cd ~/krim-ai/chatterbox-optimization
```

### Step 3: Create Root CLAUDE.md
This is the main agent definition that Claude Code reads automatically:

```bash
cat > CLAUDE.md << 'AGENT'
# Chatterbox TTS Optimization for Krim AI

## Context
Optimize Resemble AI's Chatterbox Turbo for sub-200ms latency on AWS G5 (A10G GPU).

## Architecture
- T3: GPT-2 backbone (350M) → Speech tokens (25Hz)
- S3Gen: Flow matching + Vocoder → Audio (24kHz)
- **Bottleneck**: S3Gen = 70% of inference time

## Targets
- First chunk latency: < 400ms
- RTF: < 0.5
- Daily volume: 30,000+ calls

## Phases
1. Week 1-2: Setup + Baseline benchmarks
2. Week 3-4: ONNX/TensorRT/vLLM optimization
3. Week 5-6: S3Gen TensorRT (critical!)
4. Week 7-8: FastAPI + Docker + Telephony

## Commands
- SSH: `ssh -i ~/.ssh/krim-gpu.pem ubuntu@GPU_IP`
- Conda: `conda activate chatterbox`
- Benchmark: `python scripts/benchmark/baseline.py`
- Server: `python server/main.py`
AGENT
```

### Step 4: Start Claude Code
```bash
cd ~/krim-ai/chatterbox-optimization
claude chat
```

---

## 📋 Agent Definitions Summary

| Agent | Location | Purpose |
|-------|----------|---------|
| **Root** | `/CLAUDE.md` | Main project context, orchestrates all work |
| **Infrastructure** | `/agents/infrastructure/CLAUDE.md` | AWS, CUDA, TensorRT setup |
| **ML Optimization** | `/agents/ml-optimization/CLAUDE.md` | ONNX, TensorRT, vLLM conversion |
| **Benchmarking** | `/agents/benchmarking/CLAUDE.md` | Performance measurement |
| **Deployment** | `/agents/deployment/CLAUDE.md` | Docker, FastAPI, telephony |

---

## 🔧 MCP Servers Required

Create `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "."]
    },
    "bash": {
      "command": "npx", 
      "args": ["-y", "@anthropic-ai/mcp-server-bash"]
    },
    "ssh": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-ssh"],
      "env": {
        "SSH_HOST": "your-gpu-server-ip",
        "SSH_USER": "ubuntu",
        "SSH_KEY_PATH": "~/.ssh/krim-gpu.pem"
      }
    }
  }
}
```

Install servers:
```bash
npm install -g @anthropic-ai/mcp-server-filesystem @anthropic-ai/mcp-server-bash @anthropic-ai/mcp-server-ssh
```

---

## 📅 Execution Prompts by Week

### Week 1: Infrastructure
```
> SSH into the GPU server and verify nvidia-smi shows A10G
> Create conda environment: conda create -n chatterbox python=3.11
> Install: pip install torch chatterbox-tts tensorrt onnxruntime-gpu
> Clone repos: chatterbox, chatterbox-streaming, chatterbox-vllm
> Download models from HuggingFace
```

### Week 2: Baseline
```
> Create scripts/benchmark/baseline.py with test texts
> Run baseline and save results to benchmarks/results/
> Create streaming benchmark script
> Document results in docs/BENCHMARKS.md
```

### Week 3-4: Optimization
```
> Download ONNX models from ResembleAI/chatterbox-turbo-ONNX
> Create ONNX inference wrapper with TensorRT EP
> Build TensorRT engines with trtexec
> Setup vLLM with version 0.9.2 exactly
> Benchmark all configurations
```

### Week 5-6: S3Gen (Critical)
```
> Profile S3Gen to confirm 70% bottleneck
> Export S3Gen to ONNX with dynamic shapes
> Build TensorRT engine for S3Gen
> Create hybrid pipeline: vLLM T3 + TensorRT S3Gen
> Verify 40%+ latency improvement
```

### Week 7-8: Production
```
> Create FastAPI server with streaming endpoints
> Add telephony: 24kHz→8kHz, G.711 μ-law, RTP
> Build Docker container with GPU support
> Run load tests for 200 concurrent requests
> Setup monitoring and alerts
```

---

## 💡 Key Commands

### In Claude Code Session:
```
# Read project context
> Read CLAUDE.md and understand the project

# Switch to specialized agent
> Read agents/ml-optimization/CLAUDE.md for optimization context

# Execute specific task
> SSH to GPU server, run nvidia-smi, create conda env

# Run benchmark
> Execute python scripts/benchmark/baseline.py and analyze results

# Build TensorRT engine
> Run trtexec to convert ONNX to TensorRT FP16
```

### Direct Terminal:
```bash
# Start Claude Code
claude chat

# With specific prompt
claude chat --prompt "Setup Week 1 infrastructure"

# Batch mode
claude run "Execute all Week 1 tasks and report results"
```

---

## ⚠️ Critical Notes

1. **vLLM version**: Must be exactly `0.9.2`
2. **CFG scale**: Set via `export CHATTERBOX_CFG_SCALE=0.5`
3. **S3Gen is 70% of latency** - Optimize this first!
4. **TensorRT shapes**: Always use min/opt/max profiles
5. **ONNX precision**: Use FP16 for best speed/quality tradeoff

---

## 📊 Target Metrics

| Metric | Baseline (A10G) | Target | Optimized |
|--------|-----------------|--------|-----------|
| First Chunk | 650-750ms | < 400ms | ~350ms |
| RTF | 0.70-0.80 | < 0.50 | ~0.45 |
| P95 Latency | 800ms | < 500ms | ~450ms |
| VRAM | 12GB | < 16GB | ~10GB |
