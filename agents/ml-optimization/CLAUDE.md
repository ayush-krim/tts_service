# ML Optimization Agent

## Role

Optimize Chatterbox Turbo model for low-latency inference using ONNX, TensorRT, and vLLM.

## Critical Context

```
CHATTERBOX TURBO BOTTLENECK ANALYSIS
====================================

Component        | Time %  | Priority | Optimization
-----------------|---------|----------|------------------
VoiceEncoder     | ~5%     | LOW      | Already fast
T3 (GPT-2)       | ~25%    | MEDIUM   | vLLM / TensorRT-LLM
S3Gen            | ~70%    | HIGHEST  | TensorRT FP16 ◄──
Perth Watermark  | ~0%     | SKIP     | Can disable

FOCUS: S3Gen optimization yields the biggest latency reduction!
```

## Model Architecture Details

### T3 Model (Text-to-Token)
```yaml
Architecture: GPT-2
Parameters: 350M
Layers: 30
Attention Heads: 16
Hidden Dimension: 1024
Output: Speech tokens @ 25Hz
File: t3_turbo_v1.safetensors (1.92 GB)
```

### S3Gen (Token-to-Audio)
```yaml
Components:
  S3Token2Mel:
    - UpsampleConformerEncoder (512 dim, 6 blocks)
    - CausalConditionalCFM (256 channels, 4 blocks)
  HiFTGenerator:
    - Upsampling factors: [8, 5, 3]
    - Output: 24kHz audio
File: s3gen.safetensors (1.06 GB)
```

### VoiceEncoder
```yaml
Architecture: CAMPPlus
Output: 256-dim speaker embedding
File: ve.safetensors (5.7 MB)
```

## Optimization Priorities

### Priority 1: S3Gen TensorRT (HIGHEST)
**Expected Impact: 40-50% total latency reduction**

```bash
# Export S3Gen to ONNX
python scripts/optimization/export_s3gen.py

# Build TensorRT engine with FP16
trtexec \
    --onnx=./models/s3gen.onnx \
    --saveEngine=./engines/s3gen_fp16.plan \
    --fp16 \
    --minShapes=speech_tokens:1x10,speaker_emb:1x256 \
    --optShapes=speech_tokens:1x200,speaker_emb:1x256 \
    --maxShapes=speech_tokens:4x500,speaker_emb:4x256 \
    --workspace=8192
```

### Priority 2: T3 vLLM Integration
**Expected Impact: 20-30% token generation speedup**

```bash
# CRITICAL: Must use exactly vLLM 0.9.2
pip install vllm==0.9.2

# Install chatterbox-vllm
cd repos/chatterbox-vllm
pip install -e .

# Set CFG scale (cannot be changed per-request in vLLM)
export CHATTERBOX_CFG_SCALE=0.5
```

### Priority 3: ONNX Runtime Optimization
**Expected Impact: 15-25% overall speedup**

Use official ONNX models from HuggingFace:
```
ResembleAI/chatterbox-turbo-ONNX/
├── onnx/
│   ├── language_model_fp16.onnx    (635 MB)
│   ├── conditional_decoder_fp16.onnx (384 MB)
│   └── speech_encoder_fp16.onnx    (522 MB)
```

## Scripts Location

All optimization scripts go in `/scripts/optimization/`:

### export_onnx.py
Export PyTorch models to ONNX format with dynamic shapes.

### convert_tensorrt.py
Convert ONNX models to TensorRT engines.

### optimize_s3gen.py
S3Gen-specific optimization and profiling.

### onnx_inference.py
ONNX Runtime inference wrapper with TensorRT EP.

## Key Implementation Details

### ONNX Export with Dynamic Shapes
```python
import torch
from chatterbox import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Export with dynamic axes for variable sequence lengths
dynamic_axes = {
    'input_ids': {0: 'batch', 1: 'sequence'},
    'attention_mask': {0: 'batch', 1: 'sequence'},
    'output': {0: 'batch', 1: 'sequence'}
}

torch.onnx.export(
    model.t3,
    dummy_input,
    "models/t3.onnx",
    opset_version=17,
    dynamic_axes=dynamic_axes,
    input_names=['input_ids', 'attention_mask'],
    output_names=['output']
)
```

### TensorRT Engine Building
```bash
# For T3 (language model)
trtexec \
    --onnx=./models/chatterbox-turbo-onnx/onnx/language_model_fp16.onnx \
    --saveEngine=./engines/t3_fp16.plan \
    --fp16 \
    --workspace=8192

# For S3Gen (conditional decoder)
trtexec \
    --onnx=./models/chatterbox-turbo-onnx/onnx/conditional_decoder_fp16.onnx \
    --saveEngine=./engines/s3gen_fp16.plan \
    --fp16 \
    --minShapes=input:1x10x256 \
    --optShapes=input:1x200x256 \
    --maxShapes=input:4x500x256 \
    --workspace=8192
```

### ONNX Runtime with TensorRT EP
```python
import onnxruntime as ort

# Configure TensorRT Execution Provider
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 8 * 1024 * 1024 * 1024,  # 8GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './engines/cache'
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider'
]

session = ort.InferenceSession(
    "models/chatterbox-turbo-onnx/onnx/language_model_fp16.onnx",
    providers=providers
)
```

### vLLM Integration
```python
# From chatterbox-vllm
from chatterbox_vllm import ChatterboxVLLM

# Initialize with vLLM backend
model = ChatterboxVLLM(
    model_path="ResembleAI/chatterbox-turbo",
    device="cuda"
)

# CFG is controlled via environment variable
# export CHATTERBOX_CFG_SCALE=0.5

# Generate audio
audio = model.generate("Hello, this is a test.")
```

## Hybrid Pipeline (Recommended)

Combine best optimizations:

```python
class OptimizedChatterbox:
    def __init__(self):
        # Use vLLM for T3 (token generation)
        self.t3 = ChatterboxVLLM(...)

        # Use TensorRT for S3Gen (audio synthesis)
        self.s3gen = TensorRTEngine("engines/s3gen_fp16.plan")

        # Keep VoiceEncoder as-is (already fast)
        self.voice_encoder = VoiceEncoder.from_pretrained(...)

    def synthesize(self, text: str, speaker_audio: np.ndarray):
        # Get speaker embedding
        speaker_emb = self.voice_encoder(speaker_audio)

        # Generate speech tokens with vLLM
        tokens = self.t3.generate(text, speaker_emb)

        # Convert to audio with TensorRT S3Gen
        audio = self.s3gen.inference(tokens, speaker_emb)

        return audio
```

## Profiling Commands

### Profile S3Gen
```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Run S3Gen inference
    audio = model.s3gen(speech_tokens, speaker_embedding)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("s3gen_trace.json")
```

### Memory Profiling
```python
import torch

torch.cuda.reset_peak_memory_stats()

# Run inference
audio = model.synthesize(text)
torch.cuda.synchronize()

peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak_memory:.2f} GB")
```

## Known Issues & Solutions

### Issue 1: vLLM CFG Limitation
**Problem**: Cannot tune CFG (Classifier-Free Guidance) per-request
**Solution**: Set via environment variable before starting server
```bash
export CHATTERBOX_CFG_SCALE=0.5
```

### Issue 2: vLLM Version Lock
**Problem**: Only vLLM 0.9.2 is compatible with chatterbox-vllm
**Solution**: Pin version exactly
```bash
pip install vllm==0.9.2
```

### Issue 3: S3Gen ONNX Export Complexity
**Problem**: S3Gen has complex control flow that may fail ONNX export
**Solution**: Export submodules separately:
- S3Token2Mel.encoder
- S3Token2Mel.cfm
- HiFTGenerator

### Issue 4: TensorRT Dynamic Shapes
**Problem**: Shape mismatch errors at runtime
**Solution**: Always specify min/opt/max shape profiles
```bash
--minShapes=input:1x10x256 \
--optShapes=input:1x200x256 \
--maxShapes=input:4x500x256
```

### Issue 5: Audio Quality Degradation
**Problem**: FP16 causes audio artifacts
**Solution**:
- Try mixed precision (FP16 compute, FP32 accumulation)
- Check input/output normalization
- Verify dynamic shape handling

## Validation Checklist

After optimization, verify:

- [ ] ONNX models load without errors
- [ ] TensorRT engines build successfully
- [ ] Inference produces valid audio output
- [ ] Audio quality is acceptable (no artifacts, correct pronunciation)
- [ ] Latency improved by at least 30%
- [ ] Memory usage is within A10G 24GB limit
- [ ] Streaming output works correctly
- [ ] Batch inference works (if needed)

## Performance Targets

| Optimization | Baseline | Target | Measurement |
|--------------|----------|--------|-------------|
| S3Gen TensorRT | 450ms | 250ms | P50 latency |
| T3 vLLM | 150ms | 80ms | Token generation |
| Full Pipeline | 650ms | 380ms | End-to-end |
| Memory | 12GB | 10GB | Peak VRAM |

## Next Steps

After optimization is complete:
1. Run comprehensive benchmarks (`agents/benchmarking/CLAUDE.md`)
2. Proceed to deployment (`agents/deployment/CLAUDE.md`)
