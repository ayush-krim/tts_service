# Benchmarking Agent

## Role

Measure, track, and report performance metrics for Chatterbox TTS optimization.

## Key Metrics

| Metric | Definition | Target | Stretch |
|--------|------------|--------|---------|
| TTFC (Time to First Chunk) | Latency until first audio chunk | < 400ms | < 300ms |
| RTF (Real-Time Factor) | Generation time / Audio duration | < 0.50 | < 0.40 |
| P50 Latency | Median end-to-end latency | < 450ms | < 350ms |
| P95 Latency | 95th percentile latency | < 500ms | < 400ms |
| P99 Latency | 99th percentile latency | < 750ms | < 600ms |
| Throughput | Concurrent streams per GPU | > 3 | > 5 |
| VRAM Usage | Peak GPU memory consumption | < 16GB | < 12GB |

## Benchmark Scripts

All scripts located in `/scripts/benchmark/`:

### baseline.py
Measures vanilla PyTorch inference without optimization.

### streaming.py
Measures time-to-first-chunk with streaming output.

### onnx_benchmark.py
Compares ONNX Runtime vs PyTorch.

### vllm_benchmark.py
Benchmarks vLLM-accelerated inference.

### load_test.py
Concurrent request testing with Locust.

## Standard Test Texts

**ALWAYS use these for consistent comparison:**

```python
TEST_TEXTS = {
    "ultra_short": "Hello there.",  # ~2 words

    "short": "Hi, this is Sarah from customer service. How can I help you today?",  # ~12 words

    "medium": "Thank you for calling. I can see your account here. Let me check the details of your recent transaction and explain what happened with your payment.",  # ~28 words

    "long": "I understand you're concerned about the charges on your account. Let me walk you through each item on your bill. First, there's your monthly subscription fee, then we have the usage charges from last month, and finally a small processing fee that applies to all accounts.",  # ~50 words

    "telephony_typical": "Hi, this is Alex from Krim AI support. I can see your account details here. Your current balance is forty-five dollars and thirty-two cents. Would you like me to process a payment for you today?"  # ~38 words - typical telephony scenario
}
```

## Benchmark Protocol

### 1. Environment Preparation
```python
import torch
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Set deterministic mode for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 2. Warmup Phase
```python
WARMUP_RUNS = 5

print("Running warmup...")
for i in range(WARMUP_RUNS):
    _ = model.synthesize(TEST_TEXTS["medium"])
    torch.cuda.synchronize()
print("Warmup complete")
```

### 3. Measurement Phase
```python
import time
import numpy as np

MEASUREMENT_RUNS = 20

latencies = []
for i in range(MEASUREMENT_RUNS):
    torch.cuda.synchronize()
    start = time.perf_counter()

    audio = model.synthesize(text)

    torch.cuda.synchronize()
    end = time.perf_counter()

    latencies.append((end - start) * 1000)  # Convert to ms

# Calculate statistics
results = {
    "mean_ms": np.mean(latencies),
    "std_ms": np.std(latencies),
    "p50_ms": np.percentile(latencies, 50),
    "p95_ms": np.percentile(latencies, 95),
    "p99_ms": np.percentile(latencies, 99),
    "min_ms": np.min(latencies),
    "max_ms": np.max(latencies)
}
```

### 4. RTF Calculation
```python
def calculate_rtf(latency_ms: float, audio_samples: int, sample_rate: int = 24000) -> float:
    """
    Real-Time Factor = Generation Time / Audio Duration
    RTF < 1.0 means faster than real-time
    """
    audio_duration_ms = (audio_samples / sample_rate) * 1000
    return latency_ms / audio_duration_ms
```

### 5. GPU Memory Tracking
```python
import torch

torch.cuda.reset_peak_memory_stats()

# Run inference
audio = model.synthesize(text)
torch.cuda.synchronize()

peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
current_memory_gb = torch.cuda.memory_allocated() / 1e9

print(f"Peak VRAM: {peak_memory_gb:.2f} GB")
print(f"Current VRAM: {current_memory_gb:.2f} GB")
```

## Output Format

Save all benchmarks as JSON to `/benchmarks/results/`:

```json
{
  "metadata": {
    "timestamp": "2024-12-23T10:00:00Z",
    "benchmark_type": "baseline",
    "warmup_runs": 5,
    "measurement_runs": 20
  },
  "configuration": {
    "gpu": "NVIDIA A10G",
    "gpu_memory_gb": 24,
    "cuda_version": "12.4",
    "pytorch_version": "2.1.0",
    "model": "chatterbox-turbo",
    "optimization": "none",
    "batch_size": 1
  },
  "results": {
    "ultra_short": {
      "text_length_words": 2,
      "audio_duration_ms": 800,
      "latency": {
        "mean_ms": 420,
        "std_ms": 25,
        "p50_ms": 415,
        "p95_ms": 470,
        "p99_ms": 510,
        "min_ms": 390,
        "max_ms": 520
      },
      "rtf": 0.52,
      "vram_peak_gb": 11.2
    },
    "short": { ... },
    "medium": { ... },
    "long": { ... },
    "telephony_typical": { ... }
  },
  "summary": {
    "overall_mean_ms": 550,
    "overall_p95_ms": 680,
    "overall_rtf": 0.62,
    "max_vram_gb": 12.1
  }
}
```

## File Naming Convention

```
benchmarks/results/
├── baseline_20241223_100000.json
├── streaming_20241223_110000.json
├── onnx_cpu_20241223_120000.json
├── onnx_tensorrt_20241223_130000.json
├── vllm_20241223_140000.json
├── s3gen_tensorrt_20241223_150000.json
├── hybrid_pipeline_20241223_160000.json
└── final_optimized_20241223_170000.json
```

## Comparison Template

After each optimization phase, create comparison:

```
┌──────────────────────────────────────────────────────────────────────┐
│ BENCHMARK COMPARISON: [Phase Name]                                    │
│ Date: 2024-12-23                                                     │
│ Hardware: NVIDIA A10G (24GB)                                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Metric            Before       After        Change     Status       │
│  ──────────────────────────────────────────────────────────────────  │
│  Mean Latency      650ms        420ms        -35.4%     ✅ PASS      │
│  P95 Latency       780ms        510ms        -34.6%     ❌ MISS      │
│  RTF               0.75         0.48         -36.0%     ✅ PASS      │
│  TTFC              520ms        340ms        -34.6%     ✅ PASS      │
│  VRAM Usage        12GB         10GB         -16.7%     ✅ PASS      │
│                                                                       │
│  Target Status: 4/5 metrics passing                                  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Streaming Benchmark Protocol

For measuring time-to-first-chunk:

```python
import asyncio
import time

async def benchmark_streaming(model, text: str, runs: int = 20):
    ttfc_latencies = []
    total_latencies = []
    chunk_counts = []

    for _ in range(runs):
        chunks = []
        first_chunk_time = None

        start = time.perf_counter()

        async for chunk in model.synthesize_stream(text):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            chunks.append(chunk)

        end = time.perf_counter()

        ttfc_latencies.append((first_chunk_time - start) * 1000)
        total_latencies.append((end - start) * 1000)
        chunk_counts.append(len(chunks))

    return {
        "ttfc_mean_ms": np.mean(ttfc_latencies),
        "ttfc_p95_ms": np.percentile(ttfc_latencies, 95),
        "total_mean_ms": np.mean(total_latencies),
        "total_p95_ms": np.percentile(total_latencies, 95),
        "avg_chunks": np.mean(chunk_counts)
    }
```

## Load Testing Protocol

For concurrent request testing:

```python
# Using Locust (locustfile.py)
from locust import HttpUser, task, between

class TTSUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def synthesize(self):
        self.client.post("/v1/synthesize", json={
            "text": "Hi, this is a test message for load testing.",
            "speaker_id": "default"
        })
```

```bash
# Run load test
locust -f tests/locustfile.py \
    --host http://localhost:8080 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m
```

## Benchmark Script Template

```python
#!/usr/bin/env python3
"""
Chatterbox TTS Benchmark Script
Usage: python scripts/benchmark/baseline.py
"""

import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Test texts
TEST_TEXTS = {
    "ultra_short": "Hello there.",
    "short": "Hi, this is Sarah from customer service. How can I help you today?",
    "medium": "Thank you for calling. I can see your account here. Let me check the details of your recent transaction.",
    "long": "I understand you're concerned about the charges on your account. Let me walk you through each item on your bill.",
    "telephony_typical": "Hi, this is Alex from Krim AI support. Your current balance is forty-five dollars."
}

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 20

def get_gpu_info():
    return {
        "name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda
    }

def benchmark_text(model, text: str, text_name: str):
    """Benchmark a single text input"""
    latencies = []
    audio_samples = None

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = model.synthesize(text)
        torch.cuda.synchronize()

    # Clear memory stats
    torch.cuda.reset_peak_memory_stats()

    # Measurement
    for _ in range(MEASUREMENT_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        audio = model.synthesize(text)

        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)
        if audio_samples is None:
            audio_samples = len(audio)

    # Calculate RTF (assuming 24kHz)
    audio_duration_ms = (audio_samples / 24000) * 1000
    mean_latency = np.mean(latencies)
    rtf = mean_latency / audio_duration_ms

    return {
        "text_name": text_name,
        "text_length_words": len(text.split()),
        "audio_samples": audio_samples,
        "audio_duration_ms": audio_duration_ms,
        "latency": {
            "mean_ms": round(mean_latency, 2),
            "std_ms": round(np.std(latencies), 2),
            "p50_ms": round(np.percentile(latencies, 50), 2),
            "p95_ms": round(np.percentile(latencies, 95), 2),
            "p99_ms": round(np.percentile(latencies, 99), 2),
            "min_ms": round(np.min(latencies), 2),
            "max_ms": round(np.max(latencies), 2)
        },
        "rtf": round(rtf, 3),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
    }

def main():
    from chatterbox import ChatterboxTurboTTS

    print("Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "baseline",
            "warmup_runs": WARMUP_RUNS,
            "measurement_runs": MEASUREMENT_RUNS
        },
        "configuration": {
            **get_gpu_info(),
            "model": "chatterbox-turbo",
            "optimization": "none"
        },
        "results": {}
    }

    for name, text in TEST_TEXTS.items():
        print(f"Benchmarking: {name}")
        results["results"][name] = benchmark_text(model, text, name)

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for name, data in results["results"].items():
        print(f"{name:20} | Mean: {data['latency']['mean_ms']:6.1f}ms | RTF: {data['rtf']:.3f}")

if __name__ == "__main__":
    main()
```

## Validation Checklist

Before submitting benchmark results:

- [ ] GPU memory cleared before each run
- [ ] Warmup phase completed (5 runs)
- [ ] Sufficient measurement runs (20+)
- [ ] `torch.cuda.synchronize()` called for accurate timing
- [ ] Results saved as JSON with timestamp
- [ ] Comparison table generated
- [ ] All test texts benchmarked
- [ ] RTF calculated correctly
- [ ] VRAM usage recorded

## Next Steps

After benchmarking is complete:
1. Compare against targets in root `CLAUDE.md`
2. Identify remaining bottlenecks
3. Proceed to deployment (`agents/deployment/CLAUDE.md`)
