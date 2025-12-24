#!/usr/bin/env python3
"""
streaming.py - Chatterbox Streaming Benchmark

Measures Time to First Chunk (TTFC) - the key metric for perceived latency
in real-time applications like telephony.

Usage:
    python scripts/benchmark/streaming.py
"""
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class StreamingConfig:
    """Streaming benchmark configuration"""
    warmup_runs: int = 3
    measurement_runs: int = 15
    chunk_sizes: tuple = (25, 50, 100)  # tokens per chunk
    sample_rate: int = 24000


@dataclass
class ChunkSizeResult:
    """Results for a specific chunk size"""
    chunk_size: int
    chunk_duration_s: float  # Approximate audio duration per chunk
    ttfc_mean_ms: float      # Time to first chunk
    ttfc_std_ms: float
    ttfc_p50_ms: float
    ttfc_p95_ms: float
    total_time_mean_s: float
    avg_chunks: float
    rtf: float


# Test text - typical telephony greeting
TEST_TEXT = """Welcome to our customer service line. I'm here to help you with your account today.
How may I assist you? Please let me know if you have any questions about your recent transactions
or would like to make a payment.""".replace("\n", " ")


def get_system_info() -> Dict:
    """Get system information"""
    import torch

    gpu_props = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(gpu_props.total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def benchmark_chunk_size(
    model,
    text: str,
    chunk_size: int,
    config: StreamingConfig
) -> ChunkSizeResult:
    """Benchmark streaming with a specific chunk size"""
    import torch
    import numpy as np

    ttfc_latencies = []  # Time to first chunk
    total_times = []
    chunk_counts = []
    total_audio_samples = []

    # Warmup
    print(f"    Warmup ({config.warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(config.warmup_runs):
        for chunk in model.generate_stream(text, chunk_size=chunk_size):
            pass
        torch.cuda.synchronize()
    print("done")

    # Measurement
    print(f"    Measuring ({config.measurement_runs} runs):")
    for i in range(config.measurement_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        first_chunk_time = None
        chunks = 0
        audio_samples = 0

        for audio_chunk in model.generate_stream(text, chunk_size=chunk_size):
            torch.cuda.synchronize()
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start

            chunks += 1
            # Handle both tensor and numpy outputs
            if hasattr(audio_chunk, 'shape'):
                audio_samples += audio_chunk.shape[-1]
            elif hasattr(audio_chunk, '__len__'):
                audio_samples += len(audio_chunk)

        total_time = time.perf_counter() - start

        ttfc_latencies.append(first_chunk_time * 1000)
        total_times.append(total_time)
        chunk_counts.append(chunks)
        total_audio_samples.append(audio_samples)

        if (i + 1) % 5 == 0:
            print(f"      Run {i+1:2d}: TTFC={first_chunk_time*1000:.0f}ms, "
                  f"total={total_time:.2f}s, chunks={chunks}")

    # Calculate audio duration and RTF
    mean_audio_samples = np.mean(total_audio_samples)
    mean_audio_duration = mean_audio_samples / config.sample_rate
    mean_total_time = np.mean(total_times)
    rtf = mean_total_time / mean_audio_duration if mean_audio_duration > 0 else 0

    # Approximate chunk duration (tokens at 25Hz)
    chunk_duration_s = chunk_size / 25.0

    return ChunkSizeResult(
        chunk_size=chunk_size,
        chunk_duration_s=round(chunk_duration_s, 2),
        ttfc_mean_ms=round(float(np.mean(ttfc_latencies)), 1),
        ttfc_std_ms=round(float(np.std(ttfc_latencies)), 1),
        ttfc_p50_ms=round(float(np.percentile(ttfc_latencies, 50)), 1),
        ttfc_p95_ms=round(float(np.percentile(ttfc_latencies, 95)), 1),
        total_time_mean_s=round(float(mean_total_time), 2),
        avg_chunks=round(float(np.mean(chunk_counts)), 1),
        rtf=round(rtf, 3),
    )


def run_streaming_benchmark(config: StreamingConfig = None) -> Dict:
    """Run the complete streaming benchmark"""
    import torch

    if config is None:
        config = StreamingConfig()

    print("=" * 70)
    print("CHATTERBOX STREAMING BENCHMARK")
    print("=" * 70)
    print()

    # System info
    print("System Information:")
    print("-" * 70)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    print()

    # Check for streaming support
    print("Loading streaming model...")
    print("-" * 70)

    try:
        from chatterbox_streaming import ChatterboxStreamingTTS
        model = ChatterboxStreamingTTS.from_pretrained(device="cuda")
        print("Streaming model loaded successfully!")
    except ImportError:
        print()
        print("ERROR: chatterbox-streaming not installed!")
        print()
        print("To install, run:")
        print("  pip install git+https://github.com/davidbrowne17/chatterbox-streaming.git")
        print()
        return None

    print(f"VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # Test text info
    print("Test Configuration:")
    print("-" * 70)
    print(f"  Text length: {len(TEST_TEXT)} chars, {len(TEST_TEXT.split())} words")
    print(f"  Text: \"{TEST_TEXT[:80]}...\"")
    print(f"  Chunk sizes to test: {config.chunk_sizes}")
    print()

    # Run benchmarks for each chunk size
    print("Running Benchmarks:")
    print("-" * 70)

    results: List[ChunkSizeResult] = []

    for chunk_size in config.chunk_sizes:
        print(f"\n  [Chunk size: {chunk_size} tokens (~{chunk_size/25:.1f}s audio per chunk)]")

        result = benchmark_chunk_size(model, TEST_TEXT, chunk_size, config)
        results.append(result)

        print(f"    Result: TTFC mean={result.ttfc_mean_ms:.0f}ms, "
              f"p95={result.ttfc_p95_ms:.0f}ms, RTF={result.rtf:.3f}")

    # Find best configuration
    best_result = min(results, key=lambda r: r.ttfc_mean_ms)

    # Print summary
    print()
    print("=" * 70)
    print("STREAMING BENCHMARK SUMMARY")
    print("=" * 70)
    print()

    print(f"{'Chunk Size':>12} | {'Chunk Dur':>10} | {'TTFC Mean':>10} | "
          f"{'TTFC P95':>10} | {'RTF':>6} | {'Chunks':>7}")
    print(f"{'(tokens)':>12} | {'(sec)':>10} | {'(ms)':>10} | "
          f"{'(ms)':>10} | {'':>6} | {'':>7}")
    print("-" * 70)

    for r in results:
        marker = " <-- BEST" if r.chunk_size == best_result.chunk_size else ""
        print(f"{r.chunk_size:>12} | {r.chunk_duration_s:>10.2f} | "
              f"{r.ttfc_mean_ms:>10.0f} | {r.ttfc_p95_ms:>10.0f} | "
              f"{r.rtf:>6.3f} | {r.avg_chunks:>7.1f}{marker}")

    print()
    print(f"  Best Configuration: chunk_size={best_result.chunk_size}")
    print(f"  Best TTFC: {best_result.ttfc_mean_ms:.0f}ms (p95: {best_result.ttfc_p95_ms:.0f}ms)")
    print()

    # Comparison to target
    target_ttfc = 400
    if best_result.ttfc_mean_ms <= target_ttfc:
        print(f"  Status: TARGET MET! (TTFC {best_result.ttfc_mean_ms:.0f}ms <= {target_ttfc}ms)")
    else:
        gap = best_result.ttfc_mean_ms - target_ttfc
        print(f"  Status: Need to reduce TTFC by {gap:.0f}ms to meet <{target_ttfc}ms target")

    # Streaming vs non-streaming comparison note
    print()
    print("  Note: Streaming reduces PERCEIVED latency by delivering audio")
    print("        in chunks while generation continues in background.")
    print("        TTFC is the key metric for real-time telephony applications.")

    # Save results
    output_dir = PROJECT_ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"streaming_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "streaming",
            "warmup_runs": config.warmup_runs,
            "measurement_runs": config.measurement_runs,
            "test_text_length": len(TEST_TEXT),
        },
        "system": sys_info,
        "best_config": {
            "chunk_size": best_result.chunk_size,
            "ttfc_mean_ms": best_result.ttfc_mean_ms,
            "ttfc_p95_ms": best_result.ttfc_p95_ms,
        },
        "results": {str(r.chunk_size): asdict(r) for r in results},
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print()

    return output_data


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Chatterbox Streaming Benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=15, help="Number of measurement runs")
    parser.add_argument("--chunks", type=str, default="25,50,100",
                        help="Comma-separated chunk sizes to test")
    args = parser.parse_args()

    chunk_sizes = tuple(int(x) for x in args.chunks.split(","))

    config = StreamingConfig(
        warmup_runs=args.warmup,
        measurement_runs=args.runs,
        chunk_sizes=chunk_sizes,
    )

    try:
        run_streaming_benchmark(config)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
