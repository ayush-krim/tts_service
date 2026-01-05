#!/usr/bin/env python3
"""
baseline.py - Chatterbox Turbo Baseline Benchmark on A10G

This script measures baseline (non-optimized) performance metrics:
- End-to-end latency
- Real-Time Factor (RTF)
- GPU memory usage

Usage:
    python scripts/benchmark/baseline.py
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
class BenchmarkConfig:
    """Benchmark configuration"""
    warmup_runs: int = 5
    measurement_runs: int = 20
    sample_rate: int = 24000


@dataclass
class LatencyStats:
    """Latency statistics"""
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class TextResult:
    """Result for a single text benchmark"""
    name: str
    text: str
    text_length_chars: int
    text_length_words: int
    audio_duration_s: float
    latency: LatencyStats
    rtf: float
    vram_peak_gb: float


# Test texts of varying lengths (typical telephony scenarios)
TEST_TEXTS = {
    "ultra_short": "Hello there.",

    "short": "Hi, this is Sarah from customer service. How can I help you today?",

    "medium": "Thank you for calling. I can see your account here. Let me check the details of your recent transaction and explain what happened with your payment.",

    "long": "I understand you're concerned about the charges on your account. Let me walk you through each item on your bill. First, there's your monthly subscription fee, then we have the usage charges from last month, and finally a small processing fee that applies to all accounts.",

    "telephony_typical": "Hi, this is Alex from Krim AI support. Your current balance is forty-five dollars and thirty-two cents. Would you like me to process a payment for you today?"
}


def get_device() -> str:
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def get_system_info(device: str) -> Dict:
    """Get system information"""
    import torch
    import platform

    info = {
        "device": device,
        "pytorch_version": torch.__version__,
        "platform": platform.system(),
        "processor": platform.processor() or platform.machine(),
    }

    if device == "cuda" and torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(gpu_props.total_memory / 1e9, 1),
            "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        })
    elif device == "mps":
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["gpu"] = "None (CPU only)"

    return info


def benchmark_text(
    model,
    text: str,
    name: str,
    config: BenchmarkConfig,
    device: str = "cuda"
) -> TextResult:
    """Benchmark a single text input"""
    import torch
    import numpy as np

    latencies = []
    audio_durations = []
    use_cuda = device == "cuda" and torch.cuda.is_available()

    def sync():
        """Synchronize device if needed."""
        if use_cuda:
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # Warmup runs
    print(f"    Warmup ({config.warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(config.warmup_runs):
        _ = model.generate(text)
        sync()
    print("done")

    # Clear memory stats
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Measurement runs
    print(f"    Measuring ({config.measurement_runs} runs):", flush=True)
    for i in range(config.measurement_runs):
        sync()
        start = time.perf_counter()

        wav = model.generate(text)

        sync()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        audio_duration = wav.shape[-1] / config.sample_rate

        latencies.append(latency_ms)
        audio_durations.append(audio_duration)

        # Progress indicator every 5 runs
        if (i + 1) % 5 == 0:
            print(f"      Run {i+1:2d}/{config.measurement_runs}: "
                  f"{latency_ms:.0f}ms, RTF={latency_ms/1000/audio_duration:.3f}")

    # Calculate statistics
    mean_latency = float(np.mean(latencies))
    mean_audio_duration = float(np.mean(audio_durations))
    rtf = (mean_latency / 1000) / mean_audio_duration

    latency_stats = LatencyStats(
        mean_ms=round(mean_latency, 1),
        std_ms=round(float(np.std(latencies)), 1),
        p50_ms=round(float(np.percentile(latencies, 50)), 1),
        p95_ms=round(float(np.percentile(latencies, 95)), 1),
        p99_ms=round(float(np.percentile(latencies, 99)), 1),
        min_ms=round(float(np.min(latencies)), 1),
        max_ms=round(float(np.max(latencies)), 1),
    )

    # Get VRAM usage (only on CUDA)
    vram_gb = 0.0
    if use_cuda:
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    return TextResult(
        name=name,
        text=text,
        text_length_chars=len(text),
        text_length_words=len(text.split()),
        audio_duration_s=round(mean_audio_duration, 2),
        latency=latency_stats,
        rtf=round(rtf, 3),
        vram_peak_gb=vram_gb,
    )


def run_baseline_benchmark(config: BenchmarkConfig = None) -> Dict:
    """Run the complete baseline benchmark"""
    import torch

    if config is None:
        config = BenchmarkConfig()

    # Detect device
    device = get_device()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    print("=" * 70)
    print("CHATTERBOX TURBO BASELINE BENCHMARK")
    print("=" * 70)
    print()

    if device != "cuda":
        print(f"WARNING: Running on {device.upper()}. For accurate benchmarks,")
        print("         run on AWS G5 (A10G GPU) with CUDA support.")
        print()

    # System info
    print("System Information:")
    print("-" * 70)
    sys_info = get_system_info(device)
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    print()

    # Load model
    print(f"Loading Chatterbox Turbo model on {device}...")
    print("-" * 70)
    load_start = time.perf_counter()

    # Try different import paths for Chatterbox
    model = None
    try:
        # Resemble AI's chatterbox-tts package (newer)
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
    except ImportError:
        try:
            # Alternative import path
            from chatterbox import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        except ImportError:
            print("\nERROR: Chatterbox TTS package not found.")
            print("\nTo install Resemble AI's Chatterbox:")
            print("  pip install chatterbox-tts")
            print("\nOr install from source:")
            print("  pip install git+https://github.com/resemble-ai/chatterbox.git")
            print("\nNote: There may be a package name conflict with another 'chatterbox'.")
            print("If so, uninstall the other one first:")
            print("  pip uninstall chatterbox")
            print("  pip install chatterbox-tts")
            sys.exit(1)

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    if use_cuda:
        print(f"Initial VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # Run benchmarks
    print("Running Benchmarks:")
    print("-" * 70)

    results: List[TextResult] = []

    for name, text in TEST_TEXTS.items():
        print(f"\n  [{name}] ({len(text)} chars, {len(text.split())} words)")
        print(f"    Text: \"{text[:60]}...\"" if len(text) > 60 else f"    Text: \"{text}\"")

        result = benchmark_text(model, text, name, config, device=device)
        results.append(result)

        print(f"    Result: mean={result.latency.mean_ms:.0f}ms, "
              f"p95={result.latency.p95_ms:.0f}ms, "
              f"RTF={result.rtf:.3f}")

    # Calculate summary
    import numpy as np

    all_means = [r.latency.mean_ms for r in results]
    all_p95s = [r.latency.p95_ms for r in results]
    all_rtfs = [r.rtf for r in results]
    all_vram = [r.vram_peak_gb for r in results]

    summary = {
        "overall_mean_latency_ms": round(float(np.mean(all_means)), 1),
        "overall_p95_latency_ms": round(float(max(all_p95s)), 1),
        "overall_mean_rtf": round(float(np.mean(all_rtfs)), 3),
        "max_vram_gb": round(float(max(all_vram)), 2),
        "real_time_capable": all(r.rtf < 1.0 for r in results),
    }

    # Print summary
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print(f"  Device:                {device.upper()}")
    print(f"  Overall Mean Latency:  {summary['overall_mean_latency_ms']:.0f} ms")
    print(f"  Overall P95 Latency:   {summary['overall_p95_latency_ms']:.0f} ms")
    print(f"  Overall Mean RTF:      {summary['overall_mean_rtf']:.3f}")
    if use_cuda:
        print(f"  Peak VRAM Usage:       {summary['max_vram_gb']:.1f} GB")
    print()

    if summary['real_time_capable']:
        print("  Status: REAL-TIME CAPABLE (all RTF < 1.0)")
    else:
        print("  Status: NOT REAL-TIME (some RTF >= 1.0)")

    # Comparison to targets
    print()
    print("  Comparison to Targets:")
    print("  -----------------------")
    target_latency = 400
    target_rtf = 0.5

    latency_gap = summary['overall_mean_latency_ms'] - target_latency
    rtf_gap = summary['overall_mean_rtf'] - target_rtf

    if latency_gap > 0:
        print(f"    Latency: {summary['overall_mean_latency_ms']:.0f}ms "
              f"(need to reduce by {latency_gap:.0f}ms to hit <{target_latency}ms target)")
    else:
        print(f"    Latency: {summary['overall_mean_latency_ms']:.0f}ms (TARGET MET!)")

    if rtf_gap > 0:
        print(f"    RTF: {summary['overall_mean_rtf']:.3f} "
              f"(need to reduce by {rtf_gap:.3f} to hit <{target_rtf} target)")
    else:
        print(f"    RTF: {summary['overall_mean_rtf']:.3f} (TARGET MET!)")

    # Detailed results table
    print()
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print()
    print(f"{'Text Type':<20} | {'Chars':>6} | {'Words':>5} | "
          f"{'Mean':>8} | {'P95':>8} | {'RTF':>6} | {'Audio':>6}")
    print(f"{'':<20} | {'':>6} | {'':>5} | "
          f"{'(ms)':>8} | {'(ms)':>8} | {'':>6} | {'(sec)':>6}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<20} | {r.text_length_chars:>6} | {r.text_length_words:>5} | "
              f"{r.latency.mean_ms:>8.0f} | {r.latency.p95_ms:>8.0f} | "
              f"{r.rtf:>6.3f} | {r.audio_duration_s:>6.1f}")

    # Save results
    output_dir = PROJECT_ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "baseline",
            "warmup_runs": config.warmup_runs,
            "measurement_runs": config.measurement_runs,
            "model_load_time_s": round(load_time, 1),
        },
        "system": sys_info,
        "summary": summary,
        "results": {r.name: asdict(r) for r in results},
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

    parser = argparse.ArgumentParser(description="Chatterbox Turbo Baseline Benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=20, help="Number of measurement runs")
    args = parser.parse_args()

    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        measurement_runs=args.runs,
    )

    try:
        run_baseline_benchmark(config)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
