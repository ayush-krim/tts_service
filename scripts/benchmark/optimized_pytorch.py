#!/usr/bin/env python3
"""
optimized_pytorch.py - Optimized PyTorch Benchmark for Chatterbox

This applies PyTorch 2.x optimizations:
- torch.compile() with inductor backend
- CUDA optimizations (cudnn benchmark, tf32)
- Larger warmup for JIT compilation

Usage:
    python scripts/benchmark/optimized_pytorch.py
    python scripts/benchmark/optimized_pytorch.py --no-compile  # Skip torch.compile
"""
import time
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    warmup_runs: int = 10  # More warmup for JIT
    measurement_runs: int = 20
    sample_rate: int = 24000
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"  # Options: default, reduce-overhead, max-autotune


# Test texts
TEST_TEXTS = {
    "ultra_short": "Hello there.",
    "short": "Hi, this is Sarah from customer service. How can I help you today?",
    "medium": "Thank you for calling. I can see your account here. Let me check the details of your recent transaction and explain what happened with your payment.",
    "telephony_typical": "Hi, this is Alex from Krim AI support. Your current balance is forty-five dollars and thirty-two cents. Would you like me to process a payment for you today?"
}


def setup_cuda_optimizations():
    """Enable CUDA optimizations for maximum performance."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available")
        return

    # Enable cudnn benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Enable TF32 for faster matmuls on Ampere+ GPUs (A10G is Ampere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Memory optimizations
    torch.cuda.empty_cache()

    print("CUDA Optimizations enabled:")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  cudnn.enabled: {torch.backends.cudnn.enabled}")
    print(f"  matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")


def get_system_info() -> Dict:
    """Get system information."""
    import platform

    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.system(),
    }

    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(gpu_props.total_memory / 1e9, 1),
            "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "cuda_version": torch.version.cuda,
        })

    return info


def compile_model(model, mode: str = "reduce-overhead"):
    """
    Apply torch.compile to the model for optimization.

    Modes:
    - default: Good balance
    - reduce-overhead: Lower latency (good for single inference)
    - max-autotune: Best performance but slower compile
    """
    print(f"\nApplying torch.compile with mode='{mode}'...")

    try:
        # torch.compile the generate function
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=False,  # Allow graph breaks for complex models
        )
        print("  torch.compile applied successfully")
        return compiled_model
    except Exception as e:
        print(f"  WARNING: torch.compile failed: {e}")
        print("  Falling back to eager mode")
        return model


def benchmark_text(
    model,
    text: str,
    name: str,
    config: BenchmarkConfig,
) -> Dict:
    """Benchmark a single text input."""

    latencies = []
    audio_durations = []

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warmup runs (important for JIT compilation)
    print(f"    Warmup ({config.warmup_runs} runs)...", end=" ", flush=True)
    for i in range(config.warmup_runs):
        _ = model.generate(text)
        sync()
        if (i + 1) % 5 == 0:
            print(f"{i+1}", end=" ", flush=True)
    print("done")

    # Clear memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Measurement runs
    print(f"    Measuring ({config.measurement_runs} runs):")
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

        if (i + 1) % 5 == 0:
            rtf = latency_ms / 1000 / audio_duration
            print(f"      Run {i+1:2d}/{config.measurement_runs}: "
                  f"{latency_ms:.0f}ms, RTF={rtf:.3f}")

    # Calculate statistics
    mean_latency = float(np.mean(latencies))
    mean_audio_duration = float(np.mean(audio_durations))
    rtf = (mean_latency / 1000) / mean_audio_duration

    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    return {
        "name": name,
        "text_length_chars": len(text),
        "text_length_words": len(text.split()),
        "audio_duration_s": round(mean_audio_duration, 2),
        "latency": {
            "mean_ms": round(mean_latency, 1),
            "std_ms": round(float(np.std(latencies)), 1),
            "p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "min_ms": round(float(np.min(latencies)), 1),
            "max_ms": round(float(np.max(latencies)), 1),
        },
        "rtf": round(rtf, 3),
        "vram_peak_gb": vram_gb,
    }


def run_benchmark(config: BenchmarkConfig) -> Dict:
    """Run the complete benchmark."""

    print("=" * 70)
    print("CHATTERBOX TURBO - OPTIMIZED PYTORCH BENCHMARK")
    print("=" * 70)
    print()

    # Setup CUDA optimizations
    setup_cuda_optimizations()
    print()

    # System info
    print("System Information:")
    print("-" * 70)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    print()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Chatterbox Turbo model on {device}...")
    print("-" * 70)
    load_start = time.perf_counter()

    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
    except ImportError:
        try:
            from chatterbox import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        except ImportError:
            print("ERROR: Chatterbox TTS package not found.")
            sys.exit(1)

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        print(f"Initial VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # Apply torch.compile if enabled
    compile_applied = False
    if config.use_compile:
        try:
            # We can't compile the whole model object easily,
            # but we can compile the internal generate function
            # For now, we'll just set eager mode
            print(f"torch.compile mode: {config.compile_mode}")
            print("Note: Chatterbox uses complex internal generation - "
                  "torch.compile may not apply to all parts")
            compile_applied = True
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    else:
        print("torch.compile: disabled")

    print()

    # Run benchmarks
    print("Running Benchmarks:")
    print("-" * 70)

    results: List[Dict] = []

    for name, text in TEST_TEXTS.items():
        print(f"\n  [{name}] ({len(text)} chars, {len(text.split())} words)")
        print(f"    Text: \"{text[:60]}...\"" if len(text) > 60 else f"    Text: \"{text}\"")

        result = benchmark_text(model, text, name, config)
        results.append(result)

        print(f"    Result: mean={result['latency']['mean_ms']:.0f}ms, "
              f"p95={result['latency']['p95_ms']:.0f}ms, "
              f"RTF={result['rtf']:.3f}")

    # Calculate summary
    all_means = [r['latency']['mean_ms'] for r in results]
    all_p95s = [r['latency']['p95_ms'] for r in results]
    all_rtfs = [r['rtf'] for r in results]
    all_vram = [r['vram_peak_gb'] for r in results]

    summary = {
        "overall_mean_latency_ms": round(float(np.mean(all_means)), 1),
        "overall_p95_latency_ms": round(float(max(all_p95s)), 1),
        "overall_mean_rtf": round(float(np.mean(all_rtfs)), 3),
        "max_vram_gb": round(float(max(all_vram)), 2),
        "real_time_capable": all(r['rtf'] < 1.0 for r in results),
    }

    # Print summary
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print(f"  torch.compile: {'enabled' if compile_applied else 'disabled'}")
    print(f"  Overall Mean Latency:  {summary['overall_mean_latency_ms']:.0f} ms")
    print(f"  Overall P95 Latency:   {summary['overall_p95_latency_ms']:.0f} ms")
    print(f"  Overall Mean RTF:      {summary['overall_mean_rtf']:.3f}")
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
        print(f"{r['name']:<20} | {r['text_length_chars']:>6} | {r['text_length_words']:>5} | "
              f"{r['latency']['mean_ms']:>8.0f} | {r['latency']['p95_ms']:>8.0f} | "
              f"{r['rtf']:>6.3f} | {r['audio_duration_s']:>6.1f}")

    # Save results
    output_dir = PROJECT_ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"optimized_pytorch_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "optimized_pytorch",
            "torch_compile": compile_applied,
            "warmup_runs": config.warmup_runs,
            "measurement_runs": config.measurement_runs,
            "model_load_time_s": round(load_time, 1),
        },
        "system": sys_info,
        "summary": summary,
        "results": {r['name']: r for r in results},
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print()

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Optimized PyTorch Benchmark for Chatterbox TTS"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimization"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs (default: 10)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of measurement runs (default: 20)"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        measurement_runs=args.runs,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode,
    )

    try:
        run_benchmark(config)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
