#!/usr/bin/env python3
"""
onnx_benchmark.py - ONNX Runtime Benchmark for Chatterbox TTS

Benchmarks the ONNX models using ONNX Runtime with CUDA Execution Provider.
This provides acceleration without requiring TensorRT.

Usage:
    python scripts/benchmark/onnx_benchmark.py
    python scripts/benchmark/onnx_benchmark.py --provider cpu  # Force CPU
"""
import time
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_onnx_models() -> Tuple[Optional[Path], Dict[str, Path]]:
    """Check for ONNX models and return paths."""
    models_dir = PROJECT_ROOT / "models"

    # Check for HuggingFace ONNX models
    onnx_dir = models_dir / "chatterbox-turbo-onnx" / "onnx"
    if not onnx_dir.exists():
        onnx_dir = models_dir / "onnx" / "chatterbox-turbo-onnx" / "onnx"

    if not onnx_dir.exists():
        return None, {}

    onnx_files = {}
    expected_files = [
        "language_model_fp16.onnx",
        "conditional_decoder_fp16.onnx",
        "speech_encoder_fp16.onnx",
    ]

    for fname in expected_files:
        fpath = onnx_dir / fname
        if fpath.exists():
            key = fname.replace("_fp16.onnx", "").replace(".onnx", "")
            onnx_files[key] = fpath

    return onnx_dir, onnx_files


def get_onnx_provider_info() -> Dict:
    """Get available ONNX Runtime execution providers."""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        info = {
            "onnxruntime_version": ort.__version__,
            "available_providers": available,
            "cuda_available": "CUDAExecutionProvider" in available,
            "tensorrt_available": "TensorrtExecutionProvider" in available,
        }

        return info
    except ImportError:
        return {"error": "onnxruntime not installed"}


def benchmark_onnx_model(
    model_path: Path,
    provider: str = "cuda",
    warmup_runs: int = 10,
    measurement_runs: int = 50,
) -> Dict:
    """
    Benchmark a single ONNX model.

    Returns timing statistics for inference.
    """
    import onnxruntime as ort

    print(f"\n  Loading: {model_path.name}")

    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Select providers
    if provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 16 * 1024 * 1024 * 1024,  # 16GB
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider"
        ]
    elif provider == "tensorrt":
        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": True,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]

    # Create session
    load_start = time.perf_counter()
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=providers
        )
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    load_time = time.perf_counter() - load_start
    print(f"  Loaded in {load_time:.2f}s")

    # Get actual provider being used
    actual_provider = session.get_providers()[0]
    print(f"  Provider: {actual_provider}")

    # Get input/output info
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"  Inputs: {[i.name for i in inputs]}")
    print(f"  Outputs: {[o.name for o in outputs]}")

    # Create dummy inputs based on model type
    input_data = {}
    for inp in inputs:
        shape = inp.shape
        # Replace dynamic dims with reasonable values
        resolved_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim is None or dim < 0:
                resolved_shape.append(16)  # Smaller default for speed
            else:
                resolved_shape.append(dim)

        # Determine dtype from ONNX type
        onnx_type = inp.type.lower()
        if "float16" in onnx_type or "16" in onnx_type:
            input_data[inp.name] = np.random.randn(*resolved_shape).astype(np.float16)
        elif "float" in onnx_type:
            input_data[inp.name] = np.random.randn(*resolved_shape).astype(np.float32)
        elif "int64" in onnx_type:
            input_data[inp.name] = np.random.randint(0, 1000, size=resolved_shape).astype(np.int64)
        elif "int32" in onnx_type:
            input_data[inp.name] = np.random.randint(0, 1000, size=resolved_shape).astype(np.int32)
        else:
            # Default to float32
            input_data[inp.name] = np.random.randn(*resolved_shape).astype(np.float32)

        print(f"  Input '{inp.name}': shape={resolved_shape}, dtype={input_data[inp.name].dtype}")

    # Warmup
    print(f"  Warmup ({warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(warmup_runs):
        _ = session.run(None, input_data)
    print("done")

    # Benchmark
    latencies = []
    print(f"  Measuring ({measurement_runs} runs)...")

    for i in range(measurement_runs):
        start = time.perf_counter()
        _ = session.run(None, input_data)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if (i + 1) % 10 == 0:
            print(f"    Run {i+1}: {latencies[-1]:.2f}ms")

    # Statistics
    stats = {
        "model": model_path.name,
        "provider": actual_provider,
        "load_time_s": round(load_time, 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
    }

    print(f"  Results: mean={stats['mean_ms']:.1f}ms, "
          f"p95={stats['p95_ms']:.1f}ms")

    return stats


def run_onnx_benchmark(provider: str = "cuda") -> Dict:
    """Run benchmark on all available ONNX models."""

    print("=" * 70)
    print("CHATTERBOX ONNX RUNTIME BENCHMARK")
    print("=" * 70)
    print()

    # Check ONNX Runtime
    print("ONNX Runtime Info:")
    print("-" * 70)
    ort_info = get_onnx_provider_info()
    for key, value in ort_info.items():
        print(f"  {key}: {value}")
    print()

    if "error" in ort_info:
        print("ERROR: onnxruntime not installed")
        print("Install with: pip install onnxruntime-gpu")
        return None

    # Check ONNX models
    print("ONNX Models:")
    print("-" * 70)
    onnx_dir, onnx_files = check_onnx_models()

    if not onnx_files:
        print("ERROR: No ONNX models found")
        print("\nExpected location: models/chatterbox-turbo-onnx/onnx/")
        print("Download with: python scripts/setup/download_models.py")
        return None

    print(f"  Directory: {onnx_dir}")
    for key, path in onnx_files.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {key}: {path.name} ({size_mb:.1f} MB)")

    # Run benchmarks
    print()
    print("Running Benchmarks:")
    print("=" * 70)

    results = {}

    for key, model_path in onnx_files.items():
        print(f"\n[{key}]")
        stats = benchmark_onnx_model(
            model_path,
            provider=provider,
            warmup_runs=10,
            measurement_runs=50
        )
        if stats:
            results[key] = stats

    # Summary
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<25} | {'Provider':>15} | {'Mean (ms)':>10} | {'P95 (ms)':>10}")
    print("-" * 70)

    total_mean = 0
    for key, stats in results.items():
        print(f"{key:<25} | {stats['provider']:>15} | {stats['mean_ms']:>10.1f} | {stats['p95_ms']:>10.1f}")
        total_mean += stats['mean_ms']

    print("-" * 70)
    print(f"{'TOTAL (sequential)':<25} | {'':<15} | {total_mean:>10.1f} | ")
    print()
    print("Note: This is component-level timing. Full pipeline may differ.")
    print("      The actual TTS pipeline orchestrates these models together.")

    # Save results
    output_dir = PROJECT_ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"onnx_benchmark_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "onnx_runtime",
            "provider": provider,
        },
        "onnxruntime_info": ort_info,
        "results": results,
        "summary": {
            "total_mean_ms": round(total_mean, 1),
            "models_benchmarked": len(results),
        }
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime Benchmark for Chatterbox TTS"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["cuda", "tensorrt", "cpu"],
        default="cuda",
        help="Execution provider (default: cuda)"
    )

    args = parser.parse_args()

    try:
        run_onnx_benchmark(provider=args.provider)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
