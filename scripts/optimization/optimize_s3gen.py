#!/usr/bin/env python3
"""
S3Gen Profiling and Optimization Script

S3Gen (Token-to-Audio) accounts for ~70% of Chatterbox latency.
This script profiles S3Gen components and identifies optimization opportunities.

Components:
    - S3Token2Mel: UpsampleConformerEncoder + CausalConditionalCFM
    - HiFTGenerator: Neural vocoder (upsampling 8x5x3)

Usage:
    python scripts/optimization/optimize_s3gen.py --profile
    python scripts/optimization/optimize_s3gen.py --analyze
    python scripts/optimization/optimize_s3gen.py --compare-trt
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ProfilingResult:
    """Profiling result for a component."""
    component: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    iterations: int
    input_shape: str
    memory_mb: float = 0.0


@dataclass
class S3GenProfile:
    """Complete S3Gen profiling data."""
    total: ProfilingResult
    s3token2mel: Optional[ProfilingResult] = None
    vocoder: Optional[ProfilingResult] = None
    breakdown_pct: Dict[str, float] = None


def get_device():
    """Get available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def profile_pytorch_model(
    model,
    inputs: Tuple,
    num_warmup: int = 10,
    num_iterations: int = 50,
    component_name: str = "Model",
    device: str = "cuda"
) -> ProfilingResult:
    """
    Profile PyTorch model inference timing.

    Returns detailed timing statistics.
    """
    import torch

    model.eval()
    latencies = []

    # Warmup
    print(f"  Warming up {component_name}...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(*inputs)
            if device == "cuda":
                torch.cuda.synchronize()

    # Measure
    print(f"  Profiling {component_name} ({num_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(*inputs)

            if device == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000)

    # Memory usage
    memory_mb = 0.0
    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1e6

    latencies = np.array(latencies)

    return ProfilingResult(
        component=component_name,
        mean_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        iterations=num_iterations,
        input_shape=str([tuple(t.shape) for t in inputs]),
        memory_mb=memory_mb
    )


def profile_with_torch_profiler(
    model,
    inputs: Tuple,
    device: str = "cuda",
    output_path: Optional[Path] = None
) -> Dict:
    """
    Detailed profiling using torch.profiler.

    Generates Chrome trace for visualization.
    """
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    model.eval()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("  Running torch.profiler analysis...")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("s3gen_inference"):
            with torch.no_grad():
                _ = model(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()

    # Get summary
    summary = prof.key_averages().table(
        sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
        row_limit=20
    )

    # Export trace
    if output_path:
        trace_path = output_path / "s3gen_trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"  Chrome trace saved: {trace_path}")

    return {
        "summary": summary,
        "profiler": prof
    }


def profile_s3gen(
    model,
    token_lengths: List[int] = [50, 100, 200, 400],
    num_iterations: int = 50,
    device: str = "cuda"
) -> Dict[int, S3GenProfile]:
    """
    Profile S3Gen at various sequence lengths.

    Returns profiling results for each length.
    """
    import torch

    results = {}

    for length in token_lengths:
        print(f"\n{'='*60}")
        print(f"Profiling S3Gen with {length} tokens (~{length * 40}ms audio)")
        print(f"{'='*60}")

        # Create dummy inputs
        batch_size = 1
        speaker_dim = 256

        dummy_tokens = torch.randint(0, 1024, (batch_size, length), device=device)
        dummy_speaker = torch.randn(batch_size, speaker_dim, device=device)

        # Reset memory tracking
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Profile full S3Gen
        s3gen = model.s3gen
        total_result = profile_pytorch_model(
            s3gen,
            (dummy_tokens, dummy_speaker),
            num_iterations=num_iterations,
            component_name="S3Gen (Full)",
            device=device
        )

        profile = S3GenProfile(total=total_result)

        # Try to profile subcomponents
        if hasattr(s3gen, 's3token2mel'):
            try:
                s3token2mel_result = profile_pytorch_model(
                    s3gen.s3token2mel,
                    (dummy_tokens, dummy_speaker),
                    num_iterations=num_iterations,
                    component_name="S3Token2Mel",
                    device=device
                )
                profile.s3token2mel = s3token2mel_result
            except Exception as e:
                print(f"  Could not profile S3Token2Mel separately: {e}")

        if hasattr(s3gen, 'hift') or hasattr(s3gen, 'vocoder'):
            vocoder = getattr(s3gen, 'hift', None) or getattr(s3gen, 'vocoder', None)
            if vocoder:
                try:
                    # Need mel spectrogram input for vocoder
                    mel_frames = length * 4  # Approximate upsampling
                    mel_dim = 80
                    dummy_mel = torch.randn(batch_size, mel_dim, mel_frames, device=device)

                    vocoder_result = profile_pytorch_model(
                        vocoder,
                        (dummy_mel,),
                        num_iterations=num_iterations,
                        component_name="HiFT Vocoder",
                        device=device
                    )
                    profile.vocoder = vocoder_result
                except Exception as e:
                    print(f"  Could not profile vocoder separately: {e}")

        # Calculate breakdown
        if profile.s3token2mel and profile.vocoder:
            total = profile.s3token2mel.mean_ms + profile.vocoder.mean_ms
            profile.breakdown_pct = {
                "s3token2mel": (profile.s3token2mel.mean_ms / total) * 100,
                "vocoder": (profile.vocoder.mean_ms / total) * 100
            }

        results[length] = profile

        # Print results
        print(f"\n  Results for {length} tokens:")
        print(f"    Total:  {total_result.mean_ms:.2f} +/- {total_result.std_ms:.2f} ms")
        print(f"    P95:    {total_result.p95_ms:.2f} ms")
        print(f"    Memory: {total_result.memory_mb:.1f} MB")

        if profile.breakdown_pct:
            print(f"    Breakdown:")
            print(f"      S3Token2Mel: {profile.breakdown_pct['s3token2mel']:.1f}%")
            print(f"      Vocoder:     {profile.breakdown_pct['vocoder']:.1f}%")

    return results


def analyze_bottlenecks(profiles: Dict[int, S3GenProfile]) -> Dict:
    """
    Analyze profiling results to identify bottlenecks.

    Returns analysis with recommendations.
    """
    print("\n" + "="*60)
    print("Bottleneck Analysis")
    print("="*60)

    analysis = {
        "scaling": {},
        "bottlenecks": [],
        "recommendations": []
    }

    # Analyze scaling with sequence length
    lengths = sorted(profiles.keys())
    latencies = [profiles[l].total.mean_ms for l in lengths]

    if len(lengths) > 1:
        # Calculate scaling factor
        x = np.array(lengths)
        y = np.array(latencies)

        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        analysis["scaling"] = {
            "ms_per_token": slope,
            "base_latency_ms": coeffs[1],
            "lengths": lengths,
            "latencies": latencies
        }

        print(f"\nScaling Analysis:")
        print(f"  Base latency:    {coeffs[1]:.2f} ms")
        print(f"  Per-token cost:  {slope:.3f} ms/token")
        print(f"  RTF @ 100 tokens: {latencies[lengths.index(100)] / (100 * 40):.3f}" if 100 in lengths else "")

    # Identify bottlenecks
    for length, profile in profiles.items():
        if profile.breakdown_pct:
            if profile.breakdown_pct.get('vocoder', 0) > 60:
                analysis["bottlenecks"].append({
                    "component": "vocoder",
                    "impact": "HIGH",
                    "details": f"Vocoder is {profile.breakdown_pct['vocoder']:.1f}% at {length} tokens"
                })
            if profile.breakdown_pct.get('s3token2mel', 0) > 60:
                analysis["bottlenecks"].append({
                    "component": "s3token2mel",
                    "impact": "HIGH",
                    "details": f"S3Token2Mel is {profile.breakdown_pct['s3token2mel']:.1f}% at {length} tokens"
                })

    # Recommendations
    print("\nRecommendations:")

    if any(b["component"] == "vocoder" for b in analysis["bottlenecks"]):
        rec = "Prioritize vocoder (HiFT) TensorRT conversion"
        analysis["recommendations"].append(rec)
        print(f"  1. {rec}")

    if any(b["component"] == "s3token2mel" for b in analysis["bottlenecks"]):
        rec = "Optimize S3Token2Mel flow matching with FP16"
        analysis["recommendations"].append(rec)
        print(f"  2. {rec}")

    # Check memory efficiency
    max_memory = max(p.total.memory_mb for p in profiles.values())
    if max_memory > 8000:  # 8GB
        rec = f"Memory usage high ({max_memory:.0f} MB) - consider gradient checkpointing"
        analysis["recommendations"].append(rec)
        print(f"  3. {rec}")
    else:
        rec = f"Memory usage OK ({max_memory:.0f} MB) - can increase batch size"
        analysis["recommendations"].append(rec)
        print(f"  3. {rec}")

    # TensorRT recommendation
    rec = "Convert to TensorRT with FP16 for 40-50% speedup"
    analysis["recommendations"].append(rec)
    print(f"  4. {rec}")

    return analysis


def compare_pytorch_tensorrt(
    pytorch_model,
    tensorrt_engine_path: Path,
    token_lengths: List[int] = [100, 200],
    device: str = "cuda"
) -> Dict:
    """
    Compare PyTorch vs TensorRT inference performance.
    """
    import torch

    print("\n" + "="*60)
    print("PyTorch vs TensorRT Comparison")
    print("="*60)

    results = {"pytorch": {}, "tensorrt": {}}

    # PyTorch profiling
    print("\nPyTorch inference:")
    for length in token_lengths:
        dummy_tokens = torch.randint(0, 1024, (1, length), device=device)
        dummy_speaker = torch.randn(1, 256, device=device)

        result = profile_pytorch_model(
            pytorch_model.s3gen,
            (dummy_tokens, dummy_speaker),
            num_iterations=30,
            component_name=f"PyTorch ({length} tokens)",
            device=device
        )
        results["pytorch"][length] = result.mean_ms
        print(f"  {length} tokens: {result.mean_ms:.2f} ms")

    # TensorRT profiling (if available)
    if tensorrt_engine_path.exists():
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            print("\nTensorRT inference:")
            # Load and run TensorRT engine
            # This is a simplified version - actual implementation depends on engine structure

            print("  TensorRT comparison not yet implemented")
            print("  Run trtexec for benchmarks:")
            print(f"    trtexec --loadEngine={tensorrt_engine_path} --iterations=100")

        except ImportError:
            print("\n  TensorRT Python bindings not available")
            print("  Install with: pip install tensorrt")
    else:
        print(f"\n  TensorRT engine not found: {tensorrt_engine_path}")
        print("  Build with: python scripts/optimization/convert_tensorrt.py --model s3gen")

    return results


def save_results(
    profiles: Dict[int, S3GenProfile],
    analysis: Dict,
    output_dir: Path
):
    """Save profiling results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    results = {
        "profiles": {
            length: {
                "total": asdict(profile.total),
                "s3token2mel": asdict(profile.s3token2mel) if profile.s3token2mel else None,
                "vocoder": asdict(profile.vocoder) if profile.vocoder else None,
                "breakdown_pct": profile.breakdown_pct
            }
            for length, profile in profiles.items()
        },
        "analysis": analysis
    }

    output_path = output_dir / "s3gen_profile.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile and optimize S3Gen (Token-to-Audio)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiling on S3Gen"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze profiling results and identify bottlenecks"
    )
    parser.add_argument(
        "--compare-trt",
        action="store_true",
        help="Compare PyTorch vs TensorRT performance"
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Generate detailed torch.profiler trace"
    )
    parser.add_argument(
        "--token-lengths",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400],
        help="Token sequence lengths to test"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for profiling"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--trt-engine",
        type=str,
        default="engines/s3gen_fp16.plan",
        help="TensorRT engine path for comparison"
    )

    args = parser.parse_args()

    # Default to profile + analyze if no action specified
    if not any([args.profile, args.analyze, args.compare_trt, args.torch_profile]):
        args.profile = True
        args.analyze = True

    print("="*60)
    print("S3Gen Profiling and Optimization")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. Results will not reflect GPU performance.")

    # Load model
    print("\nLoading Chatterbox Turbo model...")
    try:
        from chatterbox import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.eval()
        print("Model loaded successfully")
    except ImportError:
        print("ERROR: chatterbox-tts not installed")
        print("Install with: pip install chatterbox-tts")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / args.output_dir

    profiles = None
    analysis = None

    # Run profiling
    if args.profile:
        profiles = profile_s3gen(
            model,
            token_lengths=args.token_lengths,
            num_iterations=args.iterations,
            device=device
        )

    # Torch profiler trace
    if args.torch_profile:
        import torch
        dummy_tokens = torch.randint(0, 1024, (1, 100), device=device)
        dummy_speaker = torch.randn(1, 256, device=device)

        profile_with_torch_profiler(
            model.s3gen,
            (dummy_tokens, dummy_speaker),
            device=device,
            output_path=output_dir
        )

    # Analyze results
    if args.analyze and profiles:
        analysis = analyze_bottlenecks(profiles)

    # Compare with TensorRT
    if args.compare_trt:
        trt_path = PROJECT_ROOT / args.trt_engine
        compare_pytorch_tensorrt(
            model,
            trt_path,
            token_lengths=[100, 200],
            device=device
        )

    # Save results
    if profiles:
        save_results(profiles, analysis or {}, output_dir)

    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. Export S3Gen to ONNX:")
    print("   python scripts/optimization/export_onnx.py --component s3gen")
    print("\n2. Convert to TensorRT:")
    print("   python scripts/optimization/convert_tensorrt.py --model s3gen")
    print("\n3. Run comparison benchmark:")
    print("   python scripts/optimization/optimize_s3gen.py --compare-trt")


if __name__ == "__main__":
    main()
