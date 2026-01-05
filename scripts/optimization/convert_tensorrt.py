#!/usr/bin/env python3
"""
TensorRT Conversion Script for Chatterbox Turbo TTS

Converts ONNX models to optimized TensorRT engines for low-latency inference.
Supports FP16 precision and dynamic batch/sequence shapes.

Usage:
    python scripts/optimization/convert_tensorrt.py --all
    python scripts/optimization/convert_tensorrt.py --model t3
    python scripts/optimization/convert_tensorrt.py --model s3gen
    python scripts/optimization/convert_tensorrt.py --onnx-path models/onnx/model.onnx

Prerequisites:
    - TensorRT 10.x installed
    - NVIDIA GPU with compute capability >= 7.0
    - ONNX models exported (run export_onnx.py first)
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ShapeProfile:
    """TensorRT shape profile for dynamic inputs."""
    min_shape: str
    opt_shape: str
    max_shape: str


@dataclass
class ModelConfig:
    """Configuration for TensorRT model conversion."""
    name: str
    onnx_path: str
    engine_path: str
    fp16: bool = True
    workspace_mb: int = 8192
    shape_profiles: Dict[str, ShapeProfile] = None

    def get_trtexec_command(self) -> List[str]:
        """Generate trtexec command for this model."""
        cmd = [
            "trtexec",
            f"--onnx={self.onnx_path}",
            f"--saveEngine={self.engine_path}",
            f"--workspace={self.workspace_mb}",
        ]

        if self.fp16:
            cmd.append("--fp16")

        if self.shape_profiles:
            min_shapes = []
            opt_shapes = []
            max_shapes = []

            for input_name, profile in self.shape_profiles.items():
                min_shapes.append(f"{input_name}:{profile.min_shape}")
                opt_shapes.append(f"{input_name}:{profile.opt_shape}")
                max_shapes.append(f"{input_name}:{profile.max_shape}")

            cmd.append(f"--minShapes={','.join(min_shapes)}")
            cmd.append(f"--optShapes={','.join(opt_shapes)}")
            cmd.append(f"--maxShapes={','.join(max_shapes)}")

        return cmd


# Pre-defined configurations for Chatterbox models
MODEL_CONFIGS = {
    # T3 Language Model (GPT-2)
    "t3": ModelConfig(
        name="T3 Language Model",
        onnx_path="models/onnx/chatterbox-turbo-onnx/onnx/language_model_fp16.onnx",
        engine_path="engines/t3_fp16.plan",
        fp16=True,
        workspace_mb=8192,
        shape_profiles={
            "input_ids": ShapeProfile("1x1", "1x128", "4x512"),
            "attention_mask": ShapeProfile("1x1", "1x128", "4x512"),
        }
    ),

    # S3Gen Conditional Decoder (Token-to-Audio main component)
    "s3gen": ModelConfig(
        name="S3Gen Conditional Decoder",
        onnx_path="models/onnx/chatterbox-turbo-onnx/onnx/conditional_decoder_fp16.onnx",
        engine_path="engines/s3gen_fp16.plan",
        fp16=True,
        workspace_mb=8192,
        shape_profiles={
            "input": ShapeProfile("1x10x256", "1x200x256", "4x500x256"),
        }
    ),

    # Speech Encoder
    "speech_encoder": ModelConfig(
        name="Speech Encoder",
        onnx_path="models/onnx/chatterbox-turbo-onnx/onnx/speech_encoder_fp16.onnx",
        engine_path="engines/speech_encoder_fp16.plan",
        fp16=True,
        workspace_mb=4096,
        shape_profiles={
            "input": ShapeProfile("1x80x10", "1x80x200", "4x80x500"),
        }
    ),

    # Voice Encoder (CAMPPlus) - if exported locally
    "voice_encoder": ModelConfig(
        name="Voice Encoder",
        onnx_path="models/onnx/voice_encoder.onnx",
        engine_path="engines/voice_encoder_fp16.plan",
        fp16=True,
        workspace_mb=2048,
        shape_profiles={
            "audio": ShapeProfile("1x24000", "1x120000", "1x480000"),
        }
    ),
}


def check_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        result = subprocess.run(
            ["trtexec", "--help"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_nvidia_gpu() -> Optional[str]:
    """Check for NVIDIA GPU and return GPU name."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except FileNotFoundError:
        pass
    return None


def convert_model(config: ModelConfig, verbose: bool = True) -> bool:
    """
    Convert ONNX model to TensorRT engine.

    Args:
        config: Model configuration
        verbose: Print detailed output

    Returns:
        True if conversion successful
    """
    print(f"\n{'='*60}")
    print(f"Converting: {config.name}")
    print(f"{'='*60}")

    # Check if ONNX exists
    onnx_path = PROJECT_ROOT / config.onnx_path
    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found: {onnx_path}")
        print("\nTo get ONNX models, run one of:")
        print("  python scripts/optimization/export_onnx.py --use-pretrained-onnx")
        print("  python scripts/optimization/export_onnx.py --component all")
        return False

    # Create engine directory
    engine_path = PROJECT_ROOT / config.engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # Update paths in config for trtexec
    config.onnx_path = str(onnx_path)
    config.engine_path = str(engine_path)

    # Get trtexec command
    cmd = config.get_trtexec_command()

    if verbose:
        print(f"ONNX: {onnx_path}")
        print(f"Engine: {engine_path}")
        print(f"Precision: {'FP16' if config.fp16 else 'FP32'}")
        print(f"Workspace: {config.workspace_mb} MB")
        print(f"\nCommand: {' '.join(cmd)}")

    # Run conversion
    print("\nStarting TensorRT conversion (this may take several minutes)...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True
        )

        if result.returncode == 0:
            engine_size = engine_path.stat().st_size / 1e6
            print(f"\nSUCCESS: Engine created ({engine_size:.2f} MB)")
            return True
        else:
            print(f"\nERROR: Conversion failed")
            if not verbose and result.stderr:
                print(f"stderr: {result.stderr[:500]}")
            return False

    except Exception as e:
        print(f"\nERROR: {e}")
        return False


def benchmark_engine(engine_path: Path, iterations: int = 100) -> Optional[Dict]:
    """
    Benchmark TensorRT engine performance.

    Returns timing statistics.
    """
    print(f"\nBenchmarking: {engine_path.name}")

    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--iterations={iterations}",
        "--warmUp=500",
        "--duration=0",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Parse output for timing info
            output = result.stdout

            stats = {}
            for line in output.split('\n'):
                if 'mean' in line.lower() and 'ms' in line.lower():
                    # Parse timing lines
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'mean:':
                            try:
                                stats['mean_ms'] = float(parts[i+1])
                            except (IndexError, ValueError):
                                pass
                        elif part == 'median:':
                            try:
                                stats['median_ms'] = float(parts[i+1])
                            except (IndexError, ValueError):
                                pass

            if stats:
                print(f"  Mean latency: {stats.get('mean_ms', 'N/A')} ms")
                print(f"  Median latency: {stats.get('median_ms', 'N/A')} ms")

            return stats
        else:
            print(f"  Benchmark failed")
            return None

    except Exception as e:
        print(f"  Benchmark error: {e}")
        return None


def list_available_models():
    """List available model configurations."""
    print("\nAvailable Model Configurations:")
    print("-" * 60)

    for key, config in MODEL_CONFIGS.items():
        onnx_path = PROJECT_ROOT / config.onnx_path
        exists = "EXISTS" if onnx_path.exists() else "MISSING"
        print(f"  {key:20} - {config.name} [{exists}]")

    print("\nTo use custom ONNX path:")
    print("  python convert_tensorrt.py --onnx-path path/to/model.onnx --engine-path path/to/engine.plan")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to TensorRT engines"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Pre-configured model to convert"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all available models"
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        help="Custom ONNX model path"
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        help="Custom engine output path"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision instead of FP16"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=8192,
        help="TensorRT workspace size in MB (default: 8192)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark converted engines"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model configurations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed trtexec output"
    )

    args = parser.parse_args()

    # Header
    print("="*60)
    print("Chatterbox TTS - TensorRT Conversion")
    print("="*60)

    # Check prerequisites
    gpu = check_nvidia_gpu()
    if not gpu:
        print("ERROR: NVIDIA GPU not found")
        sys.exit(1)
    print(f"GPU: {gpu}")

    if not check_tensorrt_available():
        print("ERROR: TensorRT (trtexec) not found")
        print("Install TensorRT from: https://developer.nvidia.com/tensorrt")
        sys.exit(1)
    print("TensorRT: Available")

    # List models
    if args.list:
        list_available_models()
        return

    # Determine what to convert
    models_to_convert = []

    if args.onnx_path:
        # Custom model
        config = ModelConfig(
            name="Custom Model",
            onnx_path=args.onnx_path,
            engine_path=args.engine_path or args.onnx_path.replace('.onnx', '.plan'),
            fp16=not args.fp32,
            workspace_mb=args.workspace
        )
        models_to_convert.append(config)

    elif args.model:
        # Specific pre-configured model
        config = MODEL_CONFIGS[args.model]
        config.fp16 = not args.fp32
        config.workspace_mb = args.workspace
        models_to_convert.append(config)

    elif args.all:
        # All models
        for config in MODEL_CONFIGS.values():
            config = ModelConfig(
                name=config.name,
                onnx_path=config.onnx_path,
                engine_path=config.engine_path,
                fp16=not args.fp32,
                workspace_mb=args.workspace,
                shape_profiles=config.shape_profiles
            )
            models_to_convert.append(config)

    else:
        print("\nNo model specified. Use --model, --all, or --onnx-path")
        list_available_models()
        return

    # Convert models
    results = {}
    for config in models_to_convert:
        success = convert_model(config, verbose=args.verbose)
        results[config.name] = success

        if success and args.benchmark:
            engine_path = PROJECT_ROOT / config.engine_path
            benchmark_engine(engine_path)

    # Summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")

    print(f"\n{successful}/{total} conversions successful")

    if successful > 0:
        print("\nEngines saved to: engines/")
        print("\nNext steps:")
        print("  1. Build FastAPI server: python server/main.py")
        print("  2. Run inference with TensorRT engines")


if __name__ == "__main__":
    main()
