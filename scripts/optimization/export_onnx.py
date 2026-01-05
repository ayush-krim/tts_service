#!/usr/bin/env python3
"""
ONNX Export Script for Chatterbox Turbo TTS Models

Exports T3 (text-to-token) and S3Gen (token-to-audio) models to ONNX format
with dynamic shapes for variable-length input support.

Usage:
    python scripts/optimization/export_onnx.py --component all
    python scripts/optimization/export_onnx.py --component t3
    python scripts/optimization/export_onnx.py --component s3gen
    python scripts/optimization/export_onnx.py --component voice_encoder
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device() -> str:
    """Get available device (CUDA preferred)."""
    if torch.cuda.is_available():
        return "cuda"
    print("WARNING: CUDA not available, using CPU. Export may be slower.")
    return "cpu"


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def export_voice_encoder(
    model,
    output_dir: Path,
    opset_version: int = 17,
    device: str = "cuda"
) -> Path:
    """
    Export VoiceEncoder (CAMPPlus) to ONNX.

    Input: Reference audio waveform
    Output: 256-dim speaker embedding
    """
    print("\n" + "="*60)
    print("Exporting VoiceEncoder to ONNX")
    print("="*60)

    output_path = output_dir / "voice_encoder.onnx"

    # Get voice encoder module
    voice_encoder = model.voice_encoder
    voice_encoder.eval()

    # Create dummy input (audio waveform)
    # Shape: [batch, samples] - variable length audio
    dummy_audio = torch.randn(1, 24000 * 5, device=device)  # 5 seconds @ 24kHz

    # Dynamic axes for variable-length audio
    dynamic_axes = {
        'audio': {0: 'batch', 1: 'samples'},
        'embedding': {0: 'batch'}
    }

    print(f"  Input shape: {dummy_audio.shape}")
    print(f"  Output dir: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            voice_encoder,
            dummy_audio,
            str(output_path),
            opset_version=opset_version,
            input_names=['audio'],
            output_names=['embedding'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True
        )

    # Verify export
    file_size = output_path.stat().st_size / 1e6
    print(f"  Exported: {output_path.name} ({file_size:.2f} MB)")

    return output_path


def export_t3_model(
    model,
    output_dir: Path,
    opset_version: int = 17,
    device: str = "cuda"
) -> Path:
    """
    Export T3 (Text-to-Token) GPT-2 model to ONNX.

    Input: Text token IDs + speaker embedding
    Output: Speech tokens @ 25Hz
    """
    print("\n" + "="*60)
    print("Exporting T3 (Text-to-Token) to ONNX")
    print("="*60)

    output_path = output_dir / "t3_model.onnx"

    # Get T3 module
    t3 = model.t3
    t3.eval()

    # Create dummy inputs
    batch_size = 1
    seq_length = 128  # Typical text length
    speaker_dim = 256

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    dummy_attention_mask = torch.ones(batch_size, seq_length, device=device)
    dummy_speaker_emb = torch.randn(batch_size, speaker_dim, device=device)

    # Dynamic axes for variable sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'speaker_embedding': {0: 'batch'},
        'speech_tokens': {0: 'batch', 1: 'token_sequence'}
    }

    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    print(f"  Speaker embedding shape: {dummy_speaker_emb.shape}")
    print(f"  Output dir: {output_path}")

    # Note: T3 export may require custom wrapper depending on model implementation
    try:
        with torch.no_grad():
            torch.onnx.export(
                t3,
                (dummy_input_ids, dummy_attention_mask, dummy_speaker_emb),
                str(output_path),
                opset_version=opset_version,
                input_names=['input_ids', 'attention_mask', 'speaker_embedding'],
                output_names=['speech_tokens'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True
            )

        file_size = output_path.stat().st_size / 1e6
        print(f"  Exported: {output_path.name} ({file_size:.2f} MB)")

    except Exception as e:
        print(f"  WARNING: Direct T3 export failed: {e}")
        print("  Recommendation: Use pre-exported ONNX from HuggingFace:")
        print("  ResembleAI/chatterbox-turbo-ONNX/onnx/language_model_fp16.onnx")
        return None

    return output_path


def export_s3gen_model(
    model,
    output_dir: Path,
    opset_version: int = 17,
    device: str = "cuda"
) -> Dict[str, Path]:
    """
    Export S3Gen (Token-to-Audio) to ONNX.

    S3Gen has complex architecture with multiple submodules:
    - S3Token2Mel: UpsampleConformerEncoder + CausalConditionalCFM
    - HiFTGenerator: Neural vocoder

    Due to complexity, we export submodules separately for better optimization.
    """
    print("\n" + "="*60)
    print("Exporting S3Gen (Token-to-Audio) to ONNX")
    print("="*60)

    exported = {}
    s3gen = model.s3gen
    s3gen.eval()

    # Try full S3Gen export first
    output_path = output_dir / "s3gen_full.onnx"

    # Dummy inputs for S3Gen
    batch_size = 1
    token_seq_len = 200  # ~8 seconds of audio @ 25Hz
    speaker_dim = 256

    dummy_speech_tokens = torch.randint(0, 1024, (batch_size, token_seq_len), device=device)
    dummy_speaker_emb = torch.randn(batch_size, speaker_dim, device=device)

    dynamic_axes = {
        'speech_tokens': {0: 'batch', 1: 'token_sequence'},
        'speaker_embedding': {0: 'batch'},
        'audio': {0: 'batch', 1: 'samples'}
    }

    print(f"  Speech tokens shape: {dummy_speech_tokens.shape}")
    print(f"  Speaker embedding shape: {dummy_speaker_emb.shape}")

    try:
        with torch.no_grad():
            torch.onnx.export(
                s3gen,
                (dummy_speech_tokens, dummy_speaker_emb),
                str(output_path),
                opset_version=opset_version,
                input_names=['speech_tokens', 'speaker_embedding'],
                output_names=['audio'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True
            )

        file_size = output_path.stat().st_size / 1e6
        print(f"  Exported: {output_path.name} ({file_size:.2f} MB)")
        exported['full'] = output_path

    except Exception as e:
        print(f"  WARNING: Full S3Gen export failed: {e}")
        print("  Attempting submodule export...")

        # Export submodules separately
        exported.update(export_s3gen_submodules(s3gen, output_dir, opset_version, device))

    if not exported:
        print("  Recommendation: Use pre-exported ONNX from HuggingFace:")
        print("  ResembleAI/chatterbox-turbo-ONNX/onnx/conditional_decoder_fp16.onnx")
        print("  ResembleAI/chatterbox-turbo-ONNX/onnx/speech_encoder_fp16.onnx")

    return exported


def export_s3gen_submodules(
    s3gen,
    output_dir: Path,
    opset_version: int = 17,
    device: str = "cuda"
) -> Dict[str, Path]:
    """
    Export S3Gen submodules separately when full export fails.

    Components:
    - s3token2mel: Converts speech tokens to mel spectrogram
    - hift: HiFi-GAN vocoder, converts mel to waveform
    """
    exported = {}

    # Try to export S3Token2Mel
    if hasattr(s3gen, 's3token2mel'):
        try:
            s3token2mel = s3gen.s3token2mel
            s3token2mel.eval()

            output_path = output_dir / "s3token2mel.onnx"

            batch_size = 1
            token_seq_len = 200
            speaker_dim = 256

            dummy_tokens = torch.randint(0, 1024, (batch_size, token_seq_len), device=device)
            dummy_speaker = torch.randn(batch_size, speaker_dim, device=device)

            with torch.no_grad():
                torch.onnx.export(
                    s3token2mel,
                    (dummy_tokens, dummy_speaker),
                    str(output_path),
                    opset_version=opset_version,
                    input_names=['speech_tokens', 'speaker_embedding'],
                    output_names=['mel_spectrogram'],
                    dynamic_axes={
                        'speech_tokens': {0: 'batch', 1: 'tokens'},
                        'speaker_embedding': {0: 'batch'},
                        'mel_spectrogram': {0: 'batch', 1: 'mel_dim', 2: 'frames'}
                    },
                    do_constant_folding=True
                )

            file_size = output_path.stat().st_size / 1e6
            print(f"  Exported: {output_path.name} ({file_size:.2f} MB)")
            exported['s3token2mel'] = output_path

        except Exception as e:
            print(f"  WARNING: S3Token2Mel export failed: {e}")

    # Try to export HiFT Generator (vocoder)
    if hasattr(s3gen, 'hift') or hasattr(s3gen, 'vocoder'):
        vocoder = getattr(s3gen, 'hift', None) or getattr(s3gen, 'vocoder', None)

        if vocoder is not None:
            try:
                vocoder.eval()

                output_path = output_dir / "hift_vocoder.onnx"

                # Mel spectrogram input
                batch_size = 1
                mel_dim = 80  # Typical mel bins
                mel_frames = 800  # ~8 seconds

                dummy_mel = torch.randn(batch_size, mel_dim, mel_frames, device=device)

                with torch.no_grad():
                    torch.onnx.export(
                        vocoder,
                        dummy_mel,
                        str(output_path),
                        opset_version=opset_version,
                        input_names=['mel_spectrogram'],
                        output_names=['audio'],
                        dynamic_axes={
                            'mel_spectrogram': {0: 'batch', 2: 'frames'},
                            'audio': {0: 'batch', 1: 'samples'}
                        },
                        do_constant_folding=True
                    )

                file_size = output_path.stat().st_size / 1e6
                print(f"  Exported: {output_path.name} ({file_size:.2f} MB)")
                exported['vocoder'] = output_path

            except Exception as e:
                print(f"  WARNING: Vocoder export failed: {e}")

    return exported


def verify_onnx_export(onnx_path: Path) -> bool:
    """Verify ONNX model is valid."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"  Verified: {onnx_path.name}")
        return True
    except Exception as e:
        print(f"  Verification failed for {onnx_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export Chatterbox Turbo models to ONNX format"
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["all", "t3", "s3gen", "voice_encoder"],
        default="all",
        help="Which component to export (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Output directory for ONNX files (default: models/onnx)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported ONNX models"
    )
    parser.add_argument(
        "--use-pretrained-onnx",
        action="store_true",
        help="Download and use pre-exported ONNX from HuggingFace instead"
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    ensure_dir(output_dir)

    print("="*60)
    print("Chatterbox Turbo ONNX Export")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"ONNX opset version: {args.opset_version}")

    if args.use_pretrained_onnx:
        print("\nDownloading pre-exported ONNX models from HuggingFace...")
        print("Repository: ResembleAI/chatterbox-turbo-ONNX")

        from huggingface_hub import snapshot_download

        local_dir = output_dir / "chatterbox-turbo-onnx"
        snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo-ONNX",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )

        print(f"\nDownloaded to: {local_dir}")
        print("\nAvailable models:")
        for f in (local_dir / "onnx").glob("*.onnx"):
            size = f.stat().st_size / 1e6
            print(f"  - {f.name} ({size:.2f} MB)")

        return

    # Load Chatterbox model
    device = get_device()
    print(f"\nLoading Chatterbox Turbo model on {device}...")

    try:
        from chatterbox import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.eval()
        print("Model loaded successfully")
    except ImportError:
        print("ERROR: chatterbox-tts package not installed")
        print("Install with: pip install chatterbox-tts")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("\nTry using pre-exported ONNX models instead:")
        print("  python export_onnx.py --use-pretrained-onnx")
        sys.exit(1)

    exported_paths = []

    # Export components
    if args.component in ["all", "voice_encoder"]:
        path = export_voice_encoder(model, output_dir, args.opset_version, device)
        if path:
            exported_paths.append(path)

    if args.component in ["all", "t3"]:
        path = export_t3_model(model, output_dir, args.opset_version, device)
        if path:
            exported_paths.append(path)

    if args.component in ["all", "s3gen"]:
        paths = export_s3gen_model(model, output_dir, args.opset_version, device)
        exported_paths.extend(paths.values())

    # Verify exports
    if args.verify and exported_paths:
        print("\n" + "="*60)
        print("Verifying ONNX Exports")
        print("="*60)

        for path in exported_paths:
            if path and path.exists():
                verify_onnx_export(path)

    # Summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)

    if exported_paths:
        print(f"Successfully exported {len(exported_paths)} model(s):")
        for path in exported_paths:
            if path:
                print(f"  - {path}")
    else:
        print("No models were exported.")
        print("\nRecommendation: Use pre-exported ONNX models from HuggingFace:")
        print("  python export_onnx.py --use-pretrained-onnx")

    print("\nNext step: Convert to TensorRT")
    print("  python scripts/optimization/convert_tensorrt.py")


if __name__ == "__main__":
    main()
