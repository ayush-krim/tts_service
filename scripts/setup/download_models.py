#!/usr/bin/env python3
"""
download_models.py - Download Chatterbox models from HuggingFace

Compatible with both old and new versions of huggingface_hub.
"""
import os
import shutil
import sys
from pathlib import Path


def download_repo(repo_id: str, target_dir: Path) -> None:
    """
    Download a HuggingFace repo, compatible with old and new huggingface_hub versions.
    """
    from huggingface_hub import snapshot_download
    import inspect

    # Check which parameters are supported
    sig = inspect.signature(snapshot_download)
    params = sig.parameters

    if 'local_dir' in params:
        # New API (huggingface_hub >= 0.17)
        snapshot_download(
            repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False
        )
    else:
        # Old API - downloads to cache, then we copy
        cache_path = snapshot_download(repo_id)

        # Copy from cache to target
        target_dir.mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_path)

        for item in cache_path.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)


def main():
    print("=" * 60)
    print("DOWNLOADING CHATTERBOX MODELS FROM HUGGINGFACE")
    print("=" * 60)

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models directory: {models_dir}")
    print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install --upgrade huggingface_hub")
        from huggingface_hub import snapshot_download

    # Download PyTorch models
    print("1. Downloading PyTorch models (ResembleAI/chatterbox-turbo)...")
    print("   This is ~4GB, may take a few minutes...")
    print("-" * 60)

    pytorch_dir = models_dir / "chatterbox-turbo"
    if pytorch_dir.exists() and any(pytorch_dir.iterdir()):
        print("   PyTorch models already downloaded, skipping...")
    else:
        download_repo("ResembleAI/chatterbox-turbo", pytorch_dir)
        print("   PyTorch models downloaded!")

    print()

    # Download ONNX models
    print("2. Downloading ONNX models (ResembleAI/chatterbox-turbo-ONNX)...")
    print("   This is ~7.4GB, may take several minutes...")
    print("-" * 60)

    onnx_dir = models_dir / "chatterbox-turbo-onnx"
    if onnx_dir.exists() and any(onnx_dir.iterdir()):
        print("   ONNX models already downloaded, skipping...")
    else:
        download_repo("ResembleAI/chatterbox-turbo-ONNX", onnx_dir)
        print("   ONNX models downloaded!")

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    # List downloaded files
    print()
    print("PyTorch models:")
    if pytorch_dir.exists():
        for f in sorted(pytorch_dir.iterdir()):
            size = f.stat().st_size / 1e6 if f.is_file() else 0
            print(f"  {f.name}: {size:.1f} MB" if size > 0 else f"  {f.name}/")

    print()
    print("ONNX models:")
    if onnx_dir.exists():
        onnx_subdir = onnx_dir / "onnx"
        if onnx_subdir.exists():
            for f in sorted(onnx_subdir.iterdir())[:10]:  # First 10
                size = f.stat().st_size / 1e6 if f.is_file() else 0
                print(f"  {f.name}: {size:.1f} MB" if size > 0 else f"  {f.name}/")
            print("  ... (more files)")

if __name__ == "__main__":
    main()
