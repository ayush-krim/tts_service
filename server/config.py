"""
Server Configuration

Environment variables and settings for the Chatterbox TTS server.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # Single worker for GPU
    log_level: str = "info"
    debug: bool = False

    # Model paths
    model_path: str = "models/chatterbox-turbo"
    engine_path: str = "engines"
    onnx_path: str = "models/onnx/chatterbox-turbo-onnx/onnx"

    # Device
    device: str = "cuda"
    cuda_visible_devices: str = "0"

    # Chatterbox specific
    chatterbox_cfg_scale: float = 0.5  # Classifier-free guidance scale
    use_tensorrt: bool = True
    use_onnx: bool = False

    # Audio settings
    output_sample_rate: int = 24000  # Native Chatterbox output
    telephony_sample_rate: int = 8000  # Telephony output
    default_audio_format: str = "wav"

    # Streaming
    default_chunk_size_ms: int = 100
    max_text_length: int = 2000

    # Redis (speaker embedding cache)
    redis_url: str = "redis://localhost:6379"
    speaker_cache_ttl: int = 3600  # 1 hour
    enable_speaker_cache: bool = True

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Security
    api_key: Optional[str] = None
    cors_origins: str = "*"

    class Config:
        env_prefix = ""
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience accessor
settings = get_settings()
