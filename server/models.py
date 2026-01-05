"""
Pydantic Models for API Request/Response

Defines the data models for the TTS API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class AudioFormat(str, Enum):
    """Supported audio output formats."""
    WAV = "wav"
    PCM = "pcm"
    MULAW = "mulaw"
    ALAW = "alaw"


class SampleRate(int, Enum):
    """Supported sample rates."""
    TELEPHONY = 8000
    WIDEBAND = 16000
    FULLBAND = 24000


# =============================================================================
# Synthesize Endpoints
# =============================================================================

class SynthesizeRequest(BaseModel):
    """Request body for /v1/synthesize endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text to synthesize"
    )
    speaker_id: str = Field(
        default="default",
        description="Speaker ID for voice cloning"
    )
    format: AudioFormat = Field(
        default=AudioFormat.WAV,
        description="Audio output format"
    )
    sample_rate: SampleRate = Field(
        default=SampleRate.FULLBAND,
        description="Output sample rate in Hz"
    )

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class SynthesizeStreamRequest(BaseModel):
    """Request body for /v1/synthesize/stream endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text to synthesize"
    )
    speaker_id: str = Field(
        default="default",
        description="Speaker ID for voice cloning"
    )
    chunk_size_ms: int = Field(
        default=100,
        ge=20,
        le=500,
        description="Chunk size in milliseconds"
    )
    format: AudioFormat = Field(
        default=AudioFormat.PCM,
        description="Audio output format for chunks"
    )
    sample_rate: SampleRate = Field(
        default=SampleRate.TELEPHONY,
        description="Output sample rate in Hz"
    )


class SynthesizeResponse(BaseModel):
    """Response metadata for synthesize endpoint (returned in headers)."""
    inference_time_ms: float
    audio_duration_ms: float
    rtf: float  # Real-time factor
    model_version: str


# =============================================================================
# Speaker Endpoints
# =============================================================================

class SpeakerCreate(BaseModel):
    """Response when creating a new speaker."""
    speaker_id: str
    name: Optional[str] = None
    created_at: datetime


class SpeakerInfo(BaseModel):
    """Speaker information."""
    id: str
    name: Optional[str] = None
    created_at: datetime
    embedding_dim: int = 256


class SpeakerListResponse(BaseModel):
    """Response for listing speakers."""
    speakers: List[SpeakerInfo]
    total: int


# =============================================================================
# Health & Status
# =============================================================================

class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: HealthStatus
    gpu_available: bool
    model_loaded: bool
    version: str
    uptime_seconds: float
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# =============================================================================
# Streaming Events
# =============================================================================

class StreamEvent(BaseModel):
    """Server-Sent Event for streaming audio."""
    event: str  # "audio", "metadata", "done", "error"
    data: str  # Base64 encoded audio or JSON


class StreamMetadata(BaseModel):
    """Metadata sent at start of stream."""
    total_chunks: Optional[int] = None
    sample_rate: int
    format: str
    speaker_id: str
