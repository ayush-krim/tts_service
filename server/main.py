#!/usr/bin/env python3
"""
Chatterbox TTS Server

Low-latency TTS API for Krim AI telephony infrastructure.

Usage:
    python server/main.py
    uvicorn server.main:app --host 0.0.0.0 --port 8080
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.config import settings
from server.routers import synthesize_router, speakers_router, health_router
from server.services.tts import TTSService
from server.services.speakers import SpeakerService
from server.middleware.timing import TimingMiddleware
from server.middleware.metrics import PrometheusMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    print("="*60)
    print("Chatterbox TTS Server Starting")
    print("="*60)

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
    os.environ["CHATTERBOX_CFG_SCALE"] = str(settings.chatterbox_cfg_scale)

    print(f"Device: {settings.device}")
    print(f"CFG Scale: {settings.chatterbox_cfg_scale}")
    print(f"TensorRT: {settings.use_tensorrt}")
    print(f"ONNX: {settings.use_onnx}")

    # Initialize TTS service
    print("\nInitializing TTS service...")
    tts_service = TTSService(
        model_path=settings.model_path,
        engine_path=settings.engine_path,
        device=settings.device,
        use_tensorrt=settings.use_tensorrt,
        use_onnx=settings.use_onnx
    )

    await tts_service.load()
    app.state.tts_service = tts_service

    # Initialize speaker service
    print("Initializing speaker service...")
    redis_client = None
    if settings.enable_speaker_cache:
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(settings.redis_url)
            await redis_client.ping()
            print(f"  Redis connected: {settings.redis_url}")
        except Exception as e:
            print(f"  Redis not available: {e}")
            redis_client = None

    speaker_service = SpeakerService(
        voice_encoder=tts_service.voice_encoder,
        redis_client=redis_client,
        cache_ttl=settings.speaker_cache_ttl,
        device=settings.device
    )
    speaker_service.set_default_embedding(tts_service.default_embedding)
    app.state.speaker_service = speaker_service

    # Preload speakers if directory exists
    speakers_dir = PROJECT_ROOT / "speakers"
    if speakers_dir.exists():
        print(f"Preloading speakers from {speakers_dir}...")
        await speaker_service.preload_speakers(speakers_dir)

    print("\n" + "="*60)
    print(f"Server ready at http://{settings.host}:{settings.port}")
    print("="*60)
    print("\nEndpoints:")
    print(f"  POST /v1/synthesize       - Generate speech")
    print(f"  POST /v1/synthesize/stream - Stream speech (SSE)")
    print(f"  POST /v1/speakers         - Create speaker")
    print(f"  GET  /v1/speakers         - List speakers")
    print(f"  GET  /health              - Health check")
    print(f"  GET  /metrics             - Prometheus metrics")
    print(f"  GET  /docs                - API documentation")
    print("")

    yield

    # Shutdown
    print("\nShutting down...")

    if redis_client:
        await redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="Chatterbox TTS API",
    description="""
Low-latency Text-to-Speech API powered by Chatterbox Turbo.

## Features
- Sub-200ms latency on NVIDIA A10G GPU
- Voice cloning with speaker embeddings
- Streaming audio output (SSE)
- Telephony-ready formats (8kHz G.711 mu-law)
- Real-time performance metrics

## Authentication
API key authentication can be enabled via the `API_KEY` environment variable.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware (order matters - first added is outermost)
app.add_middleware(TimingMiddleware)

if settings.enable_metrics:
    app.add_middleware(PrometheusMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(synthesize_router, prefix="/v1", tags=["TTS"])
app.include_router(speakers_router, prefix="/v1", tags=["Speakers"])
app.include_router(health_router, tags=["Health"])


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "service": "Chatterbox TTS API",
        "version": "1.0.0",
        "docs": "/docs"
    }


def main():
    """Run the server."""
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
