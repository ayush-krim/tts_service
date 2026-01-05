"""
Health Router

Health check and metrics endpoints.
"""

import time
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from server.models import HealthResponse, HealthStatus

router = APIRouter()

# Track server start time
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health status",
)
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns:
    - status: "healthy", "unhealthy", or "degraded"
    - gpu_available: Whether CUDA GPU is available
    - model_loaded: Whether TTS model is loaded
    - version: Service version
    - uptime_seconds: Time since server started
    """
    import torch

    # Check GPU
    gpu_available = torch.cuda.is_available()

    # Check model
    model_loaded = False
    tts_service = getattr(request.app.state, 'tts_service', None)
    if tts_service:
        model_loaded = tts_service.is_loaded()

    # Determine status
    if model_loaded and gpu_available:
        status = HealthStatus.HEALTHY
    elif model_loaded:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    # Get version
    from server import __version__

    # Calculate uptime
    uptime = time.time() - _start_time

    # Additional details
    details = {}
    if gpu_available:
        details["gpu_name"] = torch.cuda.get_device_name(0)
        details["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / 1e6

    if tts_service:
        memory = tts_service.get_memory_usage()
        if memory:
            details["vram_used_mb"] = memory.get("allocated_mb", 0)

    return HealthResponse(
        status=status,
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        version=__version__,
        uptime_seconds=uptime,
        details=details if details else None
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe - always returns 200 if server is running",
)
async def liveness():
    """
    Liveness probe for Kubernetes.

    Always returns 200 if the server process is running.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe - returns 200 only if model is loaded",
)
async def readiness(request: Request):
    """
    Readiness probe for Kubernetes.

    Returns 200 only if the model is loaded and ready to serve requests.
    """
    tts_service = getattr(request.app.state, 'tts_service', None)

    if tts_service and tts_service.is_loaded():
        return {"status": "ready"}
    else:
        return PlainTextResponse(
            content="Service not ready",
            status_code=503
        )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-format metrics endpoint",
)
async def metrics(request: Request):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        # Generate basic metrics manually
        metrics_text = _generate_basic_metrics(request)
        return PlainTextResponse(
            content=metrics_text,
            media_type="text/plain"
        )


def _generate_basic_metrics(request: Request) -> str:
    """Generate basic metrics when prometheus_client is not available."""
    import torch

    lines = []

    # Uptime
    uptime = time.time() - _start_time
    lines.append(f"# HELP tts_uptime_seconds Server uptime in seconds")
    lines.append(f"# TYPE tts_uptime_seconds gauge")
    lines.append(f"tts_uptime_seconds {uptime:.2f}")

    # GPU metrics
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9

        lines.append(f"# HELP tts_gpu_memory_allocated_gb GPU memory allocated in GB")
        lines.append(f"# TYPE tts_gpu_memory_allocated_gb gauge")
        lines.append(f"tts_gpu_memory_allocated_gb {memory_allocated:.3f}")

        lines.append(f"# HELP tts_gpu_memory_reserved_gb GPU memory reserved in GB")
        lines.append(f"# TYPE tts_gpu_memory_reserved_gb gauge")
        lines.append(f"tts_gpu_memory_reserved_gb {memory_reserved:.3f}")

    # Model status
    tts_service = getattr(request.app.state, 'tts_service', None)
    model_loaded = 1 if tts_service and tts_service.is_loaded() else 0

    lines.append(f"# HELP tts_model_loaded Whether the TTS model is loaded")
    lines.append(f"# TYPE tts_model_loaded gauge")
    lines.append(f"tts_model_loaded {model_loaded}")

    return "\n".join(lines) + "\n"
