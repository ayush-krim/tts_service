"""
Prometheus Metrics Middleware

Collects request metrics for monitoring.
"""

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Histogram, Gauge

    # Request metrics
    REQUEST_COUNT = Counter(
        'tts_requests_total',
        'Total TTS requests',
        ['method', 'endpoint', 'status']
    )

    REQUEST_LATENCY = Histogram(
        'tts_request_latency_ms',
        'Request latency in milliseconds',
        ['method', 'endpoint'],
        buckets=[10, 25, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000]
    )

    # Inference metrics
    INFERENCE_LATENCY = Histogram(
        'tts_inference_latency_ms',
        'Model inference latency in milliseconds',
        buckets=[50, 100, 150, 200, 300, 400, 500, 750, 1000]
    )

    RTF_HISTOGRAM = Histogram(
        'tts_rtf',
        'Real-time factor',
        buckets=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    )

    # Active requests gauge
    ACTIVE_REQUESTS = Gauge(
        'tts_active_requests',
        'Number of active requests'
    )

    # Audio duration histogram
    AUDIO_DURATION = Histogram(
        'tts_audio_duration_ms',
        'Generated audio duration in milliseconds',
        buckets=[500, 1000, 2000, 3000, 5000, 10000, 20000]
    )

    PROMETHEUS_AVAILABLE = True

except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics.

    Tracks:
    - Request count by endpoint and status
    - Request latency
    - Active requests
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)

        # Track active requests
        ACTIVE_REQUESTS.inc()

        start_time = time.perf_counter()
        method = request.method
        endpoint = self._get_endpoint(request)

        try:
            response = await call_next(request)

            # Record metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            status = str(response.status_code)

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(elapsed_ms)

            # Extract inference metrics from response headers
            self._record_inference_metrics(response)

            return response

        except Exception as e:
            # Record error
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status="500"
            ).inc()
            raise

        finally:
            ACTIVE_REQUESTS.dec()

    def _get_endpoint(self, request: Request) -> str:
        """Get normalized endpoint path."""
        path = request.url.path

        # Normalize paths with IDs
        # e.g., /v1/speakers/abc123 -> /v1/speakers/{id}
        parts = path.split('/')
        normalized = []

        for i, part in enumerate(parts):
            if i > 0 and parts[i-1] == 'speakers' and part:
                normalized.append('{id}')
            else:
                normalized.append(part)

        return '/'.join(normalized)

    def _record_inference_metrics(self, response: Response):
        """Record inference metrics from response headers."""
        # Inference time
        inference_time = response.headers.get('X-Inference-Time-Ms')
        if inference_time:
            try:
                INFERENCE_LATENCY.observe(float(inference_time))
            except (ValueError, TypeError):
                pass

        # RTF
        rtf = response.headers.get('X-RTF')
        if rtf:
            try:
                RTF_HISTOGRAM.observe(float(rtf))
            except (ValueError, TypeError):
                pass

        # Audio duration
        audio_duration = response.headers.get('X-Audio-Duration-Ms')
        if audio_duration:
            try:
                AUDIO_DURATION.observe(float(audio_duration))
            except (ValueError, TypeError):
                pass


def record_inference(inference_time_ms: float, rtf: float, audio_duration_ms: float):
    """
    Manually record inference metrics.

    Can be called from TTS service for more accurate timing.
    """
    if PROMETHEUS_AVAILABLE:
        INFERENCE_LATENCY.observe(inference_time_ms)
        RTF_HISTOGRAM.observe(rtf)
        AUDIO_DURATION.observe(audio_duration_ms)
