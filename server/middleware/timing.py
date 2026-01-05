"""
Timing Middleware

Adds request timing to all responses.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to measure and add request timing headers.

    Adds:
    - X-Request-Time-Ms: Total request processing time
    - X-Request-Start: Request start timestamp
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Add timing headers
        response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Request-Start"] = f"{start_time:.6f}"

        return response
