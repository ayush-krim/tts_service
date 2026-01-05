# API Routers
from server.routers.synthesize import router as synthesize_router
from server.routers.speakers import router as speakers_router
from server.routers.health import router as health_router

__all__ = ["synthesize_router", "speakers_router", "health_router"]
