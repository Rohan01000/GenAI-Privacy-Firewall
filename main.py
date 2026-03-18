import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from proxy.proxy_handler import router as proxy_router
from proxy.proxy_handler import detector
from proxy.admin_routes import router as admin_router
from proxy.middleware import register_middleware

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML Models
    logger.info("Starting GenAI Privacy Firewall...")
    logger.info(f"Active ML Model Type: {settings.model_type}")
    
    # Note: Both ScratchNERInference and BertNERInference automatically 
    # load their weights and run a warmup pass during their __init__, 
    # which triggered when the proxy_handler singleton was imported.
    if hasattr(detector, 'ml_model') and hasattr(detector.ml_model, 'is_ready'):
        if detector.ml_model.is_ready():
            logger.info("ML Models warmed up and ready.")
        else:
            logger.warning("ML Models failed to initialize properly.")

    logger.info("Firewall is ready to intercept traffic.")
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down Firewall...")

app = FastAPI(
    title="GenAI Privacy Firewall",
    description="Intercepts and redacts PII from LLM prompts.",
    lifespan=lifespan
)

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from proxy.proxy_handler import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Register Middleware (Audit Logging, Timing, Request ID)
register_middleware(app)

# Register Routers
app.include_router(proxy_router, prefix="/v1", tags=["Proxy"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])

# Mount static files for the Admin Dashboard
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

@app.get("/health")
async def health_check():
    """Root level health check (duplicate of /admin/health but accessible without auth)"""
    model_ready = False
    if hasattr(detector, 'ml_model') and hasattr(detector.ml_model, 'is_ready'):
        model_ready = detector.ml_model.is_ready()
        
    return {
        "status": "healthy", 
        "model_ready": model_ready,
        "active_model": settings.model_type
    }

if __name__ == "__main__":
    import uvicorn
    # Use the port defined in .env / settings.py
    uvicorn.run("main:app", host="0.0.0.0", port=settings.proxy_port, reload=True)