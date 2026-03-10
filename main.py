import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from proxy.proxy_handler import router as proxy_router
from proxy.admin_routes import router as admin_router
from ml_engine.combined_detector import detector_instance

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML Models
    logger.info("Starting GenAI Privacy Firewall...")
    logger.info(f"Loading ML Model Type: {settings.model_type}")
    detector_instance.load_models()
    logger.info("Models loaded successfully.")
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down Firewall...")
    detector_instance.unload_models()

app = FastAPI(
    title="GenAI Privacy Firewall",
    description="Intercepts and redacts PII from LLM prompts.",
    lifespan=lifespan
)

# Register Routers
app.include_router(proxy_router, prefix="/v1", tags=["Proxy"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])

# Mount static files for dashboard
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector_instance.is_ready()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.proxy_port, reload=True)