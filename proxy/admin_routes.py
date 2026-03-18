import time
from datetime import datetime
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from proxy.middleware import audit_log_buffer
from config.settings import settings

# Import the singletons from proxy_handler to update them dynamically
from proxy.proxy_handler import detector, limiter

router = APIRouter()

# Track startup time for uptime calculation
START_TIME = time.time()

# --- DEPENDENCIES ---
def verify_admin_key(x_admin_key: str = Header(...)):
    if x_admin_key != settings.admin_secret_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return x_admin_key

# --- MODELS ---
class ConfigUpdateRequest(BaseModel):
    confidence_threshold: Optional[float] = None
    model_type: Optional[str] = None
    rate_limit_per_minute: Optional[int] = None

# --- ROUTES ---
@router.get("/health")
def get_health():
    uptime = time.time() - START_TIME
    
    model_ready = False
    if hasattr(detector, 'ml_model') and hasattr(detector.ml_model, 'is_ready'):
        model_ready = detector.ml_model.is_ready()
        
    return {
        "status": "healthy",
        "model_ready": model_ready,
        "uptime_seconds": round(uptime, 2),
        "version": "1.0.0"
    }

@router.get("/stats")
def get_stats(admin_key: str = Depends(verify_admin_key)):
    skip_paths = ["/admin/stats", "/admin/recent-requests", "/dashboard", "/health", "/favicon.ico"]
    
    total_reqs = 0
    total_entities = 0
    blocked = 0
    latencies = []
    breakdown = Counter()
    
    now = datetime.utcnow()
    hourly_counts = [0] * 24

    for log in audit_log_buffer:
        if log.get("path") in skip_paths:
            continue
            
        total_reqs += 1
        total_entities += log.get("entity_count", 0)
        
        if log.get("status_code", 200) >= 400:
            blocked += 1
            
        latencies.append(log.get("process_time_ms", 0.0))
        
        if "entity_count_by_type" in log:
            for k, v in log["entity_count_by_type"].items():
                breakdown[k] += v
        elif log.get("entity_count", 0) > 0:
            breakdown["SENSITIVE_DATA"] += log["entity_count"]

        try:
            log_time_str = log["timestamp"].replace("Z", "+00:00")
            log_time = datetime.fromisoformat(log_time_str).replace(tzinfo=None)
            delta_hours = int((now - log_time).total_seconds() // 3600)
            if 0 <= delta_hours < 24:
                hourly_counts[23 - delta_hours] += 1
        except Exception:
            pass

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_entities = total_entities / total_reqs if total_reqs else 0.0

    return {
        "total_requests": total_reqs,
        "total_entities_redacted": total_entities,
        "requests_blocked": blocked,
        "avg_entities_per_request": round(avg_entities, 2),
        "entity_type_breakdown": dict(breakdown),
        "requests_per_hour": hourly_counts,
        "model_type": settings.model_type,
        "avg_latency_ms": round(avg_latency, 2)
    }

@router.get("/recent-requests")
def get_recent_requests(admin_key: str = Depends(verify_admin_key)):
    skip_paths = ["/admin/stats", "/admin/recent-requests", "/dashboard", "/health", "/favicon.ico"]
    
    filtered = [log for log in audit_log_buffer if log.get("path") not in skip_paths]
    recent = list(reversed(filtered))[:50]
    
    formatted = []
    for log in recent:
        formatted.append({
            "request_id": log.get("request_id"),
            "timestamp": log.get("timestamp"),
            "entity_count": log.get("entity_count", 0),
            "redaction_applied": log.get("redaction_applied", False),
            "latency_ms": log.get("process_time_ms", 0.0),
            "blocked": log.get("status_code", 200) >= 400,
            "path": log.get("path")
        })
        
    return formatted

@router.post("/config")
def update_config(config: ConfigUpdateRequest, admin_key: str = Depends(verify_admin_key)):
    updates = {}
    
    if config.confidence_threshold is not None:
        settings.confidence_threshold = config.confidence_threshold
        updates["confidence_threshold"] = settings.confidence_threshold
        
    if config.rate_limit_per_minute is not None:
        updates["rate_limit_per_minute"] = config.rate_limit_per_minute

    if config.model_type is not None and config.model_type != settings.model_type:
        if config.model_type not in ["scratch", "bert"]:
            raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'scratch' or 'bert'.")
            
        settings.model_type = config.model_type
        updates["model_type"] = settings.model_type
        
        if settings.model_type == "scratch":
            from ml_engine.scratch_model.inference import ScratchNERInference
            detector.ml_model = ScratchNERInference(confidence_threshold=settings.confidence_threshold)
        elif settings.model_type == "bert":
            from ml_engine.bert_model.bert_inference import BertNERInference
            detector.ml_model = BertNERInference(confidence_threshold=settings.confidence_threshold)
            
        detector.model_type = settings.model_type

    return {
        "status": "updated",
        "new_config": updates
    }