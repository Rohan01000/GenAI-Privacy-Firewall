import uuid
import time
import json
from datetime import datetime
from collections import deque
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request

# In-memory circular buffer for the admin dashboard
audit_log_buffer = deque(maxlen=1000)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        return response

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time_ms = (time.time() - start_time) * 1000
        request.state.process_time_ms = process_time_ms
        response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"
        
        return response

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Safely extract state variables set by other middlewares and handlers
        request_id = getattr(request.state, "request_id", "unknown")
        process_time_ms = getattr(request.state, "process_time_ms", 0.0)
        entity_count = getattr(request.state, "entity_count", 0)
        redaction_applied = getattr(request.state, "redaction_applied", False)
        
        # Determine client IP
        client_ip = "unknown"
        if request.client and request.client.host:
            client_ip = request.client.host
            
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": round(process_time_ms, 2),
            "entity_count": entity_count,
            "redaction_applied": redaction_applied,
            "client_ip": client_ip
        }
        
        # Write to stdout as newline-delimited JSON
        print(json.dumps(log_entry))
        
        # Maintain in memory for dashboard
        audit_log_buffer.append(log_entry)
        
        return response

def register_middleware(app: FastAPI) -> None:
    """
    Registers all middleware in the exact requested order.
    FastAPI evaluates middleware in the reverse order they are added, 
    but we execute the exact sequential registration as mandated.
    """
    app.add_middleware(AuditLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

def update_last_log_entry(request_id: str, entity_count: int, redaction_applied: bool):
    for entry in reversed(audit_log_buffer):
        if entry.get("request_id") == request_id:
            entry["entity_count"] = entity_count
            entry["redaction_applied"] = redaction_applied
            break