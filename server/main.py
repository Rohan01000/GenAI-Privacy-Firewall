"""
main.py  —  GenAI Privacy Firewall  (FastAPI)
==============================================
Endpoints
  POST /chat                  — redacts PII then proxies to Ollama
  GET  /admin/stats           — aggregate stats
  GET  /admin/recent-requests — last N requests
  GET  /audit/requests        — full audit log (original vs redacted)
  GET  /config                — current server config (model name, detection type)
  GET  /health                — liveness probe

Run:
  uvicorn server.main:app --reload --port 8000
"""

import logging
import pathlib
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.config import (
    CONFIDENCE_THRESHOLD, DEFAULT_MODEL, MAX_AUDIT_ENTRIES,
    MODEL_DIR, MODEL_TYPE, OLLAMA_BASE_URL, RATE_LIMIT_PER_MIN,
)
from server.redactor import redact, get_model

# ── Silence noisy polling logs ────────────────────────────────────────────────
logging.getLogger("uvicorn.access").addFilter(
    type("_PollFilter", (), {
        "filter": staticmethod(
            lambda record: not any(
                p in record.getMessage()
                for p in ["/admin/stats", "/admin/recent-requests",
                          "/audit/requests", "/config", "/health"]
            )
        )
    })()
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GenAI Privacy Firewall", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Frontend HTML routes ──────────────────────────────────────────────────────
_FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"

_PAGE_MAP = {
    "/":              "index.html",
    "/dashboard":     "index.html",
    "/dashboard.html":"index.html",
    "/chat":          "firewall_chat.html",
    "/chat.html":     "firewall_chat.html",
    "/audit":         "audit.html",
    "/audit.html":    "audit.html",
}

for _route, _file in _PAGE_MAP.items():
    _path = _FRONTEND_DIR / _file
    app.get(_route, include_in_schema=False)(
        lambda _p=_path: FileResponse(_p)
    )

# ── Runtime config ───────────────────────────────────────────────────────────
_runtime = {
    "model_type":            MODEL_TYPE,
    "confidence_threshold":  CONFIDENCE_THRESHOLD,
    "rate_limit_per_minute": RATE_LIMIT_PER_MIN,
    "default_model":         DEFAULT_MODEL,
}

# ── In-memory stores ──────────────────────────────────────────────────────────
_audit_log: deque = deque(maxlen=MAX_AUDIT_ENTRIES)
_recent:    deque = deque(maxlen=50)
_stats = {
    "total_requests":        0,
    "total_entities_redacted": 0,
    "requests_blocked":      0,
    "latency_sum_ms":        0.0,
    "entity_type_breakdown": defaultdict(int),
    "requests_per_hour":     [0] * 24,
}
_rate_buckets: Dict[str, deque] = {}

# ── Logger for chat requests ─────────────────────────────────────────────────
_log = logging.getLogger("firewall")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("\033[36m[FIREWALL]\033[0m %(message)s"))
    _log.addHandler(_h)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rate_limit(ip: str):
    now = time.time()
    bucket = _rate_buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= _runtime["rate_limit_per_minute"]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def _update_stats(result, blocked: bool = False):
    _stats["total_requests"] += 1
    _stats["total_entities_redacted"] += len(result.entities)
    _stats["latency_sum_ms"] += result.latency_ms
    if blocked:
        _stats["requests_blocked"] += 1
    for e in result.entities:
        _stats["entity_type_breakdown"][e.entity_type] += 1
    hour = datetime.now().hour
    _stats["requests_per_hour"][hour] += 1


# ── Pydantic models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:       Optional[str]   = None
    messages:    List[Message]
    temperature: Optional[float] = None


# System prompt to keep Ollama responses short
_SYSTEM_MSG = {
    "role": "system",
    "content": "Keep your responses brief and concise. Answer in 2-3 sentences max unless the user asks for detail.",
}


# ── Main proxy endpoint ───────────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    client_ip = request.client.host if request.client else "unknown"
    _rate_limit(client_ip)

    request_id = str(uuid.uuid4())
    use_model  = _runtime["model_type"] == "scratch"
    model_name = body.model or _runtime["default_model"]

    # ── Redact each message ──────────────────────────────────────────────────
    redacted_messages = []
    all_entities      = []
    all_originals     = []

    for msg in body.messages:
        result = redact(msg.content, model_dir=MODEL_DIR, use_model=use_model)
        all_originals.append({"role": msg.role, "content": msg.content})
        redacted_messages.append({"role": msg.role, "content": result.redacted_text})
        all_entities.extend(result.entities)

    redaction_applied = len(all_entities) > 0
    model_type_used   = "bilstm+regex" if (use_model and get_model(MODEL_DIR).available) else "regex"

    # Use latency from last message redaction
    last_latency = result.latency_ms if body.messages else 0.0

    class _FakeResult:
        entities   = all_entities
        latency_ms = last_latency

    _update_stats(_FakeResult())

    # ── Log: what was sent vs what Ollama will get ───────────────────────────
    for orig, red in zip(all_originals, redacted_messages):
        _log.info("ORIGINAL  : %s", orig["content"])
        _log.info("REDACTED  : %s", red["content"])
    if all_entities:
        for e in all_entities:
            _log.info("  ENTITY  : %s  %r → %s  (source: %s)",
                       e.entity_type, e.original, e.placeholder, e.source)
    else:
        _log.info("  NO PII DETECTED")

    # ── Audit log entry ──────────────────────────────────────────────────────
    _orig_flat     = " | ".join(m["content"] for m in all_originals)
    _redacted_flat = " | ".join(m["content"] for m in redacted_messages)

    audit_entry = {
        "request_id":         request_id,
        "timestamp":          datetime.utcnow().isoformat() + "Z",
        "original_prompt":    _orig_flat,
        "redacted_prompt":    _redacted_flat,
        "entities":           [
            {
                "entity_type": e.entity_type,
                "original":    e.original,
                "placeholder": e.placeholder,
                "source":      e.source,
                "confidence":  e.confidence,
            }
            for e in all_entities
        ],
        "entity_count":       len(all_entities),
        "redaction_applied":  redaction_applied,
        "detection_source":   model_type_used,
        "latency_ms":         last_latency,
        "blocked":            False,
    }
    _audit_log.appendleft(audit_entry)
    _recent.appendleft({
        "request_id":       request_id,
        "timestamp":        audit_entry["timestamp"],
        "entity_count":     len(all_entities),
        "redaction_applied": redaction_applied,
        "latency_ms":       last_latency,
        "blocked":          False,
    })

    # ── Forward to Ollama ─────────────────────────────────────────────────────
    ollama_messages = [_SYSTEM_MSG] + redacted_messages
    ollama_payload = {
        "model":    model_name,
        "messages": ollama_messages,
        "stream":   False,
    }
    if body.temperature is not None:
        ollama_payload["options"] = {"temperature": body.temperature}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_payload,
            )
            resp.raise_for_status()
            ollama_data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, detail="Cannot connect to Ollama. Is it running?")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code,
                            detail=f"Ollama error: {e.response.text}")

    # ── Build response ───────────────────────────────────────────────────────
    assistant_text = (
        ollama_data.get("message", {}).get("content", "")
        or ollama_data.get("response", "")
    )

    _log.info("RESPONSE  : %s", assistant_text[:200] + ("..." if len(assistant_text) > 200 else ""))
    _log.info("─" * 60)

    return {
        "request_id": request_id,
        "model":      model_name,
        "response":   assistant_text,
        "firewall": {
            "redaction_applied": redaction_applied,
            "entity_count":      len(all_entities),
            "detection_source":  model_type_used,
            "entities": [
                {
                    "entity_type": e.entity_type,
                    "original":    e.original,
                    "placeholder": e.placeholder,
                    "source":      e.source,
                }
                for e in all_entities
            ],
        },
    }


# ── Read-only endpoints ──────────────────────────────────────────────────────
@app.get("/config")
def get_config():
    """Returns current server config so frontends can read default model etc."""
    model_ready = get_model(MODEL_DIR).available
    return {
        "default_model":        _runtime["default_model"],
        "model_type":           _runtime["model_type"],
        "confidence_threshold": _runtime["confidence_threshold"],
        "rate_limit_per_minute":_runtime["rate_limit_per_minute"],
        "model_loaded":         model_ready,
    }


@app.get("/admin/stats")
def get_stats():
    total = _stats["total_requests"] or 1
    return {
        "total_requests":          _stats["total_requests"],
        "total_entities_redacted": _stats["total_entities_redacted"],
        "requests_blocked":        _stats["requests_blocked"],
        "avg_latency_ms":          _stats["latency_sum_ms"] / total,
        "entity_type_breakdown":   dict(_stats["entity_type_breakdown"]),
        "requests_per_hour":       _stats["requests_per_hour"],
        "model_type":              _runtime["model_type"],
    }


@app.get("/admin/recent-requests")
def recent_requests():
    return list(_recent)


@app.get("/audit/requests")
def audit_requests(limit: int = 100):
    return list(_audit_log)[:limit]


@app.get("/health")
def health():
    model_ready = get_model(MODEL_DIR).available
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "model_type":   _runtime["model_type"],
        "ollama_url":   OLLAMA_BASE_URL,
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from server.config import HOST, PORT
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=True)
