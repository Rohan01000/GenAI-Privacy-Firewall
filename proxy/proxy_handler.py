from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import json
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from ml_engine.combined_detector import CombinedDetector
from ml_engine.redactor import RedactionEngine
from proxy.middleware import update_last_log_entry

logger = logging.getLogger(__name__)

# Module-level singletons
router = APIRouter()
detector = CombinedDetector(
    model_type=settings.model_type, 
    confidence_threshold=settings.confidence_threshold
)
redactor = RedactionEngine(confidence_threshold=settings.confidence_threshold)
http_client = httpx.AsyncClient()
limiter = Limiter(key_func=get_remote_address)

# In-memory rate limiting dictionary
api_key_daily_counts: Dict[str, int] = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

@router.post("/chat/completions")
@limiter.limit("60/minute")
async def chat_completions(request: Request, chat_request: ChatRequest):
    # Step 1 - Size check
    content_length = request.headers.get("Content-Length")
    if content_length and int(content_length) > 51200:
        raise HTTPException(status_code=400, detail="Request too large")

    # Rate Limiting Logic (Daily 1000 limit per API Key)
    auth_header = request.headers.get("Authorization", "")
    api_key = auth_header.split("Bearer ")[-1] if "Bearer " in auth_header else get_remote_address(request)
    
    current_count = api_key_daily_counts.get(api_key, 0)
    if current_count >= 1000:
        raise HTTPException(status_code=429, detail="Daily limit exceeded")
    api_key_daily_counts[api_key] = current_count + 1

    total_entities = 0
    all_mappings = []
    all_session_ids = []
    fallback_mode = False

    # Step 2 - Extract and redact each message
    for message in chat_request.messages:
        try:
            entities = detector.detect_sync(message.content)
        except Exception as e:
            logger.error(f"ML Model error: {e}. Falling back to rules only.")
            fallback_mode = True
            try:
                fallback_detector = CombinedDetector(model_type="rules_only", confidence_threshold=settings.confidence_threshold)
                entities = fallback_detector.detect_sync(message.content)
            except ValueError:
                entities = detector.rule_detector.detect(message.content)

        total_entities += len(entities)
        
        redacted_text, mapping, session_id = redactor.redact(message.content, entities)
        valid = redactor.validate_redaction(message.content, redacted_text, mapping)
        
        if not valid:
            for sid in all_session_ids:
                redactor.cleanup(sid)
            raise HTTPException(status_code=400, detail="Redaction failed")

        message.content = redacted_text
        if mapping:
            all_mappings.append(mapping)
            all_session_ids.append(session_id)

    # Attach stats to request state for Audit Logging middleware
    request_id = getattr(request.state, "request_id", "unknown")
    update_last_log_entry(request_id, total_entities, total_entities > 0)
    request.state.entity_count = total_entities
    request.state.redaction_applied = total_entities > 0

    # Step 3 - Forward to LLM
    sanitized_body = chat_request.model_dump(exclude_unset=True)
    
    forwarded_headers = {
        "Content-Type": "application/json",
        "X-Firewall-Processed": "true"
    }
    if auth_header:
        forwarded_headers["Authorization"] = auth_header

    # Step 4b - Streaming response
    if chat_request.stream:
        async def stream_generator():
            full_accumulated_text = ""
            try:
                async with http_client.stream("POST", settings.target_llm_url, json=sanitized_body, headers=forwarded_headers) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        logger.error(f"Target LLM Error: {resp.status_code} - {error_text}")
                        raise HTTPException(status_code=502, detail="LLM API unavailable")

                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                if "content" in delta:
                                    full_accumulated_text += delta["content"]
                            except json.JSONDecodeError:
                                continue

                reconstructed_text = full_accumulated_text
                for mapping in all_mappings:
                    reconstructed_text = redactor.reinsert(reconstructed_text, mapping)

                chunk_data = {
                    "id": "chatcmpl-firewall-stream",
                    "object": "chat.completion.chunk",
                    "model": chat_request.model,
                    "choices": [{"index": 0, "delta": {"content": reconstructed_text}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
                
            finally:
                for sid in all_session_ids:
                    redactor.cleanup(sid)

        response = StreamingResponse(stream_generator(), media_type="text/event-stream")
        if fallback_mode:
            response.headers["X-Firewall-Mode"] = "fallback"
        return response

    # Step 4a - Non-streaming response
    else:
        try:
            response = await http_client.post(settings.target_llm_url, json=sanitized_body, headers=forwarded_headers)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Target LLM request failed: {e}")
            for sid in all_session_ids:
                redactor.cleanup(sid)
            raise HTTPException(status_code=502, detail="LLM API unavailable")

        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            llm_text = response_data["choices"][0]["message"].get("content", "")
            
            reconstructed_text = llm_text
            for mapping in all_mappings:
                reconstructed_text = redactor.reinsert(reconstructed_text, mapping)
                
            response_data["choices"][0]["message"]["content"] = reconstructed_text

        for sid in all_session_ids:
            redactor.cleanup(sid)

        json_response = JSONResponse(content=response_data)
        if fallback_mode:
            json_response.headers["X-Firewall-Mode"] = "fallback"
            
        return json_response