import httpx
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from config.settings import settings
from ml_engine.combined_detector import detector_instance
from ml_engine.redactor import Redactor

router = APIRouter()
logger = logging.getLogger(__name__)
redactor = Redactor(detector_instance)

async def forward_request(url: str, headers: dict, json_data: dict):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=json_data, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error communicating with target LLM: {e}")
            raise HTTPException(status_code=502, detail="Bad Gateway: Target LLM communication failed")

@router.post("/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.json()
    
    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Missing 'messages' in request body")

    # Intercept and Redact
    redacted_messages = []
    for msg in body["messages"]:
        if msg.get("role") in ["user", "system"]:
            original_text = msg.get("content", "")
            redacted_text, entities = redactor.redact(original_text)
            
            if entities:
                logger.info(f"Redacted {len(entities)} entities from prompt.")
            
            new_msg = msg.copy()
            new_msg["content"] = redacted_text
            redacted_messages.append(new_msg)
        else:
            redacted_messages.append(msg)
            
    body["messages"] = redacted_messages

    # Forward Headers (Filtering out host, etc.)
    safe_headers = {
        k: v for k, v in request.headers.items() 
        if k.lower() in ["authorization", "content-type", "accept"]
    }
    
    # Forward to actual LLM
    llm_response = await forward_request(settings.target_llm_url, safe_headers, body)
    return JSONResponse(content=llm_response)