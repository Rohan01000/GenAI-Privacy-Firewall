from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional

from config.settings import settings
from ml_engine.combined_detector import detector_instance
from ml_engine.redactor import Redactor

router = APIRouter()
redactor = Redactor(detector_instance)

class TextRequest(BaseModel):
    text: str

class EntityResponse(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float

class AnalysisResponse(BaseModel):
    original_text: str
    redacted_text: str
    entities: List[EntityResponse]

def verify_admin(x_admin_token: str = Header(...)):
    if x_admin_token != settings.admin_secret_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest, authorized: bool = Depends(verify_admin)):
    redacted_text, entities = redactor.redact(request.text)
    
    formatted_entities = [
        EntityResponse(
            entity_type=ent["entity_group"],
            start=ent["start"],
            end=ent["end"],
            score=ent["score"]
        ) for ent in entities
    ]
    
    return AnalysisResponse(
        original_text=request.text,
        redacted_text=redacted_text,
        entities=formatted_entities
    )

@router.get("/status")
async def admin_status(authorized: bool = Depends(verify_admin)):
    return {
        "engine_ready": detector_instance.is_ready(),
        "active_model": settings.model_type,
        "confidence_threshold": settings.confidence_threshold
    }