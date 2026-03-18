import re
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RedactionEngine:
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self.sessions: Dict[str, dict] = {}

    def redact(self, text: str, entities: List[dict]) -> Tuple[str, dict, str]:
        session_id = str(uuid.uuid4())
        
        # 1. Filter by confidence and log filtered-out entities
        above_threshold = []
        for ent in entities:
            if ent['confidence'] >= self.confidence_threshold:
                above_threshold.append(ent)
            else:
                logger.debug(
                    f"Filtered out entity: {ent['entity_type']} "
                    f"(Confidence: {ent['confidence']} < {self.confidence_threshold})"
                )

        # 2. Handle overlapping spans (keep higher confidence)
        # Sort by confidence descending to prioritize highest confidence first
        above_threshold.sort(key=lambda x: x['confidence'], reverse=True)
        accepted_entities = []
        
        for ent in above_threshold:
            is_overlapping = False
            for acc in accepted_entities:
                # Check for overlap: max(start1, start2) < min(end1, end2)
                if max(ent['start'], acc['start']) < min(ent['end'], acc['end']):
                    is_overlapping = True
                    break
            if not is_overlapping:
                accepted_entities.append(ent)

        # 3. Sort accepted entities by start position ascending
        accepted_entities.sort(key=lambda x: x['start'])

        # 4. Assign placeholders in forward order to maintain logical numbering (e.g., PERSON_1, PERSON_2)
        counters = {}
        for ent in accepted_entities:
            ent_type = ent['entity_type']
            counters[ent_type] = counters.get(ent_type, 0) + 1
            ent['placeholder'] = f"[{ent_type}_{counters[ent_type]}]"

        # 5. Process string replacement in reverse order to preserve string indices
        redacted_text = text
        session_mapping = {}
        
        for ent in reversed(accepted_entities):
            start = ent['start']
            end = ent['end']
            placeholder = ent['placeholder']
            original_value = text[start:end]
            
            session_mapping[placeholder] = original_value
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

        # 6. Store in sessions and return
        self.sessions[session_id] = session_mapping
        return redacted_text, session_mapping, session_id

    def reinsert(self, llm_response: str, session_mapping: dict) -> str:
        if not session_mapping:
            return llm_response

        reconstructed_response = llm_response
        for placeholder, original_value in session_mapping.items():
            # Extract the inner text of the placeholder (e.g., "PERSON_1" from "[PERSON_1]")
            inner_text = placeholder.strip("[]")
            
            # Build regex pattern to handle:
            # - Case insensitivity
            # - Optional whitespace around the brackets inside
            # - Replaces all occurrences globally by default in re.sub
            pattern = re.compile(r'\[\s*' + re.escape(inner_text) + r'\s*\]', re.IGNORECASE)
            
            reconstructed_response = pattern.sub(original_value, reconstructed_response)

        return reconstructed_response

    def validate_redaction(self, original: str, redacted: str, mapping: dict) -> bool:
        if not mapping:
            return True
        
        for placeholder, original_value in mapping.items():
            if len(original_value) <= 3:
                continue
            # Use word boundary check to avoid matching substrings inside placeholders
            pattern = re.compile(r'\b' + re.escape(original_value) + r'\b', re.IGNORECASE)
            if pattern.search(redacted):
                logger.warning(f"VALIDATE FAIL: '{original_value}' still found in redacted text")
                return False
        return True

    def cleanup(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_audit_log(self, text: str, entities: List[dict], redacted: bool) -> dict:
        entity_count_by_type = {}
        above_threshold = 0
        below_threshold = 0

        for ent in entities:
            ent_type = ent['entity_type']
            entity_count_by_type[ent_type] = entity_count_by_type.get(ent_type, 0) + 1
            
            if ent['confidence'] >= self.confidence_threshold:
                above_threshold += 1
            else:
                below_threshold += 1

        return {
            "session_id": str(uuid.uuid4()),  # Generates a tracking ID for the log
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entity_count_by_type": entity_count_by_type,
            "total_entities_detected": len(entities),
            "entities_above_threshold": above_threshold,
            "entities_below_threshold": below_threshold,
            "redaction_applied": redacted
        }