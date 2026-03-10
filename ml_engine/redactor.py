class Redactor:
    def __init__(self, detector):
        self.detector = detector

    def redact(self, text: str):
        entities = self.detector.detect_entities(text)
        if not entities:
            return text, []

        redacted_text = ""
        last_idx = 0
        
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            entity_type = ent["entity_group"]
            
            # Append text before entity
            redacted_text += text[last_idx:start]
            # Append redaction marker
            redacted_text += f"[REDACTED_{entity_type}]"
            
            last_idx = end
            
        # Append remaining text
        redacted_text += text[last_idx:]
        
        return redacted_text, entities