from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from typing import List, Dict
import re
import base64

# ==========================================
# CUSTOM RECOGNIZERS
# ==========================================

class ApiKeyRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("OpenAI Key", r"sk-[a-zA-Z0-9]{48}", 1.0),
            Pattern("GitHub PAT", r"ghp_[a-zA-Z0-9]{36}", 1.0),
            Pattern("AWS Access Key", r"AKIA[0-9A-Z]{16}", 1.0),
            Pattern("AWS Secret", r"(?<![A-Za-z0-9/+=])[0-9a-zA-Z/+]{40}(?![A-Za-z0-9/+=])", 0.8),
            Pattern("Google API", r"AIza[0-9A-Za-z\-_]{35}", 1.0),
            Pattern("Stripe Live", r"sk_live_[0-9a-zA-Z]{24}", 1.0)
        ]
        super().__init__(supported_entity="API_KEY", patterns=patterns)

class JwtTokenRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("JWT Token", r"[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+", 0.9)
        ]
        super().__init__(supported_entity="API_KEY", patterns=patterns)

    def validate_result(self, text: str) -> bool:
        # Additional validation to check if segments are valid base64url
        parts = text.split('.')
        if len(parts) != 3:
            return False
        try:
            for part in parts[:2]:  # Header and Payload
                # Add padding if necessary
                padded = part + '=' * (4 - len(part) % 4)
                base64.urlsafe_b64decode(padded)
            return True
        except Exception:
            return False

class PrivateKeyRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("RSA Private Key", r"-----BEGIN RSA PRIVATE KEY-----[\s\S]*?-----END RSA PRIVATE KEY-----", 1.0),
            Pattern("Generic Private Key", r"-----BEGIN PRIVATE KEY-----[\s\S]*?-----END PRIVATE KEY-----", 1.0),
            Pattern("EC Private Key", r"-----BEGIN EC PRIVATE KEY-----[\s\S]*?-----END EC PRIVATE KEY-----", 1.0)
        ]
        super().__init__(supported_entity="PASSWORD", patterns=patterns)

class InternalNetworkRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("10.x.x.x", r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", 1.0),
            Pattern("172.16-31.x.x", r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b", 1.0),
            Pattern("192.168.x.x", r"\b192\.168\.\d{1,3}\.\d{1,3}\b", 1.0),
            Pattern("Localhost", r"\blocalhost\b", 1.0),
            Pattern("Loopback", r"\b127\.0\.0\.1\b", 1.0)
        ]
        super().__init__(supported_entity="IP_ADDRESS", patterns=patterns)

class SecretPasswordRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("Secret Keyword Pattern", r"(?i)(password|passwd|secret|token|api_key)\s*[=:]\s*(\S+)", 0.8)
        ]
        super().__init__(supported_entity="PASSWORD", patterns=patterns)

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for pattern in self.patterns:
            matches = re.finditer(pattern.regex, text)
            for match in matches:
                start = match.start(2)
                end = match.end(2)
                results.append(RecognizerResult(
                    entity_type="PASSWORD",
                    start=start,
                    end=end,
                    score=pattern.score
                ))
        return results

class HexSecretRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("32/64 Hex String", r"\b[0-9a-fA-F]{32}\b|\b[0-9a-fA-F]{64}\b", 0.7)
        ]
        super().__init__(supported_entity="API_KEY", patterns=patterns)


# ==========================================
# MAIN DETECTOR CLASS
# ==========================================

class RuleBasedDetector:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        
        # Register custom recognizers
        self.analyzer.registry.add_recognizer(ApiKeyRecognizer())
        self.analyzer.registry.add_recognizer(JwtTokenRecognizer())
        self.analyzer.registry.add_recognizer(PrivateKeyRecognizer())
        self.analyzer.registry.add_recognizer(InternalNetworkRecognizer())
        self.analyzer.registry.add_recognizer(SecretPasswordRecognizer())
        self.analyzer.registry.add_recognizer(HexSecretRecognizer())
        
        # Map Presidio entities to our standard schema
        self.entity_mapping = {
            "PERSON": "PERSON",
            "EMAIL_ADDRESS": "EMAIL",
            "PHONE_NUMBER": "PHONE",
            "US_SSN": "SSN",
            "CREDIT_CARD": "CREDIT_CARD",
            "IP_ADDRESS": "IP_ADDRESS",
            "API_KEY": "API_KEY",
            "PASSWORD": "PASSWORD"
        }

    def detect(self, text: str) -> List[Dict]:
        if not text:
            return []
            
        # Run Presidio analysis
        results = self.analyzer.analyze(
            text=text, 
            language="en", 
            return_decision_process=False
        )
        
        standardized_results = []
        for result in results:
            if result.entity_type in self.entity_mapping:
                mapped_type = self.entity_mapping[result.entity_type]
                
                # Double check JWT validation if applicable
                if mapped_type == "API_KEY" and text[result.start:result.end].count('.') == 2:
                    if not JwtTokenRecognizer().validate_result(text[result.start:result.end]):
                        continue
                
                standardized_results.append({
                    "entity_type": mapped_type,
                    "value": text[result.start:result.end],
                    "start": result.start,
                    "end": result.end,
                    "confidence": 1.0 # Rule-based is considered deterministic here
                })
                
        # Sort by start position
        standardized_results.sort(key=lambda x: x["start"])
        return standardized_results

if __name__ == "__main__":
    detector = RuleBasedDetector()
    text = "My AWS key is AKIAIOSFODNN7EXAMPLE and local IP is 192.168.1.10. Contact john@example.com."
    print(detector.detect(text))