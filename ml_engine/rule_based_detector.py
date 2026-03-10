from presidio_analyzer import AnalyzerEngine
import logging

logger = logging.getLogger(__name__)

class RuleBasedDetector:
    def __init__(self):
        self.analyzer = None

    def load(self):
        try:
            self.analyzer = AnalyzerEngine()
            logger.info("Presidio Analyzer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Presidio: {e}")
            raise e

    def detect(self, text: str, threshold: float = 0.5):
        if not self.analyzer:
            return []
        
        results = self.analyzer.analyze(text=text, language='en')
        
        filtered_results = []
        for res in results:
            if res.score >= threshold:
                filtered_results.append({
                    "entity_group": res.entity_type,
                    "score": res.score,
                    "word": text[res.start:res.end],
                    "start": res.start,
                    "end": res.end
                })
        return filtered_results