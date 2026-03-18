import asyncio
from typing import List, Dict
from ml_engine.rule_based_detector import RuleBasedDetector

class CombinedDetector:
    def __init__(self, model_type: str = "scratch", confidence_threshold: float = 0.85):
        self.model_type = model_type
        
        # Initialize ML Model based on configuration
        if model_type == "scratch":
            from ml_engine.scratch_model.inference import ScratchNERInference
            self.ml_model = ScratchNERInference(confidence_threshold=confidence_threshold)
        elif model_type == "bert":
            from ml_engine.bert_model.bert_inference import BertNERInference
            self.ml_model = BertNERInference(confidence_threshold=confidence_threshold)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'scratch' or 'bert'.")
            
        # Initialize Rule-Based Engine
        self.rule_detector = RuleBasedDetector()
        
        print(f"CombinedDetector initialized with ML model: {model_type.upper()} and Rule-Based engine.")

    def _merge_results(self, ml_results: List[Dict], rule_results: List[Dict]) -> List[Dict]:
        """
        Merges results prioritizing rule-based exact matches over ML predictions.
        """
        # Step 1: Start with all rule-based results
        final_list = list(rule_results)
        
        # Step 2 & 3 & 4: Add ML results only if they don't overlap with existing rule results
        for ml in ml_results:
            overlap = False
            for rule in rule_results:
                # Check for overlap: max(start1, start2) < min(end1, end2)
                if max(ml["start"], rule["start"]) < min(ml["end"], rule["end"]):
                    overlap = True
                    break
                    
            if not overlap:
                final_list.append(ml)
                
        # Deduplicate exactly identical entries (same type, start, end)
        seen = set()
        deduped_list = []
        for ent in final_list:
            identifier = (ent["entity_type"], ent["start"], ent["end"])
            if identifier not in seen:
                seen.add(identifier)
                deduped_list.append(ent)
                
        # Sort by start position
        deduped_list.sort(key=lambda x: x["start"])
        return deduped_list

    async def detect(self, text: str) -> List[Dict]:
        if not text:
            return []
            
        loop = asyncio.get_event_loop()
        
        # Run both detectors concurrently using the thread pool executor
        ml_task = loop.run_in_executor(None, self.ml_model.detect_entities, text)
        rule_task = loop.run_in_executor(None, self.rule_detector.detect, text)
        
        ml_results, rule_results = await asyncio.gather(ml_task, rule_task)
        
        return self._merge_results(ml_results, rule_results)

    def detect_sync(self, text: str) -> List[Dict]:
        if not text:
            return []
            
        # Run sequentially
        ml_results = self.ml_model.detect_entities(text)
        rule_results = self.rule_detector.detect(text)
        
        return self._merge_results(ml_results, rule_results)

    def detect_batch(self, texts: List[str]) -> List[List[Dict]]:
        if not texts:
            return []
            
        # For batching, we process the ML model in batch, but rule-based sequentially
        # as Presidio doesn't have a native batched analyzer endpoint.
        ml_batch_results = self.ml_model.detect_entities_batch(texts)
        
        final_batch_results = []
        for i, text in enumerate(texts):
            rule_results = self.rule_detector.detect(text)
            merged = self._merge_results(ml_batch_results[i], rule_results)
            final_batch_results.append(merged)
            
        return final_batch_results


if __name__ == "__main__":
    # Test Block
    print("Initializing CombinedDetector (Sync Test)...")
    detector = CombinedDetector(model_type="scratch", confidence_threshold=0.0)
    
    test_text = "My name is John Smith. My IP is 192.168.1.1 and my AWS key is AKIAIOSFODNN7EXAMPLE."
    
    print(f"\nINPUT: {test_text}")
    results = detector.detect_sync(test_text)
    
    print("\nMERGED ENTITIES FOUND:")
    for ent in results:
        print(f"  - {ent['entity_type']}: \"{ent['value']}\" (pos {ent['start']}-{ent['end']}, conf: {ent['confidence']:.2f})")