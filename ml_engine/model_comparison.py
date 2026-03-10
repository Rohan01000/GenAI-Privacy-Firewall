import time
from .bert_model.bert_inference import BertDetector
from .scratch_model.inference import ScratchDetector

def run_comparison(test_sentences):
    print("Loading models...")
    bert = BertDetector()
    scratch = ScratchDetector()
    
    try:
        bert.load()
        scratch.load()
    except Exception as e:
        print(f"Skipping comparison: Models must be trained first. ({e})")
        return

    results = []
    
    for text in test_sentences:
        # BERT Time
        t0 = time.time()
        b_res = bert.detect(text)
        bert_time = time.time() - t0
        
        # Scratch Time
        t0 = time.time()
        s_res = scratch.detect(text)
        scratch_time = time.time() - t0
        
        results.append({
            "text": text,
            "bert_entities": b_res,
            "bert_time": bert_time,
            "scratch_entities": s_res,
            "scratch_time": scratch_time
        })
        
    print("\n--- Model Comparison ---")
    for r in results:
        print(f"\nText: {r['text']}")
        print(f"BERT    ({r['bert_time']:.4f}s): {[e['entity_group'] for e in r['bert_entities']]}")
        print(f"Scratch ({r['scratch_time']:.4f}s): {[e['entity_group'] for e in r['scratch_entities']]}")

if __name__ == "__main__":
    sentences = [
        "My name is John Doe and I live in New York.",
        "Contact me at 555-123-4567 or admin@example.com."
    ]
    run_comparison(sentences)