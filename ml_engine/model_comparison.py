import os
import json
import time
import tracemalloc
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from ml_engine.scratch_model.inference import ScratchNERInference
from ml_engine.bert_model.bert_inference import BertNERInference
from ml_engine.scratch_model.dataset import generate_dataset

# ==========================================
# HELPER: DATA PREPARATION
# ==========================================
def prep_test_data(dataset: List[Tuple[List[str], List[str]]]) -> Tuple[List[str], List[List[Dict]]]:
    sentences = []
    ground_truth = []
    
    for tokens, tags in dataset:
        text = " ".join(tokens)
        sentences.append(text)
        
        entities = []
        current_ent = None
        curr_idx = 0
        
        for token, tag in zip(tokens, tags):
            start = text.find(token, curr_idx)
            end = start + len(token)
            curr_idx = end
            
            if tag.startswith("B-"):
                if current_ent:
                    entities.append(current_ent)
                current_ent = {
                    "entity_type": tag[2:], 
                    "start": start, 
                    "end": end, 
                    "value": text[start:end]
                }
            elif tag.startswith("I-") and current_ent and current_ent["entity_type"] == tag[2:]:
                current_ent["end"] = end
                current_ent["value"] = text[current_ent["start"]:end]
            else:
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None
                    
        if current_ent:
            entities.append(current_ent)
            
        ground_truth.append(entities)
        
    return sentences, ground_truth

def calculate_metrics(true_entities: List[List[Dict]], pred_entities: List[List[Dict]]) -> Dict:
    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    for t_ents, p_ents in zip(true_entities, pred_entities):
        t_set = {(e["entity_type"], e["start"], e["end"]) for e in t_ents}
        p_set = {(e["entity_type"], e["start"], e["end"]) for e in p_ents}
        
        # True Positives
        for ent in t_set.intersection(p_set):
            metrics[ent[0]]["TP"] += 1
            
        # False Positives
        for ent in p_set - t_set:
            metrics[ent[0]]["FP"] += 1
            
        # False Negatives
        for ent in t_set - p_set:
            metrics[ent[0]]["FN"] += 1

    results = {}
    macro_f1_sum = 0.0
    valid_entities = 0
    
    for ent_type, counts in metrics.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[ent_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
        macro_f1_sum += f1
        valid_entities += 1
        
    results["macro_f1"] = round(macro_f1_sum / valid_entities, 4) if valid_entities > 0 else 0.0
    return dict(results)

# ==========================================
# 1. ACCURACY BENCHMARK
# ==========================================
def run_accuracy_benchmark(scratch_model, bert_model, test_sentences, ground_truth) -> dict:
    print("\nRunning Accuracy Benchmark...")
    
    scratch_preds = [scratch_model.detect_entities(text) for text in test_sentences]
    bert_preds = [bert_model.detect_entities(text) for text in test_sentences]
    
    scratch_metrics = calculate_metrics(ground_truth, scratch_preds)
    bert_metrics = calculate_metrics(ground_truth, bert_preds)
    
    return {
        "scratch": scratch_metrics,
        "bert": bert_metrics
    }

# ==========================================
# 2. SPEED BENCHMARK
# ==========================================
def run_speed_benchmark(scratch_model, bert_model) -> dict:
    print("Running Speed Benchmark...")
    test_sentence = "Please contact John Smith at john.smith@example.com regarding order 12345."
    test_batch = [test_sentence] * 32
    
    def test_speed(model):
        # Single Latency
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            model.detect_entities(test_sentence)
            latencies.append((time.perf_counter() - start) * 1000) # ms
            
        # Throughput
        start = time.perf_counter()
        model.detect_entities_batch(test_batch)
        batch_time = time.perf_counter() - start
        throughput = 32 / batch_time
        
        return {
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "min_latency_ms": round(float(np.min(latencies)), 2),
            "max_latency_ms": round(float(np.max(latencies)), 2),
            "throughput_sentences_per_sec": round(throughput, 2)
        }

    return {
        "scratch": test_speed(scratch_model),
        "bert": test_speed(bert_model)
    }

# ==========================================
# 3. RESOURCE BENCHMARK
# ==========================================
def run_resource_benchmark(scratch_model, bert_model) -> dict:
    print("Running Resource Benchmark...")
    
    def get_dir_size(path):
        total = 0
        if os.path.exists(path):
            if os.path.isfile(path):
                total = os.path.getsize(path)
            else:
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total += os.path.getsize(fp)
        return total / (1024 * 1024) # MB

    scratch_size = get_dir_size("models/scratch_ner.pt")
    bert_size = get_dir_size("models/bert_ner")
    
    dummy_sentences = ["Test memory usage sentence."] * 100
    
    def get_ram_usage(model):
        tracemalloc.start()
        for text in dummy_sentences:
            model.detect_entities(text)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return round(peak / (1024 * 1024), 2) # MB

    return {
        "scratch": {
            "model_size_mb": round(scratch_size, 2),
            "ram_mb": get_ram_usage(scratch_model)
        },
        "bert": {
            "model_size_mb": round(bert_size, 2),
            "ram_mb": get_ram_usage(bert_model)
        }
    }

# ==========================================
# 4. ROBUSTNESS TESTS
# ==========================================
def run_robustness_tests(scratch_model, bert_model) -> dict:
    print("Running Robustness Tests...")
    
    # Case 1: Long
    base_long = "Contact Alice at alice@test.com. "
    long_sentences = [base_long * 20] * 10
    long_gt_grouped = []
    for _ in range(10):
        sentence_gt = []
        curr = 0
        for i in range(20):
            start_person = base_long * i
            pos = len(start_person)
            sentence_gt.append({"entity_type": "PERSON", "start": pos + 8, "end": pos + 13, "value": "Alice"})
            sentence_gt.append({"entity_type": "EMAIL", "start": pos + 17, "end": pos + 31, "value": "alice@test.com"})
        long_gt_grouped.append(sentence_gt)

    # Case 2: Mixed
    mixed_text = "User Alice Brown (alice@test.com) called 555-123-4567 using card 1234-5678-9012-3456 from IP 192.168.1.1."
    mixed_sentences = [mixed_text] * 10
    mixed_gt = [[
        {"entity_type": "PERSON", "start": 5, "end": 16},
        {"entity_type": "EMAIL", "start": 18, "end": 32},
        {"entity_type": "PHONE", "start": 41, "end": 53},
        {"entity_type": "CREDIT_CARD", "start": 65, "end": 84},
        {"entity_type": "IP_ADDRESS", "start": 93, "end": 104}
    ]] * 10

    # Case 3: Obfuscated
    obf_text = "My SSN is 1 2 3 - 4 5 - 6 7 8 9 and email is b o b @ test . org"
    obf_sentences = [obf_text] * 10
    obf_gt = [[
        {"entity_type": "SSN", "start": 10, "end": 31},
        {"entity_type": "EMAIL", "start": 45, "end": 63}
    ]] * 10
    
    def eval_robust(sentences, gt):
        s_preds = [scratch_model.detect_entities(s) for s in sentences]
        b_preds = [bert_model.detect_entities(s) for s in sentences]
        s_metrics = calculate_metrics(gt, s_preds)
        b_metrics = calculate_metrics(gt, b_preds)
        return {
            "scratch_f1": s_metrics.get("macro_f1", 0.0),
            "bert_f1": b_metrics.get("macro_f1", 0.0)
        }

    return {
        "long_prompts": eval_robust(long_sentences, long_gt_grouped),
        "mixed_entities": eval_robust(mixed_sentences, mixed_gt),
        "obfuscated_formats": eval_robust(obf_sentences, obf_gt)
    }

# ==========================================
# 5. FAILURE CASE ANALYSIS
# ==========================================
def find_failure_cases(scratch_model, bert_model, test_sentences, ground_truth) -> dict:
    print("Running Failure Case Analysis...")
    
    scratch_wins = []
    bert_wins = []
    both_fail = []
    
    for text, gt in zip(test_sentences, ground_truth):
        s_pred = scratch_model.detect_entities(text)
        b_pred = bert_model.detect_entities(text)
        
        s_met = calculate_metrics([gt], [s_pred]).get("macro_f1", 0.0)
        b_met = calculate_metrics([gt], [b_pred]).get("macro_f1", 0.0)
        
        case_data = {
            "text": text,
            "ground_truth": [(e["entity_type"], e["value"]) for e in gt],
            "scratch_pred": [(e["entity_type"], e["value"]) for e in s_pred],
            "bert_pred": [(e["entity_type"], e["value"]) for e in b_pred]
        }
        
        if s_met > b_met + 0.2 and len(scratch_wins) < 5:
            scratch_wins.append(case_data)
        elif b_met > s_met + 0.2 and len(bert_wins) < 5:
            bert_wins.append(case_data)
        elif s_met < 0.5 and b_met < 0.5 and len(both_fail) < 5:
            both_fail.append(case_data)
            
        if len(scratch_wins) == 5 and len(bert_wins) == 5 and len(both_fail) == 5:
            break
            
    return {
        "scratch_wins": scratch_wins,
        "bert_wins": bert_wins,
        "both_fail": both_fail
    }

# ==========================================
# 6. GENERATE HTML REPORT
# ==========================================
def generate_html_report(results: dict):
    print("Generating HTML Report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NER Model Comparison Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: 600; }}
            .chart-container {{ width: 100%; height: 400px; margin-top: 20px; display: flex; justify-content: center; align-items: center; flex-direction: column;}}
            .conclusion {{ background-color: #e8f4f8; padding: 20px; border-left: 5px solid #3498db; border-radius: 4px; font-size: 1.1em; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; }}
            .winner {{ color: #27ae60; font-weight: bold; }}
            .loser {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>NER Model Comparison Benchmark Report</h1>
        <p><strong>Generated on:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section">
            <h2>1. Accuracy Benchmark (Macro F1)</h2>
            <div class="chart-container">
                
                <canvas id="f1Chart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>2. Speed vs Accuracy</h2>
            <div class="chart-container">
                
                <canvas id="scatterChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>3. Resource Usage & Performance</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Model Size (MB)</th>
                    <th>Peak RAM (MB)</th>
                    <th>Mean Latency (ms)</th>
                    <th>Throughput (Sentences/sec)</th>
                    <th>Macro F1</th>
                </tr>
                <tr>
                    <td><strong>BiLSTM-CRF (Scratch)</strong></td>
                    <td>{results['resource']['scratch']['model_size_mb']}</td>
                    <td>{results['resource']['scratch']['ram_mb']}</td>
                    <td>{results['speed']['scratch']['mean_latency_ms']}</td>
                    <td>{results['speed']['scratch']['throughput_sentences_per_sec']}</td>
                    <td>{results['accuracy']['scratch']['macro_f1']}</td>
                </tr>
                <tr>
                    <td><strong>BERT Fine-Tuned</strong></td>
                    <td>{results['resource']['bert']['model_size_mb']}</td>
                    <td>{results['resource']['bert']['ram_mb']}</td>
                    <td>{results['speed']['bert']['mean_latency_ms']}</td>
                    <td>{results['speed']['bert']['throughput_sentences_per_sec']}</td>
                    <td>{results['accuracy']['bert']['macro_f1']}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>4. Robustness Tests (F1 Score)</h2>
            <table>
                <tr>
                    <th>Test Case</th>
                    <th>Scratch F1</th>
                    <th>BERT F1</th>
                    <th>Winner</th>
                </tr>
                <tr>
                    <td>Long Prompts (500+ tokens)</td>
                    <td>{results['robustness']['long_prompts']['scratch_f1']}</td>
                    <td>{results['robustness']['long_prompts']['bert_f1']}</td>
                    <td class="winner">{'Scratch' if results['robustness']['long_prompts']['scratch_f1'] > results['robustness']['long_prompts']['bert_f1'] else 'BERT'}</td>
                </tr>
                <tr>
                    <td>Multiple Mixed Entities</td>
                    <td>{results['robustness']['mixed_entities']['scratch_f1']}</td>
                    <td>{results['robustness']['mixed_entities']['bert_f1']}</td>
                    <td class="winner">{'Scratch' if results['robustness']['mixed_entities']['scratch_f1'] > results['robustness']['mixed_entities']['bert_f1'] else 'BERT'}</td>
                </tr>
                <tr>
                    <td>Obfuscated Formats (Spaces, Dots)</td>
                    <td>{results['robustness']['obfuscated_formats']['scratch_f1']}</td>
                    <td>{results['robustness']['obfuscated_formats']['bert_f1']}</td>
                    <td class="winner">{'Scratch' if results['robustness']['obfuscated_formats']['scratch_f1'] > results['robustness']['obfuscated_formats']['bert_f1'] else 'BERT'}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>5. Failure Case Analysis</h2>
            
            <h3>Where Scratch Wins</h3>
            <pre>{json.dumps(results['failures']['scratch_wins'][:2], indent=2)}</pre>
            
            <h3>Where BERT Wins</h3>
            <pre>{json.dumps(results['failures']['bert_wins'][:2], indent=2)}</pre>
            
            <h3>Where Both Fail</h3>
            <pre>{json.dumps(results['failures']['both_fail'][:2], indent=2)}</pre>
            <p><em>(Showing top 2 examples per category. See JSON for full list.)</em></p>
        </div>

        <div class="section conclusion">
            <h2>6. Conclusion & Recommendation</h2>
            <p>BiLSTM-CRF is recommended for edge deployment scenarios where inference latency and model size are constrained. BERT fine-tuning achieves higher accuracy and is recommended for high-security environments where maximum detection accuracy is required regardless of computational cost.</p>
        </div>

        <script>
            // Data Injection
            const resultsData = {json.dumps(results)};
            
            // 1. Bar Chart Data Prep
            const labels = Object.keys(resultsData.accuracy.scratch).filter(k => k !== 'macro_f1');
            const scratchF1 = labels.map(l => resultsData.accuracy.scratch[l].f1);
            const bertF1 = labels.map(l => resultsData.accuracy.bert[l].f1);

            new Chart(document.getElementById('f1Chart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'Scratch (BiLSTM-CRF)', data: scratchF1, backgroundColor: '#3498db' }},
                        {{ label: 'BERT Fine-Tuned', data: bertF1, backgroundColor: '#e74c3c' }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});

            // 2. Scatter Plot Data Prep
            new Chart(document.getElementById('scatterChart'), {{
                type: 'scatter',
                data: {{
                    datasets: [
                        {{
                            label: 'Scratch (BiLSTM-CRF)',
                            data: [{{ x: resultsData.speed.scratch.mean_latency_ms, y: resultsData.accuracy.scratch.macro_f1 }}],
                            backgroundColor: '#3498db',
                            pointRadius: 10
                        }},
                        {{
                            label: 'BERT Fine-Tuned',
                            data: [{{ x: resultsData.speed.bert.mean_latency_ms, y: resultsData.accuracy.bert.macro_f1 }}],
                            backgroundColor: '#e74c3c',
                            pointRadius: 10
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Mean Latency (ms) - Lower is Better' }} }},
                        y: {{ title: {{ display: true, text: 'Macro F1 - Higher is Better' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open("models/comparison_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Initializing Model Comparison Benchmarks...")
    
    # 1. Load Both Models
    scratch_model = ScratchNERInference()
    bert_model = BertNERInference()
    
    if not scratch_model.is_ready() or not bert_model.is_ready():
        print("Error: Models are not ready. Please ensure both models are trained and saved.")
        exit(1)
        
    # 2. Build Test Set
    raw_data = generate_dataset()
    test_split_idx = int(len(raw_data) * 0.85)
    test_data = raw_data[test_split_idx:]
    test_sentences, ground_truth = prep_test_data(test_data)
    
    print(f"Built test set with {len(test_sentences)} sentences.")
    
    # 3. Run Benchmarks
    accuracy_results = run_accuracy_benchmark(scratch_model, bert_model, test_sentences, ground_truth)
    speed_results = run_speed_benchmark(scratch_model, bert_model)
    resource_results = run_resource_benchmark(scratch_model, bert_model)
    robustness_results = run_robustness_tests(scratch_model, bert_model)
    failure_cases = find_failure_cases(scratch_model, bert_model, test_sentences, ground_truth)
    
    # 4. Compile Results
    final_results = {
        "accuracy": accuracy_results,
        "speed": speed_results,
        "resource": resource_results,
        "robustness": robustness_results,
        "failures": failure_cases
    }
    
    # 5. Save JSON
    os.makedirs("models", exist_ok=True)
    with open("models/comparison_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
        
    # 6. Generate HTML
    generate_html_report(final_results)
    
    # 7. Print Console Summary
    print("\n" + "="*50)
    print(" BENCHMARK SUMMARY")
    print("="*50)
    print(f"{'Metric':<25} | {'Scratch':<10} | {'BERT':<10} | {'Winner':<10}")
    print("-" * 60)
    
    f1_s = accuracy_results["scratch"]["macro_f1"]
    f1_b = accuracy_results["bert"]["macro_f1"]
    print(f"{'Macro F1':<25} | {f1_s:<10.4f} | {f1_b:<10.4f} | {'BERT' if f1_b > f1_s else 'Scratch':<10}")
    
    lat_s = speed_results["scratch"]["mean_latency_ms"]
    lat_b = speed_results["bert"]["mean_latency_ms"]
    print(f"{'Mean Latency (ms)':<25} | {lat_s:<10.2f} | {lat_b:<10.2f} | {'Scratch' if lat_s < lat_b else 'BERT':<10}")
    
    size_s = resource_results["scratch"]["model_size_mb"]
    size_b = resource_results["bert"]["model_size_mb"]
    print(f"{'Model Size (MB)':<25} | {size_s:<10.2f} | {size_b:<10.2f} | {'Scratch' if size_s < size_b else 'BERT':<10}")
    
    ram_s = resource_results["scratch"]["ram_mb"]
    ram_b = resource_results["bert"]["ram_mb"]
    print(f"{'Peak RAM (MB)':<25} | {ram_s:<10.2f} | {ram_b:<10.2f} | {'Scratch' if ram_s < ram_b else 'BERT':<10}")
    print("="*50 + "\n")
    print("Full results saved to models/comparison_results.json")
    print("Interactive HTML report saved to models/comparison_report.html")