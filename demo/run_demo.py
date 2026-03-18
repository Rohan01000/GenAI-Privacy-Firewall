import subprocess
import time
import requests
import json
import sys
import os

from ml_engine.combined_detector import CombinedDetector

MOCK_LLM_CODE = """
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/mock-llm/v1/chat/completions")
async def mock_chat(request: Request):
    body = await request.json()
    content = body.get("messages", [{}])[0].get("content", "")
    return JSONResponse({
        "choices": [{"message": {"content": content}}],
        "debug_received": content
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="warning")
"""

def setup_mock_server():
    with open("demo/mock_llm_server.py", "w") as f:
        f.write(MOCK_LLM_CODE)

def wrap_text(text: str, width: int = 58) -> str:
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (word + " ") if current_line else word + " "
        else:
            lines.append(current_line.rstrip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.rstrip())
    return "\n║   ".join(lines) if lines else ""

def run_demo():
    print("Initializing demo environment...")
    setup_mock_server()
    
    mock_llm_proc = None
    proxy_proc = None
    
    try:
        # Start Mock LLM
        mock_llm_proc = subprocess.Popen([sys.executable, "demo/mock_llm_server.py"])
        
        # Start Proxy Server with custom environment
        env = os.environ.copy()
        env["TARGET_LLM_URL"] = "http://localhost:8081/mock-llm/v1/chat/completions"
        env["PROXY_PORT"] = "8080"
        proxy_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8080"], env=env)
        
        print("Waiting for servers to start...")
        import requests as req
        for _ in range(30):
            try:
                req.get("http://localhost:8080/health", timeout=1)
                break
            except:
                time.sleep(1)
        else:
            print("Server failed to start!")
            return

        print("✅ Proxy server started on port 8080")
        print("✅ Mock LLM started on port 8081")
        
        # Initialize local detector for UI breakdown
        detector = CombinedDetector(confidence_threshold=0.85)
        
        prompts = [
            ("Clean prompt", "Can you explain how transformers work in NLP?"),
            ("Person + Email", "Please contact Sarah Johnson at sarah.j@acme.com about the Q4 report"),
            ("AWS API Key", "My AWS access key is AKIAIOSFODNN7EXAMPLE and it's not working"),
            ("SSN + Credit Card", "Account SSN 456-78-9012 with card 4532-1234-5678-9012 needs review"),
            ("Code + Password", "Debug this: conn = db.connect(host='prod-server', password='Xk9#mNp2!@3')"),
            ("Mixed entities", "User John Smith (SSN: 789-01-2345) logged in from 192.168.1.50 using key sk-" + "b"*48)
        ]
        
        summary_data = []
        total_entities_protected = 0
        
        for i, (title, text) in enumerate(prompts, 1):
            payload = {
                "model": "demo-model",
                "messages": [{"role": "user", "content": text}]
            }
            
            # Use unique bearer to avoid rate limiting across loop iterations
            headers = {"Authorization": f"Bearer demo-key-{i}"}
            
            response = requests.post("http://localhost:8080/v1/chat/completions", json=payload, headers=headers)
            response_json = response.json()
            
            sanitized_received = response_json.get("debug_received", "")
            restored_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            entities = detector.detect_sync(text)
            total_entities_protected += len(entities)
            is_redacted = len(entities) > 0
            
            summary_data.append((i, title, len(entities), "Yes" if is_redacted else "No"))
            
            print("╔" + "═" * 62 + "╗")
            print(f"║ PROMPT {i}: {title:<50} ║")
            print("╠" + "═" * 62 + "╣")
            print("║ ORIGINAL SENT:                                               ║")
            print(f"║   {wrap_text(text):<58} ║")
            print("╠" + "═" * 62 + "╣")
            print("║ SANITIZED (what LLM received):                               ║")
            print(f"║   {wrap_text(sanitized_received):<58} ║")
            print("╠" + "═" * 62 + "╣")
            print("║ ENTITIES DETECTED:                                           ║")
            
            if not entities:
                print("║   (None)                                                     ║")
            else:
                for ent in entities:
                    ent_str = f"• {ent['entity_type']}: \"{ent['value']}\" (confidence: {ent['confidence']:.2f})"
                    print(f"║   {ent_str:<58} ║")
                    
            print("╠" + "═" * 62 + "╣")
            print("║ RESTORED RESPONSE:                                           ║")
            print(f"║   {wrap_text(restored_response):<58} ║")
            print("╚" + "═" * 62 + "╝\n")
            time.sleep(0.5)
            
        # Print Summary Table
        print("╔" + "═" * 62 + "╗")
        print("║                   DEMO SUMMARY                               ║")
        print("╠══════╦══════════════════════╦═══════════╦═══════════════════╣")
        print("║  #   ║ Prompt               ║ Entities  ║ Redacted?         ║")
        print("╠══════╬══════════════════════╬═══════════╬═══════════════════╣")
        
        for num, title, ent_count, redacted in summary_data:
            print(f"║  {num:<3} ║ {title:<20} ║     {ent_count:<5} ║ {redacted:<17} ║")
            
        print("╠══════╩══════════════════════╩═══════════╩═══════════════════╣")
        print(f"║ Total entities protected: {total_entities_protected:<34} ║")
        print("╚" + "═" * 62 + "╝\n")

    finally:
        if mock_llm_proc:
            mock_llm_proc.terminate()
        if proxy_proc:
            proxy_proc.terminate()
            
        if os.path.exists("demo/mock_llm_server.py"):
            os.remove("demo/mock_llm_server.py")
            
        print("Demo complete. Both servers stopped.")

if __name__ == "__main__":
    run_demo()