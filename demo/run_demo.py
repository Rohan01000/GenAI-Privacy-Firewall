import asyncio
import httpx
import json

async def run_demo():
    print("=== GenAI Privacy Firewall Demo ===")
    print("Make sure the FastAPI server is running on port 8000.")
    print("If you haven't started it, run: uvicorn main:app --reload\n")
    
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, my name is Alice Smith and my social security number is 123-45-6789. Please remember this."}
        ]
    }
    
    print("Sending payload via Proxy:")
    print(json.dumps(payload, indent=2))
    print("\nProcessing...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
            print(f"\nResponse Status: {response.status_code}")
            
            # Since this points to a real LLM URL in settings, it might fail if OpenAI key isn't set.
            # But the proxy redaction happens *before* the failure. 
            # Look at your server console to see the redaction logs!
            print("\nCheck your FastAPI server logs to see the redacted prompt sent to the LLM!")
            
        except httpx.ConnectError:
            print("Error: Could not connect to the firewall. Is it running?")

if __name__ == "__main__":
    asyncio.run(run_demo())