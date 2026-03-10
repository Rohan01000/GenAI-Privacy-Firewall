# GenAI Privacy Firewall

An enterprise-grade, ML-powered proxy firewall for Large Language Models (LLMs). It intercepts outgoing prompts, detects Personally Identifiable Information (PII) using a combination of rule-based engines (Presidio) and custom ML models (BERT or custom LSTM), redacts the sensitive data, and forwards the cleaned prompt to the target LLM.

## Setup
1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_lg` (Required for Presidio)
3. `cp .env.example .env` and configure.
4. Run the demo: `python demo/run_demo.py`
5. Or run the server: `uvicorn main:app --reload --port 8000`