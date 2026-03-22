"""config.py — Server-wide configuration (reads from .env)"""
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
DEFAULT_MODEL        = os.getenv("DEFAULT_MODEL",        "llama3")
HOST                 = os.getenv("HOST",                 "0.0.0.0")
PORT                 = int(os.getenv("PORT",             "8000"))
RATE_LIMIT_PER_MIN   = int(os.getenv("RATE_LIMIT_PER_MIN","60"))
MODEL_DIR            = os.getenv("MODEL_DIR",            "model/saved")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD","0.80"))
MODEL_TYPE           = os.getenv("MODEL_TYPE",           "scratch")   # scratch | regex
MAX_AUDIT_ENTRIES    = 200
