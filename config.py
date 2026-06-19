"""
config.py — PulseAgent configuration

LLM backend is env-var driven:
  Local (default): LM Studio at localhost:1234
  Cloud (GCP):     set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL env vars
                   e.g. Groq: LLM_BASE_URL=https://api.groq.com/openai/v1
                               LLM_MODEL=llama-3.3-70b-versatile
"""
import os
from pathlib import Path

ROOT        = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs" / "evidence"

# LLM — override via env vars for cloud deployment (e.g. Groq)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "lm-studio")
LLM_MODEL    = os.getenv("LLM_MODEL",    "qwen2.5-7b-instruct")

# Backward-compat aliases (nodes.py uses these names)
LM_STUDIO_BASE_URL = LLM_BASE_URL
LM_STUDIO_API_KEY  = LLM_API_KEY
LM_STUDIO_MODEL    = LLM_MODEL

MAX_RETRIES              = 2
TOP_K_RETRIEVE           = 10
TOP_K_VERIFY             = 3
CONTEXT_WINDOW           = 1200
NLI_CONFIDENCE_THRESHOLD = 0.85
NLI_MODEL_NAME           = "cross-encoder/nli-deberta-v3-small"
MCP_HOST                 = "localhost"
MCP_PORT                 = 8765
