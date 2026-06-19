"""
config.py — PulseAgent configuration
"""
from pathlib import Path

ROOT = Path(__file__).parent
PULSEKNOWLEDGE_ROOT = ROOT.parent / "pulseknowledge"
OUTPUTS_DIR = ROOT / "outputs" / "evidence"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"
LM_STUDIO_MODEL    = "qwen2.5-7b-instruct"

MAX_RETRIES       = 2
TOP_K_RETRIEVE    = 10
TOP_K_VERIFY      = 3
CONTEXT_WINDOW    = 1200
NLI_CONFIDENCE_THRESHOLD = 0.85
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MCP_HOST = "localhost"
MCP_PORT = 8765
