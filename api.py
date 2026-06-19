"""
api.py — PulseAgent FastAPI service

POST /query  →  multi-agent supervisor (auth required, rate-limited)
GET  /health →  liveness check (open, no auth)

Auth:   X-API-Key header. Key stored in Secret Manager as pulse-api-key,
        injected as PULSE_API_KEY env var at deploy time.

Rate:   10 requests/min per IP on /query (slowapi).

Obs:    Structured JSON logs on every request — request_id, query (truncated),
        route, latency_ms, cited_count, error. Cloud Run captures stdout and
        makes these queryable in Cloud Logging.

LLM backend (env-var configurable):
  Local  (default): LM Studio at localhost:1234
  Cloud  (GCP):     LLM_BASE_URL=https://api.groq.com/openai/v1
                    LLM_API_KEY=<groq_key>
                    LLM_MODEL=llama-3.3-70b-versatile
"""
from __future__ import annotations
import sys, os, time, uuid, logging, json
from pathlib import Path

PA_ROOT = Path(__file__).parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Structured logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",   # Cloud Run captures raw stdout — emit JSON directly
)
logger = logging.getLogger("pulseagent.api")


def _log(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))


# ── Auth ──────────────────────────────────────────────────────────────────────

API_KEY_NAME   = "X-API-Key"
PULSE_API_KEY  = os.getenv("PULSE_API_KEY", "")   # injected via Secret Manager
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _require_api_key(key: str = Security(api_key_header)) -> str:
    if not PULSE_API_KEY:
        return key                          # dev mode: key not configured, skip check
    if key != PULSE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Rate limiting ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="PulseAgent",
    description=(
        "Agentic RAG: LangGraph Supervisor + 3 specialist sub-agents "
        "(RetrievalAgent, VerifierAgent, SynthesisAgent) over 6,221 WixQA articles. "
        "Hybrid BM25+dense RRF retrieval, NLI citation verification, "
        "self-reflective synthesis. Auth: X-API-Key header."
    ),
    version="1.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer:      str
    citations:   list[str]
    verdict:     str
    sub_queries: list[str]
    latency_ms:  float
    request_id:  str


# ── Startup: pre-warm retrieval index in background ───────────────────────────

@app.on_event("startup")
async def startup_event():
    import threading
    def _warm():
        _log("index_warm_start")
        from src.tools.retriever_tool import _get_index
        _get_index()
        _log("index_warm_complete")
    threading.Thread(target=_warm, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    cache_dir = PA_ROOT / ".cache"
    cache_ok  = all((cache_dir / f).exists()
                    for f in ["chunks.pkl", "bm25.pkl", "vectors.npy"])
    return {"status": "ok", "cache": "ready" if cache_ok else "building"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(
    request: Request,
    req: QueryRequest,
    _key: str = Security(_require_api_key),
):
    request_id = str(uuid.uuid4())[:8]

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    t0 = time.time()
    try:
        from src.agents.supervisor import run_supervisor
        result = run_supervisor(req.question)
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        _log("query_error",
             request_id=request_id,
             query=req.question[:120],
             latency_ms=latency_ms,
             error=type(e).__name__)
        raise HTTPException(status_code=500, detail="Internal server error")

    latency_ms = round((time.time() - t0) * 1000, 1)
    citations  = result.get("cited_ids", [])

    _log("query_complete",
         request_id=request_id,
         query=req.question[:120],
         route=result.get("route", "ABSTAIN"),
         cited_count=len(citations),
         latency_ms=latency_ms)

    return QueryResponse(
        answer      = result.get("final_answer", ""),
        citations   = citations,
        verdict     = result.get("route", "ABSTAIN"),
        sub_queries = result.get("sub_queries", []),
        latency_ms  = latency_ms,
        request_id  = request_id,
    )
