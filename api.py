"""
api.py — PulseAgent FastAPI service

POST /query  →  runs multi-agent supervisor (planner + retrieval fan-out +
                NLI verifier + synthesis) and returns structured response.

GET  /health →  liveness check (returns status + cache state)

LLM backend is env-var configurable:
  Local  (default): LM Studio at localhost:1234
  Cloud  (GCP):     LLM_BASE_URL=https://api.groq.com/openai/v1
                    LLM_API_KEY=<groq_key>
                    LLM_MODEL=llama-3.3-70b-versatile

Run locally:
  uvicorn api:app --reload --port 8000

GCP Cloud Run (after docker build):
  gcloud run deploy pulseagent --source . --region us-central1 \
    --set-env-vars LLM_BASE_URL=https://api.groq.com/openai/v1 \
    --set-secrets   LLM_API_KEY=groq-api-key:latest \
    --set-env-vars LLM_MODEL=llama-3.3-70b-versatile \
    --allow-unauthenticated
"""
from __future__ import annotations
import sys, time
from pathlib import Path

PA_ROOT = Path(__file__).parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="PulseAgent",
    description=(
        "Agentic RAG system: LangGraph Supervisor + 3 specialist sub-agents "
        "(RetrievalAgent, VerifierAgent, SynthesisAgent) over 6,221 WixQA articles."
    ),
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer:      str
    citations:   list[str]
    verdict:     str           # ANSWER | ABSTAIN
    sub_queries: list[str]
    latency_ms:  float


# ── Startup: pre-warm retrieval index ────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Warm the retrieval index in a background thread so uvicorn binds to
    port 8080 immediately (Cloud Run startup timeout ~60s).
    First run builds 3-part cache (~5-8 min). Subsequent starts: ~15s.
    Requests received before the index is ready will block inside the tool
    until it finishes — safe because the tool uses a module-level lock.
    """
    import threading
    def _warm():
        print("[api] Warming retrieval index in background...")
        from src.tools.retriever_tool import _get_index
        _get_index()
        print("[api] Index ready.")
    threading.Thread(target=_warm, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from pathlib import Path
    cache_dir = PA_ROOT / ".cache"
    cache_ok  = all((cache_dir / f).exists()
                    for f in ["chunks.pkl", "bm25.pkl", "vectors.npy"])
    return {"status": "ok", "cache": "ready" if cache_ok else "building"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    t0 = time.time()
    try:
        from src.agents.supervisor import run_supervisor
        result = run_supervisor(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer      = result.get("final_answer", ""),
        citations   = result.get("cited_ids", []),
        verdict     = result.get("route", "ABSTAIN"),
        sub_queries = result.get("sub_queries", []),
        latency_ms  = round((time.time() - t0) * 1000, 1),
    )
