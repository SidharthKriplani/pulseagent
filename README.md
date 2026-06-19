<div align="center">

# PulseAgent — Citation-Grounded Multi-Agent RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6B6B?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![fastembed](https://img.shields.io/badge/fastembed-BAAI%2Fbge--small-8B5CF6?style=flat-square)](https://github.com/qdrant/fastembed)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F97316?style=flat-square)](https://groq.com/)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

> A citation-grounded multi-agent RAG system — built to answer: **when should an agent refuse to answer rather than hallucinate?**  
> Three specialist sub-agents orchestrated by a LangGraph Supervisor. NLI contract gates every response: `ANSWER_WITH_CITATION` or `ABSTAIN`. No hallucination path exists by design.

</div>

---

## Failure Mode Addressed

**When should an AI system abstain instead of guessing?** RAG pipelines fail when they produce plausible-sounding answers without verifiable grounding — when citation is treated as optional metadata rather than a hard contract. PulseAgent is built around making that enforcement explicit, measurable, and auditable.

The domain — 6,221 Wix Help Center articles — is the test environment. The abstain-vs-hallucinate decision is the thesis.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  SupervisorAgent  (LangGraph StateGraph)                        │
│                                                                 │
│  [planner_node]  ──  LLM decomposes query → sub-queries        │
│       │                                                         │
│       │  Send() fan-out (parallel)                              │
│       ├──────────────────────────────────┐                      │
│       ▼                                  ▼                      │
│  ┌──────────────┐               ┌──────────────┐               │
│  │RetrievalAgent│               │RetrievalAgent│  (×N subgraph)│
│  │  sub-graph   │               │  sub-graph   │               │
│  │  BM25+dense  │               │  BM25+dense  │               │
│  │  RRF fusion  │               │  RRF fusion  │               │
│  └──────┬───────┘               └──────┬───────┘               │
│         │  chunks                      │  chunks                │
│         └──────────────┬───────────────┘                       │
│                        │  Annotated[List[dict], operator.add]   │
│                        ▼                                        │
│              [aggregate_node]  dedup + rank                     │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────┐                               │
│  │     VerifierAgent sub-graph │                               │
│  │  cross-encoder/nli-deberta  │                               │
│  │  -v3-small · threshold 0.85 │                               │
│  │  ANSWER_WITH_CITATION /     │                               │
│  │  ABSTAIN per passage        │                               │
│  └──────────────┬──────────────┘                               │
│                 │  verified_chunks + contract_decision          │
│                 ▼                                               │
│  ┌─────────────────────────────┐                               │
│  │    SynthesisAgent sub-graph │                               │
│  │  [generator] grounded answer│                               │
│  │  [reflector] self-critique  │                               │
│  │  PASS → END  RETRY → gen   │                               │
│  └──────────────┬──────────────┘                               │
└─────────────────┼───────────────────────────────────────────────┘
                  ▼
       ANSWER_WITH_CITATION | ABSTAIN
```

Each specialist (RetrievalAgent, VerifierAgent, SynthesisAgent) is a separately compiled `StateGraph` subgraph. The Supervisor uses LangGraph's `Send()` API to fan sub-queries out to parallel RetrievalAgent instances; results accumulate via an `Annotated[List[dict], operator.add]` reducer.

---

## Eval Results (200-query, WixQA corpus)

| Metric | Value |
|--------|-------|
| Queries evaluated | 200 |
| ANSWER_WITH_CITATION rate | **56.0%** |
| ABSTAIN rate (principled, zero hallucination) | **44.0%** |
| Error rate | **0.0%** |
| NLI-verified citation precision | **75.9%** |
| Mean retrieval + NLI latency | 0.218s |
| P95 retrieval + NLI latency | 0.247s |

The 44% abstain rate is a feature, not a failure. The system refuses to answer when it cannot produce a citation-verified response — the alternative is hallucination.

Run eval (no LM Studio required — runs retrieval + NLI only):
```bash
python3 src/eval/eval_runner.py          # 200 queries
python3 src/eval/eval_runner.py --n 50   # quick sample
```

---

## MCP Interface

Retrieval and NLI are exposed as MCP tools via `fastmcp`:

```
retrieve_passages(query: str)              → list[dict]
verify_citation(claim: str, passage: str)  → dict{verdict, confidence, passes}
```

```bash
python3 src/mcp_server/server.py           # stdio (Claude Desktop)
python3 src/mcp_server/server.py --sse     # SSE transport
```

Claude Desktop config:
```json
{
  "mcpServers": {
    "pulseagent": {
      "command": "python3",
      "args": ["/path/to/pulseagent/src/mcp_server/server.py"]
    }
  }
}
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph `StateGraph` + `Send()` fan-out |
| Specialist sub-agents | 3 compiled subgraphs: Retrieval · Verifier · Synthesis |
| MCP tools | `fastmcp` (`retrieve_passages`, `verify_citation`) |
| LLM | Groq `llama-3.3-70b-versatile` (cloud) · LM Studio `qwen2.5-7b` (local) |
| Dense retrieval | fastembed `BAAI/bge-small-en-v1.5` · 384-dim · Qdrant in-memory |
| Sparse retrieval | BM25Okapi (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| NLI citation gate | `cross-encoder/nli-deberta-v3-small` · threshold 0.85 · 17ms/pair |
| Serving | FastAPI · `slowapi` rate limiting (10/min) · X-API-Key auth |
| Deployment | GCP Cloud Run · Secret Manager for API keys |
| Index cache | `chunks.pkl` + `bm25.pkl` + `vectors.npy` · committed · ~15s cold start |

---

## Key Engineering Decisions

**Why `Send()` for fan-out?**
`Send()` dispatches each sub-query to a separate RetrievalAgent instance running as an independent graph execution. Results accumulate via `Annotated[List[dict], operator.add]` in `SupervisorState`. This is canonical LangGraph parallel fan-out — not cosmetic threading.

**Why compile each specialist as a separate `StateGraph`?**
Each specialist has its own typed state schema and is independently testable. The Supervisor doesn't know the internals of retrieval or NLI — it invokes compiled subgraphs. Correct abstraction for multi-agent systems.

**Why NLI claim conversion?**
NLI entailment models expect declarative hypothesis-premise pairs — questions always fail entailment. Converting "How do I add a blog?" → "This article provides information about: How do I add a blog" gives the cross-encoder a falsifiable statement. Measured precision: 75.9%.

**Why 3-part cache instead of pickling the full index?**
The `RetrievalIndex` holds a fastembed ONNX `InferenceSession`, which is not picklable. Fix: serialize chunks as plain dicts, BM25 separately, vectors as NumPy. Rebuild Qdrant in-memory from saved vectors on load. Cold start from cache: ~15s vs ~8 min fresh.

**Why Groq for cloud?**
Groq's API is OpenAI-compatible — identical call format to LM Studio. Switching from local to cloud is two env vars. Free tier: 14,400 req/day. No Kubernetes needed at this scale.

---

## Project Structure

```
pulseagent/
├── api.py                      FastAPI service (auth, rate limiting, /query + /health)
├── config.py                   All config via env vars — no hardcoded secrets
├── main.py                     CLI entry point (local + interactive)
├── setup_pulseagent.py         Bootstrap scaffold
├── requirements.txt
├── Dockerfile
├── docs/
│   └── PulseAgent_Interview_Defense.pdf   22-page system defense
├── src/
│   ├── agents/
│   │   ├── state.py            SupervisorState, RetrievalState, VerifierState, SynthesisState
│   │   ├── supervisor.py       SupervisorAgent: planner + fan-out + aggregate
│   │   ├── retrieval_agent.py  RetrievalAgent subgraph (BM25+dense+RRF)
│   │   ├── verifier_agent.py   VerifierAgent subgraph (NLI)
│   │   └── synthesis_agent.py  SynthesisAgent subgraph (generator+reflector)
│   ├── agent/                  Original single-agent (preserved, --legacy flag)
│   ├── mcp_server/
│   │   └── server.py           MCP tool server (stdio + SSE)
│   ├── tools/
│   │   ├── retriever_tool.py   @tool: hybrid RRF (3-part persistent cache)
│   │   └── nli_tool.py         @tool: NLI citation verification
│   ├── retrieval/              Bundled retrieval layer
│   ├── citation/               Bundled NLI layer
│   ├── corpus/                 WixQA corpus adapter
│   └── eval/
│       └── eval_runner.py      200-query evaluation harness
└── .cache/                     bm25.pkl + chunks.pkl + vectors.npy (committed, 73MB)
```

---

## Setup

**Requirements:** Python 3.10+, LM Studio (local) or Groq API key (cloud)

```bash
git clone https://github.com/SidharthKriplani/pulseagent.git
cd pulseagent
pip install -r requirements.txt

# Local (LM Studio at localhost:1234)
python3 main.py "how do I add a blog to my Wix site?"

# Cloud (Groq — no local LLM needed)
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_API_KEY=<your_groq_key>
export LLM_MODEL=llama-3.3-70b-versatile
python3 main.py "how do I add a blog to my Wix site?"

# FastAPI server
uvicorn api:app --reload --port 8000
# POST http://localhost:8000/query  {"question": "..."}
```

**First run:** embeds 6,221 articles (~5–8 min, one-time). Saved to `.cache/`. All subsequent runs load in ~15s.

---

## Cloud Deployment (GCP Cloud Run)

```bash
gcloud run deploy pulseagent \
  --source . \
  --region us-central1 \
  --set-env-vars LLM_BASE_URL=https://api.groq.com/openai/v1 \
  --set-env-vars LLM_MODEL=llama-3.3-70b-versatile \
  --set-secrets  LLM_API_KEY=groq-api-key:latest \
  --memory 4Gi \
  --timeout 300 \
  --allow-unauthenticated
```

The `.cache/` directory is committed (73MB, all files under GitHub's 100MB limit) so Cloud Run cold starts load the pre-built index in ~15s rather than rebuilding from scratch.

---

## Related Projects

- **[PulseGuard](https://github.com/SidharthKriplani/pulseguard)** — credit risk governance system. Same "explicit abstain over silent failure" principle applied to ML model scoring: a champion that scores at 99.62% of its Bayes ceiling knows when it's at the limit of what the data can tell it.
- **[PulseDiscover](https://github.com/SidharthKriplani/pulsediscover)** — recommender decision system. Same "prove the claim before shipping" principle applied to ranking: offline metrics are audited for OPE bias before any serving policy ships.

---

*A personal portfolio / research project. Results are backed by the eval harness in `src/eval/`. If a number isn't reproducible by running `eval_runner.py`, it isn't claimed.*
