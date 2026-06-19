# PulseAgent

A multi-agent RAG system built on LangGraph. Three specialist sub-agents — RetrievalAgent, VerifierAgent, SynthesisAgent — orchestrated by a SupervisorAgent that decomposes queries and fans out retrieval in parallel. Retrieval and NLI layers exposed as MCP tools. Containerized FastAPI service deployable on GCP Cloud Run.

Built to demonstrate production AI engineering skills: multi-agent orchestration, tool use, citation-grounded generation, NLI evaluation, and cloud deployment.

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
│                        │  Annotated reducer accumulates         │
│                        ▼                                        │
│              [aggregate_node]  dedup + rank                     │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────┐                               │
│  │     VerifierAgent sub-graph │                               │
│  │  NLI: cross-encoder/nli-    │                               │
│  │  deberta-v3-small ≥0.85     │                               │
│  │  ANSWER_WITH_CITATION /     │                               │
│  │  ABSTAIN per passage        │                               │
│  └──────────────┬──────────────┘                               │
│                 │  verified_chunks + decision                   │
│                 ▼                                               │
│  ┌─────────────────────────────┐                               │
│  │    SynthesisAgent sub-graph │                               │
│  │  [generator] grounded answer│                               │
│  │  [reflector] self-critique  │                               │
│  │  PASS → END  RETRY → gen   │                               │
│  └──────────────┬──────────────┘                               │
└─────────────────┼───────────────────────────────────────────────┘
                  ▼
         Route: ANSWER | ABSTAIN
```

Each specialist (RetrievalAgent, VerifierAgent, SynthesisAgent) is a separately compiled `StateGraph` subgraph. The Supervisor uses LangGraph's `Send()` API to fan sub-queries out to parallel RetrievalAgent instances; results accumulate via an `Annotated[List[dict], operator.add]` reducer.

---

## MCP Interface

Retrieval and NLI are exposed as MCP tools via `fastmcp`:

```
retrieve_passages(query: str)              → list[dict]
verify_citation(claim: str, passage: str)  → dict{verdict, confidence, passes}
```

Run the MCP server:
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

## Eval results (200-query, WixQA corpus)

| Metric | Value |
|--------|-------|
| Queries evaluated | 200 |
| Answer rate (ANSWER_WITH_CITATION) | 56.0% |
| Abstain rate (principled, no hallucination) | 44.0% |
| Error rate | 0.0% |
| Mean retrieval+NLI latency | 0.218s |
| P95 retrieval+NLI latency | 0.247s |

The abstain rate is a feature: the agent refuses to answer when it cannot produce a citation-verified response, rather than hallucinating.

Run eval (no LM Studio required):
```bash
python3 src/eval/eval_runner.py          # 200 queries
python3 src/eval/eval_runner.py --n 50   # quick sample
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph StateGraph + Send() fan-out |
| Specialist sub-agents | 3 compiled subgraphs (Retrieval, Verifier, Synthesis) |
| MCP tools | fastmcp (retrieve_passages, verify_citation) |
| LLM | Groq (cloud) / LM Studio (local) · qwen2.5-7b or llama-3.3-70b |
| Dense retrieval | fastembed BAAI/bge-small-en-v1.5 · 384-dim · Qdrant in-memory |
| Sparse retrieval | BM25Okapi (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| NLI verification | cross-encoder/nli-deberta-v3-small (HuggingFace) |
| API | FastAPI + uvicorn |
| Deployment | Docker + GCP Cloud Run |
| Corpus | Wix/WixQA · 6,221 articles · MIT license |

---

## Project structure

```
pulseagent/
├── main.py                     Entry point (--legacy flag uses original single-agent)
├── api.py                      FastAPI service (POST /query, GET /health)
├── config.py                   Env-var driven LLM config (local/Groq/any OpenAI-compat)
├── Dockerfile                  Container for GCP Cloud Run
├── requirements.txt
├── src/
│   ├── agents/                 Multi-agent system
│   │   ├── state.py            SupervisorState, RetrievalState, VerifierState, SynthesisState
│   │   ├── supervisor.py       SupervisorAgent: planner + fan-out + aggregate
│   │   ├── retrieval_agent.py  RetrievalAgent subgraph (BM25+dense+RRF)
│   │   ├── verifier_agent.py   VerifierAgent subgraph (NLI)
│   │   └── synthesis_agent.py  SynthesisAgent subgraph (generator+reflector)
│   ├── agent/                  Original single-agent (preserved, used by --legacy)
│   ├── mcp_server/
│   │   └── server.py           MCP tool server (stdio + SSE)
│   ├── tools/
│   │   ├── retriever_tool.py   @tool: hybrid RRF (3-part persistent cache)
│   │   └── nli_tool.py         @tool: NLI citation verification
│   ├── retrieval/              Bundled retrieval layer (corpus.py, indexer.py)
│   ├── citation/               Bundled NLI layer (entailment.py)
│   ├── corpus/                 Bundled corpus adapter (wixqa_adapter.py)
│   └── eval/
│       └── eval_runner.py      200-query evaluation harness
└── .cache/                     Persistent index (gitignored — built on first run)
```

---

## Setup

**Requirements:** Python 3.10+, LM Studio (local) or Groq API key (cloud)

```bash
pip install -r requirements.txt

# Run multi-agent supervisor (local, LM Studio at localhost:1234)
python3 main.py "how do I add a blog?"

# Run with Groq (no local LLM needed)
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_API_KEY=<your_groq_key>
export LLM_MODEL=llama-3.3-70b-versatile
python3 main.py "how do I add a blog?"

# Run FastAPI server
uvicorn api:app --reload --port 8000
# POST http://localhost:8000/query  {"question": "how do I add a blog?"}
```

**First run:** embeds 6,221 articles (~5-8 min, one-time). Saved to `.cache/`. All subsequent runs load in ~15s.

---

## Cloud deployment (GCP Cloud Run)

```bash
# Build and deploy
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

**Note on cold start:** the corpus cache (`.cache/`) is not baked into the image (it's gitignored). First request after a cold start triggers the embedding step (~5-8 min). The `/health` endpoint reports `cache: building` vs `cache: ready`. For production, mount the cache from Cloud Storage.

---

## Key engineering decisions

**Why LangGraph Send() for fan-out?**
`Send()` dispatches each sub-query to a separate RetrievalAgent instance that runs as an independent graph execution. Results accumulate via `Annotated[List[dict], operator.add]` in SupervisorState. This is the canonical LangGraph multi-agent parallel pattern — not cosmetic parallelism.

**Why compile each specialist as a separate StateGraph?**
Each specialist has its own typed state schema and is independently testable. The Supervisor doesn't need to know the internals of how retrieval or NLI work — it just invokes the compiled subgraph. This is the correct abstraction for multi-agent systems.

**Why Groq for cloud deployment?**
Groq's API is OpenAI-compatible (same format as LM Studio). Switching from local to cloud is two env vars. Free tier supports 14,400 requests/day — sufficient for a portfolio demo. No Kubernetes needed: Cloud Run abstracts container orchestration at this scale.

**Why 3-part cache instead of pickling the full index?**
The `RetrievalIndex` holds a fastembed ONNX `InferenceSession`, which is not picklable. Solution: serialize chunks as plain dicts, BM25 separately, vectors as numpy. Rebuild Qdrant in-memory from saved vectors on load. Cold start from cache: ~15s vs ~8 min fresh.

**Why NLI claim conversion?**
NLI entailment models expect declarative hypothesis-premise pairs. Questions always fail entailment. Converting "How do I add a blog?" → "This article provides information about: How do I add a blog" gives the cross-encoder a falsifiable statement it can actually verify.
