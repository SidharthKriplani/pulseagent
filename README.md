# PulseAgent

An agentic AI system built on a production-grade RAG stack. PulseAgent wraps hybrid BM25+dense retrieval and NLI citation verification inside a LangGraph StateGraph, adding multi-step planning, tool use, and self-reflection on top of the retrieval layer.

Built to demonstrate production AI engineering skills for Staff/Senior AI Engineer roles — specifically: agentic orchestration, tool use patterns, citation-grounded generation, and evaluation rigor.

---

## What it does

A user query enters the graph and flows through five nodes before an answer is returned:

```
User Query
    │
    ▼
[planner]        Decomposes query into search intent + up to 3 sub-queries (LM Studio)
    │
    ▼
[retriever]      Hybrid RRF retrieval: BM25 + dense vector (BAAI/bge-small-en-v1.5, 384-dim)
                 over 6,221 Wix help center articles. Deduplicates across sub-queries.
    │
    ▼
[nli_verifier]   NLI citation check: cross-encoder/nli-deberta-v3-small
                 Converts query → declarative claim. Strict policy: confidence ≥ 0.85, verdict == SUPPORTS.
    │
    ▼
[generator]      Grounded answer using verified chunks (falls back to retrieved if NLI abstains).
                 Runs via LM Studio (qwen2.5-7b-instruct, local).
    │
    ▼
[reflector]      Self-critique: PASS → return answer. RETRY → back to generator (max 2 retries).
    │
    ▼
Route: ANSWER_WITH_CITATION or ABSTAIN
```

The agent abstains rather than hallucinating when it cannot produce a citation-verified answer. This is intentional — precision over recall.

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
| Total wall time (200 queries, no LLM) | 43.8s |

Eval runs retriever + NLI only (no LM Studio required). Run with:

```bash
python3 src/eval/eval_runner.py          # 200 queries
python3 src/eval/eval_runner.py --n 50   # smaller sample
```

Output → `outputs/evidence/<query_id>.json` + `outputs/eval_summary.json`

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph StateGraph (typed `AgentState`) |
| Tool interface | LangChain `@tool` decorator |
| LLM | LM Studio · `qwen2.5-7b-instruct` · OpenAI-compatible API |
| Dense retrieval | fastembed `BAAI/bge-small-en-v1.5` · 384-dim · Qdrant in-memory |
| Sparse retrieval | BM25Okapi (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| NLI verification | `cross-encoder/nli-deberta-v3-small` (HuggingFace) |
| Corpus | Wix/WixQA · 6,221 articles · MIT license |
| Cache | 3-part: `chunks.pkl` + `bm25.pkl` + `vectors.npy` (cold start ~15s, no re-embedding) |

---

## Project structure

```
pulseagent/
├── main.py                    Entry point — runs the agent graph interactively
├── config.py                  LM Studio URL, thresholds, top-K params
├── src/
│   ├── agent/
│   │   ├── graph.py           LangGraph StateGraph definition
│   │   ├── nodes.py           5 node functions: planner, retriever, nli_verifier, generator, reflector
│   │   └── state.py           AgentState TypedDict
│   ├── tools/
│   │   ├── retriever_tool.py  @tool: hybrid RRF retrieval (3-part cache)
│   │   └── nli_tool.py        @tool: NLI citation verification
│   ├── retrieval/             Bundled retrieval layer
│   │   ├── corpus.py          CorpusChunk dataclass
│   │   └── indexer.py         RetrievalIndex: BM25 + Qdrant + hybrid_search()
│   ├── citation/              Bundled NLI layer
│   │   └── entailment.py      NLICitationChecker (cross-encoder/nli-deberta-v3-small)
│   ├── corpus/                Bundled corpus adapter
│   │   └── wixqa_adapter.py   Loads WixQA from HuggingFace → CorpusChunk list
│   └── eval/
│       └── eval_runner.py     200-query evaluation harness (no LLM required)
└── .cache/                    Persistent index (created on first run, never re-embedded)
    ├── chunks.pkl
    ├── bm25.pkl
    └── vectors.npy
```

---

## Setup

**Requirements:**
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) with `qwen2.5-7b-instruct` loaded (for planner/generator/reflector nodes)
- LM Studio server running on `http://localhost:1234`

```bash
# Install dependencies
pip install langchain langchain-openai langgraph \
            fastembed qdrant-client rank-bm25 \
            sentence-transformers datasets numpy

# Run the agent
python3 main.py
```

**First run:** embeds 6,221 articles (~5-8 min one-time). Cache is saved to `.cache/` — all subsequent runs load in ~15s.

**Eval only (no LM Studio needed):**
```bash
python3 src/eval/eval_runner.py
```

---

## Key engineering decisions

**Why 3-part cache instead of pickling the full index?**
The `RetrievalIndex` holds a fastembed `TextEmbedding` object backed by an ONNX `InferenceSession`, which is not picklable. The solution: serialize chunks as plain dicts, BM25 separately, vectors as numpy. Rebuild Qdrant in-memory from saved vectors on load. Cold start from cache: ~15s vs ~8 min fresh.

**Why does the NLI verifier convert questions to declarative claims?**
NLI entailment models are trained on premise→hypothesis pairs where the hypothesis is a declarative statement. A question like "How do I add a payment method?" will almost never entail any chunk because questions don't state facts. Converting to `"This article provides information about: How do I add a payment method"` gives the model a falsifiable hypothesis it can actually verify.

**Why abstain at 44% rather than lower the threshold?**
At 0.85 NLI confidence the system answers only when it has strong citation support. Lowering the threshold increases answer rate but allows unverified chunks through — trading precision for recall. For a help center assistant, precision is the right call: a non-answer is better than a wrong answer.
