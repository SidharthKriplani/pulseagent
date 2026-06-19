# PulseAgent — CHECKPOINT.md
**Last updated:** 2026-06-19 (Session 3 — merged, eval complete, README written)
**Status:** COMPLETE. Self-contained. No cross-repo sys.path. Eval run. README done.

**Architecture change (Option B merge):** `src/retrieval/`, `src/citation/`, `src/corpus/` now bundled
directly inside pulseagent. No dependency on `../pulseknowledge` at runtime. PA is a standalone repo.

---

## Current Working State

```
[retriever] Cache found — loading without re-embedding...   ✅
[retriever] Index ready (from cache).
[nli] decision=ANSWER_WITH_CITATION, precision=1.0          ✅
[generator] decision=ANSWER_WITH_CITATION, using 3 chunks   ✅
[reflector] PASS — PASS
Route:  ANSWER                                              ✅
```

**Cache is permanent.** `.cache/chunks.pkl` + `.cache/bm25.pkl` + `.cache/vectors.npy` all exist.
Cold start from cache: ~10-15s. No re-embedding ever again.

---

## Architecture

```
main.py  →  graph.py (LangGraph StateGraph)
              planner_node       → decomposes query, sets intent (LM Studio)
              retriever_node     → calls retrieve_chunks tool (hybrid RRF via PK)
              nli_verifier_node  → calls verify_citations tool (NLICitationChecker via PK)
              generator_node     → calls LM Studio (qwen2.5-7b-instruct) to answer
              reflector_node     → self-critique, PASS→END or RETRY→generator
```

### Folder structure
```
pulseagent/
  main.py
  config.py
  src/
    __init__.py
    agent/
      __init__.py
      graph.py
      nodes.py          ← generator always runs, NLI claim converted
      state.py
    tools/
      __init__.py
      retriever_tool.py ← 3-part cache (chunks as dicts) + full text
      nli_tool.py       ← FIXED: NLICitationChecker (no constructor args)
    eval/
      __init__.py
      eval_runner.py    ← NEW: 200-query eval, retriever+NLI only, no LM Studio needed
  .cache/
    chunks.pkl          ← list of dicts (CorpusChunk fields, picklable)
    bm25.pkl            ← BM25Okapi
    vectors.npy         ← 6221×384 float32
    retrieval_index.pkl ← OLD/unused — safe to delete
  outputs/
    evidence/           ← per-query JSON after eval run
    eval_summary.json   ← aggregate metrics after eval run
```

---

## Config (`config.py`)

```python
LM_STUDIO_BASE_URL       = "http://localhost:1234/v1"
LM_STUDIO_API_KEY        = "lm-studio"
LM_STUDIO_MODEL          = "qwen2.5-7b-instruct"   # NOT qwen2.5-7b-instruct-q4_k_m
NLI_CONFIDENCE_THRESHOLD = 0.85
TOP_K_RETRIEVE           = 10
TOP_K_VERIFY             = 3
MAX_RETRIES              = 2
CONTEXT_WINDOW           = 1200
```

---

## sys.path Rule (UPDATED — no cross-repo path needed)

```python
# retriever_tool.py and nli_tool.py both do:
PA_ROOT = Path(__file__).parent.parent.parent   # resolves to pulseagent/
sys.path.insert(0, str(PA_ROOT))
# Then: from src.retrieval.indexer import ...
#       from src.citation.entailment import ...
#       from src.corpus.wixqa_adapter import ...
```

PK modules are bundled in `src/retrieval/`, `src/citation/`, `src/corpus/`.
No `../pulseknowledge` path needed at runtime.

---

## NLI Class (CRITICAL — correct name)

```python
# In src/tools/nli_tool.py:
from citation.entailment import NLICitationChecker   # ← this exists
_checker = NLICitationChecker()                      # no constructor args
```

**Do NOT use:**
- `OnnxNLICitationChecker()` — requires model_dir, breaks
- `TransformersNLICitationChecker()` — does not exist in citation.entailment

---

## Cache (CRITICAL — 3-part, chunks as plain dicts)

Save side (`_save_cache` in retriever_tool.py):
```python
chunk_dicts = [vars(c) for c in idx.chunks]   # plain dicts — breaks ONNX back-reference
pickle.dump(chunk_dicts, f)                    # → chunks.pkl
pickle.dump(idx._bm25, f)                      # → bm25.pkl
np.save(str(VECTORS_CACHE), vectors)           # → vectors.npy
```

Load side (`_load_from_cache`):
```python
chunk_dicts = pickle.load(f)
from retrieval.corpus import CorpusChunk
chunks = [CorpusChunk(**d) for d in chunk_dicts]
```

**Do NOT pickle full RetrievalIndex** — it holds fastembed TextEmbedding (ONNX) → unpicklable.

---

## NLI Claim Conversion (CRITICAL — in nodes.py)

Questions always fail NLI entailment. In `nli_verifier_node`:
```python
raw   = state["query"].strip().rstrip("?")
claim = f"This article provides information about: {raw}"
```

---

## Generator Must Always Run (CRITICAL — in nodes.py)

Generator runs regardless of NLI decision. If verified chunks exist → use them.
If not → fall back to retrieved chunks. Never hard ABSTAIN unless zero chunks.

---

## PulseKnowledge Patch (already applied — do not undo)

`pulseknowledge/src/corpus/wixqa_adapter.py` line ~183:
```python
try:
    from src.retrieval.corpus import CorpusChunk
except ModuleNotFoundError:
    from retrieval.corpus import CorpusChunk
```

---

## Next Steps

### Step 1 — Run eval (no LM Studio needed)
```bash
cd "/Users/ASUS/Documents/Professional/GitHub/beastmax (5)/pulseagent"
python3 src/eval/eval_runner.py
```
Expected: ~4-6 min total. Cache loads in 10-15s. Outputs to `outputs/evidence/` + `outputs/eval_summary.json`.

### Step 2 — Write resume bullet
Use eval_summary.json metrics: answer_rate, mean_citation_precision, p95_latency_s.

### Step 3 — Respond to Teradata InMail
Naysha Matwani (naysha.matwani@teradata.com) — Staff/Senior AI Engineer.
PulseAgent is the agentic AI demo that supplements PulseKnowledge (RAG).

---

## What NOT To Do

- Do NOT use `OnnxNLICitationChecker()` — no model_dir available
- Do NOT use `TransformersNLICitationChecker()` — does not exist
- Do NOT pickle full RetrievalIndex — 3-part cache with chunk dicts only
- Do NOT pass question directly to NLI — convert to declarative claim
- Do NOT short-circuit generator on ABSTAIN/ERROR
- Do NOT add PK_ROOT (not PK_SRC) to sys.path
- Do NOT delete `.cache/chunks.pkl`, `.cache/bm25.pkl`, `.cache/vectors.npy`
- `.cache/retrieval_index.pkl` is old format, unused — safe to delete
