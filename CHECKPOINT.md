# PulseAgent — CHECKPOINT.md
**Last updated:** 2026-06-24 (Session 6 — full eval suite complete, defense PDF built, audit handoff written)  
**Status:** GOLD. All 4 evals run. PRD at 9.6/10. 29-page defense PDF in docs/. Audit handoff at AUDIT_HANDOFF.md.

---

## Current Working State

All pipeline runs correctly end-to-end:
```
[supervisor.planner] intent='...' | 3 sub-queries
[supervisor.retrieval × 3] parallel via Send() fan-out
[supervisor.aggregate] N raw → 10 unique chunks
[supervisor.verifier] ANSWER_WITH_CITATION / ABSTAIN
[supervisor.synthesis] final_answer generated
Route: ANSWER | ABSTAIN
```

Cache is permanent: `.cache/chunks.pkl` + `.cache/bm25.pkl` + `.cache/vectors.npy`  
Cold start from cache: ~10–15s. No re-embedding ever again.

---

## Architecture (CURRENT — Multi-Agent, NOT the old single-agent)

There are TWO implementations in this repo:

### PRIMARY: Multi-Agent Supervisor (`src/agents/`) ← what api.py and main.py use
```
SupervisorAgent (src/agents/supervisor.py)
  planner_node           → LLM decomposes query → intent + up to 3 sub-queries
                           Returns Command(goto=[Send("retrieval_agent", {q}) for q in sub_queries])
                           Send() is IMPLEMENTED and REAL — parallel fan-out per sub-query
  retrieval_agent × N    → independently compiled subgraph (src/agents/retrieval_agent.py)
                           hybrid BM25Okapi + bge-small-en-v1.5 dense via fastembed ONNX + RRF k=60
                           results accumulated via Annotated[List[dict], operator.add] reducer
  aggregate_node         → deduplicate by article_id, re-rank by rrf_score, top-10
  verifier_wrapper_node  → runs compiled VerifierAgent subgraph (src/agents/verifier_agent.py)
                           Numeric Policy Verifier (Rule 1) → Cross-Encoder NLI → CitationRoutingDecision
  synthesis_wrapper_node → runs compiled SynthesisAgent subgraph (src/agents/synthesis_agent.py)
                           generator always runs (even on ABSTAIN, uses retrieved chunks)
                           reflector retry loop MAX_RETRIES=2
```

### LEGACY: Single-Agent (`src/agent/`) ← old implementation, --legacy flag only
```
LangGraph StateGraph (src/agent/graph.py):
  planner_node → retriever_node → nli_verifier_node → generator_node → reflector_node
  Sequential sub-query retrieval (no Send() parallelism)
  NOT what runs in production
```

**Critical:** The old CHECKPOINT described the legacy single-agent. That is no longer the primary path.

---

## Config (`config.py`)

```python
LLM_BASE_URL             = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")  # override for Groq
LLM_API_KEY              = os.getenv("LLM_API_KEY",  "lm-studio")
LLM_MODEL                = os.getenv("LLM_MODEL",    "qwen2.5-7b-instruct")
# Cloud: LLM_BASE_URL=https://api.groq.com/openai/v1, LLM_MODEL=llama-3.3-70b-versatile

NLI_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.45   # any verdict below this → NOT_ENOUGH_INFO regardless
TOP_K_RETRIEVE           = 10
TOP_K_VERIFY             = 3
MAX_RETRIES              = 2
CONTEXT_WINDOW           = 1200
NLI_MODEL_NAME           = "cross-encoder/nli-deberta-v3-small"
```

LLM backend switching: set env vars only. Zero code change between LM Studio local and Groq cloud.

---

## All Eval Results (Ground Truth — do not modify without re-running)

### eval_summary.json (eval_runner.py, N=200, title-query self-retrieval)
- answer_rate: 0.56 (112/200)
- abstain_rate: 0.44 (88/200)
- error_rate: 0.0
- mean_citation_precision: **0.26** (= mean of verified_chunks/TOP_K_VERIFY=3 across ALL 200 queries incl. 88 ABSTAINs)
- citation precision answered-only: ~0.46 (= 0.26×200/112)
- mean_latency_s: 0.231, p50: 0.146s, p95: 0.267s
- **SELF-RETRIEVAL BIAS:** queries = article titles; BM25 dominates; over-estimates recall vs real traffic

### expert_eval_summary.json (expert_eval_runner.py, N=200, wixqa_expertwritten, ground-truth article_ids)
- hybrid_rrf: P@1=0.405, R@1=0.340, P@3=0.232, R@3=0.565, P@10=0.099, R@10=0.798
- dense_only: P@1=0.350, R@10=0.781
- bm25_only:  P@1=0.280, R@10=0.573
- hybrid vs BM25: +45% P@1, +39% R@10
- hybrid answer_rate: 0.185 (NLI at 0.85 strict on naturalistic phrasing)
- hybrid p95: 0.336s

### ablation_summary.json (ablation_runner.py, N=50, seed=42, title-query)
- bm25_only:  answer_rate=0.48, p95=0.307s
- dense_only: answer_rate=0.42, p95=0.205s
- hybrid_rrf: answer_rate=0.44, p95=0.173s
- **ABLATION INVERSION:** BM25 wins answer rate on title queries; inverts on expert eval. Title queries bias toward BM25 exact match.

### llm_eval_summary.json (llm_eval_runner.py, N=37 answered queries from expert_eval_detail.json)
- qwen2.5-7b (LM Studio): token_f1=0.388, p95=12.515s, mean_answer=70 words
- llama-3.3-70b (Groq):   token_f1=0.331, p95=0.900s,  mean_answer=42 words
- **VERBOSITY ARTIFACT:** Qwen higher token F1 because longer answers, not better quality. Llama 14x faster at P95.
- Production backend: Llama-3.3-70b via Groq

---

## Key Design Decisions (with empirical grounding)

| Decision | Reason | Evidence |
|----------|--------|----------|
| Hybrid BM25 + dense RRF over dense-only | BM25 catches exact identifiers; dense catches paraphrase | P@1 0.405 hybrid vs 0.280 BM25 (+45%) on expert eval |
| RRF k=60 over linear fusion | Scale-free; no labeled calibration data needed | Cormack et al. 2009; p95 0.173s |
| Cross-encoder NLI over bi-encoder similarity | Entailment != similarity; contradiction detection needed | Claim conversion: answer rate 20% → 56% |
| Declarative claim conversion | Questions fail NLI entailment (out-of-distribution for SNLI) | nodes.py: "This article provides information about: {query}" |
| Numeric Policy Verifier as Rule 1 | NLI cannot do arithmetic; 30 days ≠ 60 days | entailment.py NUMERIC_MISMATCH overrides NLI |
| Send() parallel fan-out | 3 sub-queries in parallel reduces latency vs sequential | supervisor.py Command/Send implemented |
| Llama-3.3-70b via Groq for production | 14x faster at P95 (0.9s vs 12.5s); token F1 gap is verbosity artifact | llm_eval_summary.json |
| 3-part cache not full pickle | fastembed TextEmbedding holds ONNX InferenceSession (C++, unpicklable) | Silent write, RuntimeError on read; fixed |

---

## Known Issues / Pending Work (as of this session)

### Confirmed Issues Needing Fixes (AUDIT_HANDOFF.md has full details)

**Issue 1 — Defense PDF wrong about Send():**  
`docs/PulseAgent_Interview_Defense.pdf` Section 6E says "No parallelism — planner calls sub-queries sequentially. V2: LangGraph Send() API for fan-out." THIS IS WRONG. Send() is implemented and real. Cannot edit PDF; `docs/DEFENSE_ERRATA.md` needs to be written (by Opus audit).

**Issue 2 — 75.9% NLI precision needs reconciliation:**  
`README.md` badge + eval table + truth boundary + resume-safe claim + `nli_tool.py` comment all cite 75.9%.  
`eval_summary.json` shows mean_citation_precision=0.26.  
These are different calculations. 75.9% may be defensible under a specific definition (answered-queries only, or different TOP_K_VERIFY), or it may be a stale number from an old run. Must be mathematically resolved by reading `eval_runner.py`. If not derivable from current JSON, delete from all files.

**Issue 3 — "No hallucination path exists" too strong:**  
README intro says "There is no hallucination path." Generator always runs. Caller must check `route` field. Change to: "no citation-backed answer path without NLI verification."

**Issue 4 — README says "22-page" defense PDF; actual PDF is 29 pages.**

**Issue 5 — pipeline_architecture.svg missing Numeric Policy Verifier node; possibly says "ABSTAIN — no hallucination."**

### Manual Actions Still Required (Sidharth must do these, not Claude)
- GitHub: pin swap — uncheck riskframe/devpulse/pulserank; check pulseagent/pulseguard/pulsediscover
- GitHub: make 4 platform repos private (riskframe_platform, devpulse_platform, metasignal_platform, pulserank_platform)
- GitHub Profile README: copy updated metrics from `outputs/profile_README.md` to SidharthKriplani/SidharthKriplani repo

---

## Documents Produced This Session

| File | What It Is |
|------|-----------|
| `PRD.md` | Primary design/eval document — RiskFrame 9.6/10, 27 sections |
| `docs/PulseAgent_Interview_Defense.pdf` | 29-page interview defense (Gold Final v2) |
| `AUDIT_HANDOFF.md` | Full audit brief for Opus 4.8 — multiple passes until clean |
| `outputs/ablation_summary.json` + `outputs/ablation_detail.json` | Retrieval ablation N=50 results |
| `outputs/expert_eval_summary.json` + `outputs/expert_eval_detail.json` | Expert eval N=200 results |
| `outputs/llm_eval_summary.json` + `outputs/llm_eval_detail.json` | LLM comparison results |
| `src/eval/ablation_runner.py` | Ablation script |
| `src/eval/expert_eval_runner.py` | Expert eval script (ground-truth article IDs) |
| `src/eval/llm_eval_runner.py` | LLM generation comparison script |

---

## Session History (for lineage)

| Session | Key Outcome |
|---------|------------|
| 1 | Initial build — single-agent LangGraph, NLI gate, 3-part cache, LM Studio integration |
| 2 | Bug fixes — ONNX unpicklable (3-part cache), NLI claim conversion, eval_runner.py built |
| 3 | Eval run (N=200), README written, eval_summary.json produced, GCP Cloud Run |
| 4 | Multi-agent refactor — src/agents/ with supervisor, Send() fan-out, compiled subgraphs |
| 5 | PRD.md written to 9.3/10; ablation runner + expert eval runner built and run; LLM eval runner built and run; PRD updated to 9.6/10; defense bank updated |
| 6 | Defense PDF built (29 pages); ChatGPT review caught 4 issues; audit handoff written; CHECKPOINT rewritten |

---

## Safe Claims (for interviews / resume — verified against JSON artifacts)

**Safe now:**
- "Multi-agent RAG on 6,221 WixQA articles — LangGraph Send() fan-out across independently compiled subgraphs"
- "Hybrid BM25 + dense RRF: P@1=0.41, R@10=0.80 on 200 expert-written questions vs ground-truth IDs"
- "NLI citation gate (cross-encoder DeBERTa-v3-small, ≥0.85 confidence): 56% answer, 44% principled abstain, 0% error"
- "Numeric Policy Verifier: NUMERIC_MISMATCH overrides NLI — deterministic pre-check for hedged numeric claims"
- "Llama-3.3-70b via Groq: P95 0.9s; Qwen2.5-7B local: P95 12.5s — 14x gap; token F1 gap is verbosity artifact"
- "Ablation inversion: BM25 wins on title-query self-retrieval (48%), hybrid wins on naturalistic expert questions (P@1 +45%)"

**Do NOT say:**
- "75.9% NLI precision" without its exact definition — not yet reconciled with eval_summary.json
- "No hallucination path exists" — too strong; generator always runs; caller checks route field
- "Qwen2.5-7B achieves higher quality than Llama" — verbosity artifact, not quality
- Any RAGAS claim — not yet run

---

## NLI Architecture (CRITICAL — do not break)

```python
# Claim conversion (questions fail NLI entailment — out-of-distribution for SNLI)
raw   = state["query"].strip().rstrip("?")
claim = f"This article provides information about: {raw}"

# Numeric Policy Verifier runs FIRST (Rule 1 in CitationRoutingDecision)
# NUMERIC_MISMATCH → BLOCK regardless of NLI verdict
# Verdicts: NUMERIC_MATCH | NUMERIC_MISMATCH | NUMERIC_HEDGED | NOT_PRESENT | UNCERTAIN

# NLI gate
# P(SUPPORTS | claim, chunk) >= 0.85 → ANSWER_WITH_CITATION
# LOW_CONFIDENCE_THRESHOLD = 0.45 → any verdict below this → NOT_ENOUGH_INFO

# 7-rule CitationRoutingDecision priority chain (fires on first match):
# Rule 1: NUMERIC_MISMATCH → BLOCK
# Rule 2: any SUPPORTS >= 0.85 → ANSWER_WITH_CITATION
# Rule 3: all NOT_ENOUGH_INFO or any conf < 0.45 → ABSTAIN
# Rule 4: any CONTRADICTS → BLOCK
# Rule 5: NUMERIC_HEDGED → ESCALATE (requires_human_review=True)
# Rule 6: citation_precision < 0.2 → ABSTAIN
# Rule 7: default → ABSTAIN
```

---

## Cache Architecture (CRITICAL — do not break)

```python
# WHY 3-part: fastembed TextEmbedding holds ONNX InferenceSession (C++ object, unpicklable)
# pickle.dump succeeds silently; RuntimeError fires at access time on load

# Save:
chunk_dicts = [vars(c) for c in idx.chunks]   # plain dicts only
pickle.dump(chunk_dicts, f)                     # chunks.pkl
pickle.dump(idx._bm25, f)                       # bm25.pkl (BM25Okapi is pure Python)
np.save(str(VECTORS_CACHE), vectors)            # vectors.npy

# Load:
chunk_dicts = pickle.load(f)
chunks      = [CorpusChunk(**d) for d in chunk_dicts]
bm25_index  = pickle.load(f)
vectors     = np.load(str(VECTORS_CACHE))
# Reconstruct Qdrant in-memory from saved vectors — no re-embedding
```

---

## What NOT To Do

- Do NOT use OnnxNLICitationChecker() — requires model_dir, breaks
- Do NOT pass question directly to NLI — convert to declarative claim first
- Do NOT short-circuit generator on ABSTAIN — generator always runs
- Do NOT pickle full RetrievalIndex — 3-part cache only
- Do NOT delete `.cache/chunks.pkl`, `.cache/bm25.pkl`, `.cache/vectors.npy`
- Do NOT say "Send() is V2" — it is implemented in src/agents/supervisor.py
- Do NOT say "75.9% NLI precision" without a reconciled definition
- Do NOT say "no hallucination path exists" — say "no citation-backed answer without NLI verification"
- Do NOT edit the defense PDF directly — it's compiled; write docs/DEFENSE_ERRATA.md instead
- Do NOT add PK_ROOT to sys.path — PK modules are bundled in src/retrieval/, src/citation/, src/corpus/

---

## Resuming This Work

If you need to continue from here in a new session, the context you need is:
1. Read this CHECKPOINT.md in full
2. Read AUDIT_HANDOFF.md — the next big task is Opus 4.8 doing a full audit
3. Read PRD.md Section 27 (scoring) and Section 12 (Evidence Ledger) for current state
4. Key eval numbers are in `outputs/` JSON files — those are the ground truth
5. The defense PDF at `docs/PulseAgent_Interview_Defense.pdf` has a known error (Section 6E re: Send())
6. The audit is NOT done yet — AUDIT_REPORT.md and docs/DEFENSE_ERRATA.md do not exist yet
