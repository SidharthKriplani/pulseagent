# PulseAgent — Full Audit Handoff for Opus 4.8

**Owner:** Sidharth Kriplani  
**Repo:** `/Users/ASUS/Documents/Professional/GitHub/beastmax (5)/pulseagent`  
**Task:** Do multiple adversarial audit passes across every file in this repo. Find every inconsistency between code, eval artifacts, README, defense PDF, and resume claims. Keep iterating until you find no new issues. Fix every issue you find — edit files directly. Leave a final audit report at `AUDIT_REPORT.md`.

---

## What This Project Is

PulseAgent is a citation-grounded multi-agent RAG system on 6,221 WixQA help articles. It answers natural-language questions about Wix products. The architecture:

```
LangGraph SupervisorAgent
  └─ planner_node → Send() fan-out → RetrievalAgent × N (parallel)
                                          ↓ (aggregate)
                                     VerifierAgent (NLI citation gate)
                                          ↓
                                     SynthesisAgent (generator + reflector)
```

Key components:
- **Hybrid retrieval:** BM25Okapi + BAAI/bge-small-en-v1.5 dense (fastembed ONNX) fused via RRF k=60
- **NLI gate:** cross-encoder/nli-deberta-v3-small, confidence ≥ 0.85, declarative claim conversion
- **Numeric Policy Verifier:** deterministic pre-check for hedged numeric claims; NUMERIC_MISMATCH overrides NLI
- **Citation Routing:** 7-rule priority chain → ANSWER_WITH_CITATION / ABSTAIN / BLOCK / CLARIFY / ESCALATE
- **Config:** `config.py` — all knobs via env vars; NLI_CONFIDENCE_THRESHOLD=0.85, TOP_K_RETRIEVE=10, TOP_K_VERIFY=3, MAX_RETRIES=2
- **Deploy:** GCP Cloud Run, FastAPI, X-API-Key auth, 10 req/min rate limit

---

## All Eval Artifacts (Ground Truth)

These are the actual numbers from the eval JSON files. Every claim in every file must trace to one of these.

### eval_summary.json (title-query eval, N=200, eval_runner.py)
- Queries: 200 (article titles as queries — self-retrieval)
- ANSWER_WITH_CITATION: 112 (56.0%)
- ABSTAIN: 88 (44.0%)
- ERROR: 0 (0.0%)
- mean_citation_precision: **0.26** (definition: len(verified_chunks)/TOP_K_VERIFY=3, averaged across ALL 200 queries including 88 ABSTAINs with precision=0)
- mean_latency_s: 0.231, p50: 0.146s, p95: 0.267s
- **Self-retrieval bias warning:** These queries are article titles. BM25 exact match dominates. Over-estimates recall on real user traffic.

### expert_eval_summary.json (naturalistic eval, N=200, expert_eval_runner.py, wixqa_expertwritten, ground-truth article IDs)
- **hybrid_rrf:** P@1=0.405, R@1=0.340, P@3=0.232, R@3=0.565, P@10=0.099, R@10=0.798, answer_rate=0.185, p95=0.336s
- **dense_only:** P@1=0.350, R@10=0.781, answer_rate=0.200, p95=0.335s
- **bm25_only:** P@1=0.280, R@10=0.573, answer_rate=0.155, p95=0.383s
- Headline: Hybrid +45% P@1 vs BM25, +39% R@10 vs BM25

### ablation_summary.json (N=50, seed=42, title-query ablation, ablation_runner.py)
- bm25_only: answer_rate=0.48, mean_citation_prec=0.193, p95=0.307s
- dense_only: answer_rate=0.42, mean_citation_prec=0.180, p95=0.205s
- hybrid_rrf: answer_rate=0.44, mean_citation_prec=0.187, p95=0.173s

### llm_eval_summary.json (N=37 answered queries, llm_eval_runner.py)
- qwen2.5-7b: mean_token_f1=0.388, p95_latency=12.515s, mean_answer_len=70 words, error=0
- llama-3.3-70b: mean_token_f1=0.331, p95_latency=0.900s, mean_answer_len=42 words, error=0
- Token F1 gap is a verbosity artifact: Qwen generates longer answers (70w vs 42w), more tokens hit ground-truth

---

## Known Issues — Confirmed Before This Handoff

These were caught in prior review. Audit them first; then find more.

### Issue 1: Send() fan-out — CONFIRMED IMPLEMENTED (defense doc was wrong)
**Files affected:** `docs/PulseAgent_Interview_Defense.pdf`, `CHECKPOINT.md`, `PRD.md`  
**The problem:** The defense PDF (Section 6E) says: *"No parallelism — planner calls sub-queries sequentially. V2: LangGraph Send() API for fan-out."* This is WRONG. `src/agents/supervisor.py` clearly implements `Send()`:
```python
from langgraph.types import Command, Send
...
return Command(
    update={...},
    goto=[Send("retrieval_agent", {"query": q, ...}) for q in sub_queries],
)
```
The README truth boundary table correctly marks Send() as `✅ Real`. The defense doc is wrong and needs to be corrected — remove the "No parallelism" claim and update Section 6E accordingly.  
**Audit action:** Read `src/agents/supervisor.py` fully. Verify Send() is wired correctly end-to-end. Check if `src/agent/` (singular, legacy) is distinct from `src/agents/` (plural, multi-agent). Determine which one `api.py` and `main.py` actually invoke. Update defense doc claims to match actual code.

### Issue 2: 75.9% NLI precision — definition mismatch, needs reconciliation
**Files affected:** `README.md` (badges, truth boundary table, key decisions section, resume-safe claim), `src/tools/nli_tool.py` (comment line 6)  
**The problem:** README shows `NLI_PRECISION: 75.9%` in shield badges and in the eval table as "Verified chunks / total candidate chunks". But `eval_summary.json` shows `mean_citation_precision: 0.26`. These appear contradictory.  
**How they could both be real:** 0.26 = mean of (verified/3) across ALL 200 queries including 88 ABSTAINs (precision=0). 75.9% might be computed differently — perhaps aggregated only over ANSWER_WITH_CITATION queries (N=112), or from an earlier eval run with different TOP_K_VERIFY, or from a different measurement (e.g. fraction of queries where top-1 chunk passed NLI).  
**Audit action:** Grep `eval_runner.py` for how citation_precision is calculated. Check if 75.9% is mathematically derivable from `eval_summary.json` under any consistent definition. If it is: add the exact definition everywhere it's cited. If it is NOT derivable from current eval artifacts: it's a stale number and must be removed from all public-facing files (README badges, truth boundary table, resume-safe claim). The `nli_tool.py` comment must be updated regardless.

### Issue 3: "No hallucination path exists" — overstatement
**Files affected:** `README.md` (intro blurb, truth boundary table)  
**The problem:** README says: *"There is no hallucination path."* and truth boundary says `✅ Real — Multi-agent LangGraph orchestration with Send() fan-out`. But: the generator always runs. On ABSTAIN, the system generates a best-effort answer from unverified chunks with a confidence note. The NLI gate gates CITATION, not generation. Caller must check the `route` field.  
**Safe alternative:** Replace "no hallucination path exists" with "no citation-backed answer path without NLI verification" or "ABSTAIN means no cited answer — best-effort answer with uncertainty flag is returned".

### Issue 4: defense PDF says "22-page" but is actually 29 pages
**Files affected:** `README.md` (Interview Defense section says "A 22-page system defense document")  
**Audit action:** Update the page count in README to reflect the actual PDF. Verify by checking `docs/PulseAgent_Interview_Defense.pdf`.

### Issue 5: Profile README (GitHub SidharthKriplani/SidharthKriplani repo) — NOT in this repo
**The problem:** The GitHub profile README (separate repo) contains:
- `LangGraph Send() fan-out` in the PulseAgent tagline — now confirmed real ✅ (no fix needed)
- No `75.9%` in the profile README (it uses different metrics) — check to confirm

This is a separate repo not in this folder. Flag if fixes are needed but don't edit it here.

---

## What to Audit — Complete Checklist

Work through these systematically. For each file, read it fully and check every claim against the eval artifacts above.

### Pass 1: Code vs Claims Audit

**`src/agents/supervisor.py`**
- Confirm Send() is implemented, wired correctly, and actually executes parallel fan-out
- Confirm aggregate_node runs after all parallel retrievals complete (via Annotated reducer)
- Confirm verifier and synthesis are separate compiled subgraphs

**`src/agents/retrieval_agent.py`**
- Confirm hybrid BM25 + dense + RRF is the actual retrieval implementation
- Confirm fetch_k=40 per leg (or whatever value is used), top-10 returned

**`src/agents/verifier_agent.py`**
- Confirm NLI gate uses cross-encoder/nli-deberta-v3-small at threshold 0.85
- Confirm declarative claim conversion is applied before NLI
- Confirm Numeric Policy Verifier runs as Rule 1 (before NLI)
- Confirm 7-rule CitationRoutingDecision exists and fires in correct priority order

**`src/agents/synthesis_agent.py`**
- Confirm generator always runs (even on ABSTAIN)
- Confirm reflector retry loop with MAX_RETRIES=2
- Confirm route field is set correctly in output

**`src/agent/` (legacy)**
- Understand what this is. Is it the old single-agent implementation?
- Does `api.py` invoke the legacy agent or the multi-agent supervisor?
- Does `main.py` invoke the multi-agent supervisor?

**`api.py`**
- Which agent does the `/query` endpoint actually call?
- Is it `run_supervisor()` from `src/agents/supervisor.py` or the legacy single-agent?
- Are rate limiting (10/min) and X-API-Key auth implemented?

**`src/eval/eval_runner.py`**
- How is citation_precision computed? Confirm definition matches 0.26 in eval_summary.json
- Is 75.9% derivable from this script under any code path?

**`src/eval/expert_eval_runner.py`**
- Confirm it uses wixqa_expertwritten split
- Confirm it evaluates against ground-truth article_ids (not just NLI pass rate)
- Confirm N=200

**`src/eval/ablation_runner.py`**
- Confirm N=50, seed=42
- Confirm three modes: bm25_only, dense_only, hybrid_rrf

**`src/eval/llm_eval_runner.py`**
- Confirm it loads from expert_eval_detail.json (the N=37 ANSWER_WITH_CITATION subset)
- Confirm token_f1 definition: bag-of-words F1

**`src/citation/entailment.py`**
- Confirm Numeric Policy Verifier exists and fires before NLI
- Confirm NUMERIC_MISMATCH overrides NLI verdict
- Confirm 7-rule CitationRoutingDecision priority chain
- Confirm LOW_CONFIDENCE_THRESHOLD=0.45 behavior

**`src/retrieval/indexer.py`**
- Confirm RRF implementation uses k=60
- Confirm fetch_k per leg

**`config.py`**
- Confirm all values: NLI_CONFIDENCE_THRESHOLD=0.85, TOP_K_RETRIEVE=10, TOP_K_VERIFY=3, MAX_RETRIES=2, CONTEXT_WINDOW=1200

**`docs/assets/pipeline_architecture.svg`**
- The SVG title bar says: "LangGraph StateGraph · Send() Fan-Out · NLI Contract Gate · GCP Cloud Run" — now confirmed correct ✅
- Check: does the VerifierAgent box mention Numeric Policy Verifier? If not, add it.
- Check: does it say "ABSTAIN — no hallucination"? If so, change to "ABSTAIN — citation evidence below threshold"
- Check the NLI precision badge value shown in SVG (if any)

### Pass 2: README Audit

Read `README.md` line by line. Check every claim against eval artifacts.

Key checks:
- Shield badge `NLI_PRECISION: 75.9%` — reconcile or remove
- Eval table row `NLI-verified citation precision: 75.9%` — reconcile or remove
- Eval table row `Mean retrieval + NLI latency: 0.218s` — cross-check eval_summary.json (it says mean_latency_s=0.231, p50=0.146) — inconsistency?
- Truth boundary table — every `✅ Real` claim must be verified against code
- "There is no hallucination path" — change to safe version
- "A 22-page system defense document" — update to correct page count
- Resume-safe claim block — 75.9% appears here; reconcile
- Portfolio table — 75.9% appears here; reconcile
- Key Engineering Decisions: "Why Send() for fan-out?" section — confirm code matches the description

### Pass 3: PRD.md Audit

`PRD.md` is the primary design document (9.6/10 per prior session). Check:
- Does it correctly describe Send() as implemented (not V2)?
- Section 8A (ablation inversion): is it accurate?
- Section 8D (LLM generation layer): numbers match llm_eval_summary.json?
- Section 12 (Evidence Ledger): all numbers match JSON artifacts?
- Section 22 (Defense Bank): are any Q&As contradicted by actual code?
- Section 27 scoring: is the 9.6/10 still warranted after corrections?

### Pass 4: CHECKPOINT.md Audit

Read `CHECKPOINT.md`. Does it contain any stale claims?  
Known issue: it may say "V2: Send() fan-out" — which is now wrong since Send() is implemented in V1.

### Pass 5: Defense PDF Assessment

`docs/PulseAgent_Interview_Defense.pdf` cannot be edited directly (it's a compiled PDF). But:
- Identify every claim in the PDF that is now incorrect
- Write a `docs/DEFENSE_ERRATA.md` file listing all corrections needed
- The most critical: Section 6E says "No parallelism — planner calls sub-queries sequentially. V2: LangGraph Send() API for fan-out." This is WRONG — must be in errata

### Pass 6: Cross-cutting consistency check

After fixing individual files, do a final pass:
- All metric appearances of "75.9%" — consistent definition everywhere or removed
- All appearances of "Send()" — all say "implemented" or "real", none say "V2"
- All appearances of "no hallucination" — all use safe language
- Page count of defense PDF — correct everywhere
- Mean latency claim (0.218s vs 0.231s from eval_summary.json) — investigate discrepancy
- "22-page" defense doc — updated to correct count

---

## File Map (all relevant files)

```
pulseagent/
├── api.py                          ← CHECK: which agent does /query call?
├── config.py                       ← CHECK: confirm all constants
├── main.py                         ← CHECK: which agent does CLI call?
├── README.md                       ← EDIT: fix 75.9%, hallucination claim, page count
├── PRD.md                          ← EDIT: fix Send() claims
├── CHECKPOINT.md                   ← EDIT: fix Send() V2 claim if present
├── docs/
│   ├── assets/pipeline_architecture.svg   ← CHECK/EDIT: Numeric Verifier, abstain language
│   └── PulseAgent_Interview_Defense.pdf   ← CANNOT EDIT: write DEFENSE_ERRATA.md instead
├── src/
│   ├── agents/
│   │   ├── supervisor.py           ← READ: Send() confirmed implemented here
│   │   ├── retrieval_agent.py      ← READ: confirm hybrid RRF
│   │   ├── verifier_agent.py       ← READ: confirm NLI gate + Numeric Verifier
│   │   ├── synthesis_agent.py      ← READ: confirm generator always runs
│   │   └── state.py                ← READ: confirm TypedDict structure
│   ├── agent/                      ← READ: understand legacy vs multi-agent
│   ├── citation/
│   │   └── entailment.py           ← READ: confirm Numeric Policy Verifier Rule 1
│   ├── retrieval/
│   │   └── indexer.py              ← READ: confirm RRF k=60
│   ├── tools/
│   │   ├── nli_tool.py             ← EDIT: remove/correct 75.9% comment
│   │   └── retriever_tool.py       ← READ: confirm hybrid fetch
│   └── eval/
│       ├── eval_runner.py          ← READ: find source of 75.9%
│       ├── expert_eval_runner.py   ← READ: confirm ground-truth eval
│       ├── ablation_runner.py      ← READ: confirm N=50, seed=42
│       └── llm_eval_runner.py      ← READ: confirm N=37, token F1 definition
└── outputs/
    ├── eval_summary.json           ← GROUND TRUTH: mean_citation_precision=0.26
    ├── expert_eval_summary.json    ← GROUND TRUTH: P@1=0.405, R@10=0.798
    ├── ablation_summary.json       ← GROUND TRUTH: hybrid p95=0.173s
    └── llm_eval_summary.json       ← GROUND TRUTH: Llama p95=0.900s
```

---

## Correct Numbers to Use (verified from JSON artifacts)

Use these when fixing incorrect claims in any file:

| Metric | Correct Value | Source | Definition |
|--------|--------------|--------|------------|
| Answer rate | 56.0% | eval_summary.json | ANSWER_WITH_CITATION queries / 200 total |
| Abstain rate | 44.0% | eval_summary.json | ABSTAIN queries / 200 total |
| Error rate | 0.0% | eval_summary.json | ERROR queries / 200 total |
| Mean citation precision (all queries) | 0.26 | eval_summary.json | mean(verified_chunks/3) across ALL 200 including 88 ABSTAINs |
| Mean citation precision (answered only) | ~0.46 | derived | mean(verified_chunks/3) across 112 ANSWER queries only |
| Mean latency (title eval) | 0.231s | eval_summary.json | full pipeline excluding LLM |
| P95 latency (title eval) | 0.267s | eval_summary.json | |
| Hybrid P@1 (expert eval) | 0.405 | expert_eval_summary.json | N=200, naturalistic questions |
| Hybrid R@10 (expert eval) | 0.798 | expert_eval_summary.json | N=200, naturalistic questions |
| Hybrid answer rate (expert eval) | 18.5% | expert_eval_summary.json | NLI gate at 0.85 on naturalistic phrasing |
| BM25 P@1 (expert eval) | 0.280 | expert_eval_summary.json | |
| BM25 R@10 (expert eval) | 0.573 | expert_eval_summary.json | |
| Qwen2.5-7B token F1 | 0.388 | llm_eval_summary.json | N=37, verbosity artifact |
| Llama-3.3-70b token F1 | 0.331 | llm_eval_summary.json | N=37, concise (42w vs 70w) |
| Llama P95 latency | 0.900s | llm_eval_summary.json | Groq LPU inference |
| Qwen P95 latency | 12.515s | llm_eval_summary.json | local CPU inference |

---

## Safe Language Rules (apply everywhere)

**Never write:** "no hallucination path exists"  
**Write instead:** "no citation-backed answer path without NLI verification" or "ABSTAIN — citation evidence below threshold"

**Never write:** "Send() fan-out — V2 target" or "Send() planned for V2"  
**Write instead:** "Send() fan-out implemented — planner dispatches sub-queries in parallel via Command/Send"

**Never write:** "75.9% NLI precision" without its exact definition  
**Write instead:** Either use the reconciled number with full definition, or use "Hybrid P@1=0.41, R@10=0.80 on expert eval (N=200, ground-truth IDs)"

**Never write:** "Qwen achieves higher quality than Llama"  
**Write instead:** "Qwen achieves higher token F1 (0.388 vs 0.331), a verbosity artifact from longer answers (70w vs 42w); Llama is 14x faster at P95"

---

## Deliverables Expected from Opus

1. **All edited files committed (or ready to commit)** — README.md, PRD.md, CHECKPOINT.md, nli_tool.py, pipeline_architecture.svg, any others with issues found
2. **`docs/DEFENSE_ERRATA.md`** — full list of corrections needed in the defense PDF (cannot edit PDF directly)
3. **`AUDIT_REPORT.md`** — final report listing:
   - Every issue found (including new ones beyond the 5 known above)
   - What was fixed in which file
   - What remains unfixed and why
   - Whether 75.9% is reconcilable or should be removed
   - Final verdict on whether all public-facing claims are now defensible

Do not stop after one pass. After fixing everything you find, re-read every file from scratch and look for anything you missed. Keep going until AUDIT_REPORT.md says "no new issues found on re-read."

---

## Session Context (what led here)

This project is a portfolio piece for Senior DS / MLE / Applied Scientist interviews. The owner (Sidharth Kriplani) is actively interviewing. A third-party review (ChatGPT) caught:
1. Send() overclaim in a diagram
2. 75.9% in a diagram
3. Numeric Policy Verifier missing from architecture diagram
4. "ABSTAIN — no hallucination" too strong

Items 1 and 3 turned out to be wrong in opposite directions (Send() is actually implemented; Numeric Verifier should be added to SVG). Item 2 (75.9%) needs mathematical reconciliation. Item 4 (hallucination language) needs wording fix.

The prior session produced a 29-page defense PDF. It contains the wrong claim about Send() (says V2) — this is the highest-priority fix because an interviewer can trivially disprove it by asking "show me the Send() call in supervisor.py" and the code will show it exists.
