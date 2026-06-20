"""
llm_eval_runner.py — LLM generation quality comparison: Qwen2.5-7B vs Llama-3.3-70b

Compares two LLM backends on generation quality using answered queries from expert_eval.
For each ANSWER_WITH_CITATION query (hybrid_rrf mode from expert_eval_detail.json):
  1. Re-run hybrid retrieval → NLI gate → get verified chunks
  2. Call each LLM backend with the same prompt + context
  3. Compute Token F1 of generated answer vs expert ground-truth answer field

Token F1: bag-of-words overlap between generated and expert answer (same as SQuAD metric).
Not perfect, but no LLM judge needed — deterministic and reproducible.

Backends compared:
  - qwen2.5-7b-instruct  (LM Studio, localhost:1234, local/free)
  - llama-3.3-70b-versatile (Groq, cloud)  — requires GROQ_API_KEY env var

Usage:
    python3 src/eval/llm_eval_runner.py            # all answered queries
    python3 src/eval/llm_eval_runner.py --n 20     # first N answered queries
    python3 src/eval/llm_eval_runner.py --qwen-only  # skip Groq (no API key)

Output:
    outputs/llm_eval_summary.json
    outputs/llm_eval_detail.json
"""

from __future__ import annotations
import sys, json, time, pickle, argparse, re, os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PA_ROOT))

from config import TOP_K_RETRIEVE, TOP_K_VERIFY, NLI_CONFIDENCE_THRESHOLD, CONTEXT_WINDOW

CACHE_DIR          = PA_ROOT / ".cache"
CHUNKS_CACHE       = CACHE_DIR / "chunks.pkl"
BM25_CACHE         = CACHE_DIR / "bm25.pkl"
VECTORS_CACHE      = CACHE_DIR / "vectors.npy"
OUTPUTS_DIR        = PA_ROOT / "outputs"
EXPERT_DETAIL_PATH = OUTPUTS_DIR / "expert_eval_detail.json"
SUMMARY_PATH       = OUTPUTS_DIR / "llm_eval_summary.json"
DETAIL_PATH        = OUTPUTS_DIR / "llm_eval_detail.json"

HF_DATASET    = "Wix/WixQA"
EW_CONFIG     = "wixqa_expertwritten"
CHUNK_ID_PFX  = "wixqa_"

BACKEND_QWEN = {
    "name":     "qwen2.5-7b (LM Studio)",
    "base_url": "http://localhost:1234/v1",
    "api_key":  "lm-studio",
    "model":    "qwen2.5-7b-instruct",
    "local":    True,
}
BACKEND_GROQ = {
    "name":     "llama-3.3-70b (Groq)",
    "base_url": "https://api.groq.com/openai/v1",
    "api_key":  os.getenv("GROQ_API_KEY", ""),
    "model":    "llama-3.3-70b-versatile",
    "local":    False,
}

SYSTEM_PROMPT = (
    "You are a precise knowledge assistant for Wix help articles. "
    "Answer ONLY using the provided sources. "
    "If the sources are insufficient, say so explicitly. "
    "Be concise and cite sources as [Source N]. No hallucination."
)


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-words token F1 (SQuAD-style). Proxy for answer quality vs expert answer."""
    p_toks = set(_tok(pred))
    g_toks = set(_tok(gold))
    if not p_toks or not g_toks:
        return 0.0
    common    = p_toks & g_toks
    precision = len(common) / len(p_toks)
    recall    = len(common) / len(g_toks)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ── Cache + embedding model ────────────────────────────────────────────────────

def load_cache():
    print("[llm_eval] Loading retrieval cache...")
    with open(CHUNKS_CACHE, "rb") as f:
        chunk_dicts = pickle.load(f)
    with open(BM25_CACHE, "rb") as f:
        bm25 = pickle.load(f)
    vectors = np.load(str(VECTORS_CACHE))
    print(f"[llm_eval] {len(chunk_dicts)} chunks, vectors {vectors.shape}")
    return chunk_dicts, bm25, vectors


def load_emb_model():
    from fastembed import TextEmbedding
    print("[llm_eval] Loading embedding model (BAAI/bge-small-en-v1.5)...")
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed_query(query: str, emb_model) -> np.ndarray:
    return np.array(list(emb_model.embed([query]))[0], dtype=np.float32)


# ── Hybrid retrieval (mirrors expert_eval_runner.py) ──────────────────────────

def retrieve_hybrid(query: str, bm25, vectors: np.ndarray, chunk_dicts: list[dict],
                    emb_model, k: int = TOP_K_RETRIEVE, rrf_k: int = 60) -> list[dict]:
    fetch_k = min(k * 4, len(chunk_dicts))

    # BM25 leg
    b_scores = bm25.get_scores(_tok(query))
    b_idxs   = sorted(range(len(b_scores)), key=lambda i: b_scores[i], reverse=True)[:fetch_k]
    b_ranks  = {chunk_dicts[i].get("chunk_id", f"chunk_{i}"): (r + 1)
                for r, i in enumerate(b_idxs)}
    b_text   = {chunk_dicts[i].get("chunk_id", ""): chunk_dicts[i].get("text", "")
                for i in b_idxs}

    # Dense leg
    q_vec  = embed_query(query, emb_model)
    norms  = np.linalg.norm(vectors, axis=1)
    q_norm = np.linalg.norm(q_vec)
    d_ranks, d_text = {}, {}
    if q_norm >= 1e-9:
        sims   = (vectors @ q_vec) / (norms * q_norm + 1e-9)
        d_idxs = np.argsort(sims)[::-1][:fetch_k]
        for r, i in enumerate(d_idxs):
            cid = chunk_dicts[int(i)].get("chunk_id", f"chunk_{i}")
            d_ranks[cid] = r + 1
            d_text[cid]  = chunk_dicts[int(i)].get("text", "")

    # RRF fusion
    all_ids = set(b_ranks) | set(d_ranks)
    def rrf(cid): return (1/(rrf_k + b_ranks.get(cid, fetch_k+1)) +
                          1/(rrf_k + d_ranks.get(cid, fetch_k+1)))
    ranked = sorted(all_ids, key=rrf, reverse=True)[:k]
    all_text = {**b_text, **d_text}
    return [{"article_id": cid, "text": all_text.get(cid, "")[:CONTEXT_WINDOW],
             "rrf_score": round(rrf(cid), 6), "rank": r + 1}
            for r, cid in enumerate(ranked)]


# ── NLI gate ──────────────────────────────────────────────────────────────────

_checker = None

def run_nli(query: str, chunks: list[dict]) -> dict:
    global _checker
    try:
        if _checker is None:
            from src.citation.entailment import NLICitationChecker
            _checker = NLICitationChecker()
        raw   = query.strip().rstrip("?")
        claim = f"This article provides information about: {raw}"
        verified = [
            c for c in chunks[:TOP_K_VERIFY]
            if (r := _checker.check(claim, c.get("text", ""), c.get("article_id", "")))
            and r.verdict == "SUPPORTS" and r.confidence >= NLI_CONFIDENCE_THRESHOLD
        ]
        return {"decision": "ANSWER_WITH_CITATION" if verified else "ABSTAIN",
                "verified": verified}
    except Exception as e:
        return {"decision": "ERROR", "verified": [], "error": str(e)}


# ── LLM generator ─────────────────────────────────────────────────────────────

def generate(question: str, chunks: list[dict], backend: dict) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    context = "\n\n".join(
        f"[Source {i+1} | ID: {c['article_id']}]\n{c.get('text', '')[:CONTEXT_WINDOW]}"
        for i, c in enumerate(chunks)
    )
    prompt = f"QUERY: {question}\n\nSOURCES:\n{context}\n\nAnswer concisely. Cite [Source N]."

    t0 = time.time()
    try:
        llm = ChatOpenAI(
            base_url=backend["base_url"],
            api_key=backend["api_key"],
            model=backend["model"],
            temperature=0.2,
            timeout=60,
        )
        resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                           HumanMessage(content=prompt)])
        answer  = resp.content.strip()
        latency = round(time.time() - t0, 3)
        return {"answer": answer, "latency_s": latency,
                "n_tokens_approx": len(answer.split()), "error": None}
    except Exception as e:
        return {"answer": "", "latency_s": round(time.time() - t0, 3),
                "n_tokens_approx": 0, "error": str(e)}


# ── Load answered queries from expert_eval_detail.json ─────────────────────────

def load_answered_queries(n: int | None) -> list[dict]:
    """Pull ANSWER_WITH_CITATION rows from the hybrid_rrf mode in expert_eval_detail.json."""
    with open(EXPERT_DETAIL_PATH) as f:
        detail = json.load(f)
    hybrid_rows = detail.get("hybrid_rrf", [])
    answered = [r for r in hybrid_rows if r.get("route") == "ANSWER_WITH_CITATION"]
    if n:
        answered = answered[:n]
    print(f"[llm_eval] {len(answered)} ANSWER_WITH_CITATION queries from hybrid eval")
    return answered


# ── Load ground-truth answers from HuggingFace ────────────────────────────────

def load_gt_answers(query_ids: set[str]) -> dict[str, str]:
    """Load wixqa_expertwritten and return {query_id: expert_answer}."""
    from datasets import load_dataset
    print(f"[llm_eval] Loading ground-truth answers from {HF_DATASET}/{EW_CONFIG}...")
    ds = load_dataset(HF_DATASET, EW_CONFIG, split="train")
    gt = {}
    for i, row in enumerate(ds):
        qid = f"expert_{i:04d}"
        if qid in query_ids:
            gt[qid] = row.get("answer", "").strip()
    print(f"[llm_eval] Loaded {len(gt)} ground-truth answers")
    return gt


# ── Aggregate ──────────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    f1s     = [r["token_f1"]  for r in records if r["token_f1"] is not None]
    lats    = sorted(r["latency_s"] for r in records if r["latency_s"] is not None)
    errors  = sum(1 for r in records if r.get("error"))
    lengths = [r["n_tokens_approx"] for r in records if r["n_tokens_approx"]]
    return {
        "n":              len(records),
        "error_count":    errors,
        "mean_token_f1":  round(sum(f1s) / len(f1s), 4) if f1s else None,
        "p50_token_f1":   sorted(f1s)[len(f1s)//2]      if f1s else None,
        "mean_latency_s": round(sum(lats) / len(lats), 3)        if lats else None,
        "p50_latency_s":  lats[len(lats)//2]                     if lats else None,
        "p95_latency_s":  lats[int(0.95*len(lats))]              if lats else None,
        "mean_answer_len_words": round(sum(lengths)/len(lengths)) if lengths else None,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_llm_eval(n: int | None = None, qwen_only: bool = False) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Backends to run
    backends = [BACKEND_QWEN]
    if not qwen_only:
        if BACKEND_GROQ["api_key"]:
            backends.append(BACKEND_GROQ)
        else:
            print("[llm_eval] GROQ_API_KEY not set — skipping Groq backend. "
                  "Set env var or use --qwen-only to suppress this warning.")

    # Load retrieval infrastructure
    chunk_dicts, bm25, vectors = load_cache()
    emb_model = load_emb_model()

    # Load answered queries from prior expert eval
    answered = load_answered_queries(n)
    if not answered:
        print("[llm_eval] No answered queries found in expert_eval_detail.json. "
              "Run expert_eval_runner.py first.")
        return {}

    # Load ground-truth answers
    query_ids = {r["query_id"] for r in answered}
    gt_answers = load_gt_answers(query_ids)

    print(f"\n[llm_eval] {len(answered)} queries × {len(backends)} backends")
    print("=" * 64)

    all_records: dict[str, list[dict]] = {b["name"]: [] for b in backends}

    for i, qa_row in enumerate(answered):
        qid      = qa_row["query_id"]
        question = qa_row["question"]
        gt_ans   = gt_answers.get(qid, "")

        # Re-run hybrid retrieval (cache is loaded — fast)
        chunks   = retrieve_hybrid(question, bm25, vectors, chunk_dicts, emb_model)
        nli      = run_nli(question, chunks)
        verified = nli.get("verified", []) or chunks[:TOP_K_VERIFY]

        for backend in backends:
            gen = generate(question, verified, backend)
            f1  = token_f1(gen["answer"], gt_ans) if gt_ans and gen["answer"] else None

            record = {
                "query_id":         qid,
                "question":         question,
                "gt_answer_len":    len(gt_ans.split()),
                "backend":          backend["name"],
                "answer":           gen["answer"][:500],  # truncate for JSON
                "token_f1":         f1,
                "latency_s":        gen["latency_s"],
                "n_tokens_approx":  gen["n_tokens_approx"],
                "nli_decision":     nli["decision"],
                "error":            gen.get("error"),
            }
            all_records[backend["name"]].append(record)

        if (i + 1) % 5 == 0 or (i + 1) == len(answered):
            print(f"[llm_eval] {i+1}/{len(answered)} done")

    # Aggregate + print
    summary = {
        "eval_date":       datetime.now(timezone.utc).isoformat(),
        "n_queries":       len(answered),
        "nli_threshold":   NLI_CONFIDENCE_THRESHOLD,
        "top_k_verify":    TOP_K_VERIFY,
        "metric":          "token_f1 (bag-of-words F1 vs expert ground-truth answer)",
        "backends":        {b["name"]: aggregate(all_records[b["name"]]) for b in backends},
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    DETAIL_PATH.write_text(json.dumps(all_records, indent=2))

    # Print comparison table
    print()
    print("=" * 72)
    print("  LLM GENERATION QUALITY — TOKEN F1 vs EXPERT GROUND-TRUTH ANSWER")
    print("=" * 72)
    bnames = [b["name"] for b in backends]
    col_w  = 22
    print(f"  {'Metric':<30}" + "".join(f"{n:>{col_w}}" for n in bnames))
    print("  " + "-" * (30 + col_w * len(bnames)))

    def prow(label, key, fmt="{:.4f}"):
        vals = [summary["backends"][n].get(key) for n in bnames]
        cells = [fmt.format(v) if v is not None else "  N/A" for v in vals]
        print(f"  {label:<30}" + "".join(f"{c:>{col_w}}" for c in cells))

    prow("Mean Token F1",        "mean_token_f1")
    prow("P50 Token F1",         "p50_token_f1")
    prow("Mean latency (s)",     "mean_latency_s",     fmt="{:.3f}")
    prow("P50 latency (s)",      "p50_latency_s",      fmt="{:.3f}")
    prow("P95 latency (s)",      "p95_latency_s",      fmt="{:.3f}")
    prow("Mean answer len (w)",  "mean_answer_len_words", fmt="{:.0f}")
    prow("Error count",          "error_count",        fmt="{:.0f}")

    print("=" * 72)
    print(f"  N={len(answered)}, metric=token_f1 (SQuAD-style bag-of-words overlap)")
    print(f"  Summary → {SUMMARY_PATH}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Number of answered queries to evaluate (default: all)")
    parser.add_argument("--qwen-only", action="store_true",
                        help="Run Qwen2.5-7B only (skip Groq)")
    args = parser.parse_args()
    run_llm_eval(n=args.n, qwen_only=args.qwen_only)
