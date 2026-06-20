"""
expert_eval_runner.py — Eval on real expert-written questions with ground-truth precision@K

Uses wixqa_expertwritten split: 200 expert-authored questions with ground-truth article_ids.
Unlike eval_runner.py (which uses article titles), this uses naturalistic user questions
and measures precision@1, precision@3, precision@10 against known ground-truth article IDs.

This is the honest eval the PRD calls for under "stronger eval" — closes the self-retrieval
bias of the title-based eval in eval_runner.py.

Usage:
    python3 src/eval/expert_eval_runner.py
    python3 src/eval/expert_eval_runner.py --n 50

Output:
    outputs/expert_eval_summary.json   — precision@K + answer_rate across all 3 modes
    outputs/expert_eval_detail.json    — per-query evidence
"""

from __future__ import annotations
import sys, json, time, pickle, argparse, re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PA_ROOT))

from config import TOP_K_RETRIEVE, TOP_K_VERIFY, NLI_CONFIDENCE_THRESHOLD

CACHE_DIR    = PA_ROOT / ".cache"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
BM25_CACHE   = CACHE_DIR / "bm25.pkl"
VECTORS_CACHE= CACHE_DIR / "vectors.npy"
OUTPUTS_DIR  = PA_ROOT / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "expert_eval_summary.json"
DETAIL_PATH  = OUTPUTS_DIR / "expert_eval_detail.json"

DEFAULT_N = 200
HF_DATASET = "Wix/WixQA"
EW_CONFIG  = "wixqa_expertwritten"

# chunk_id in the index uses first 16 chars of SHA-256 article_id:
# e.g. "wixqa_7e965b1f1ac71d00" → article_id prefix "7e965b1f1ac71d00"
CHUNK_ID_PREFIX = "wixqa_"


# ── Tokenizer (must match indexer.py) ─────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ── Cache loading ──────────────────────────────────────────────────────────────

def load_cache():
    print("[expert_eval] Loading cache...")
    with open(CHUNKS_CACHE, "rb") as f:
        chunk_dicts = pickle.load(f)
    with open(BM25_CACHE, "rb") as f:
        bm25 = pickle.load(f)
    vectors = np.load(str(VECTORS_CACHE))
    print(f"[expert_eval] {len(chunk_dicts)} chunks, vectors {vectors.shape}")
    return chunk_dicts, bm25, vectors


def load_emb_model():
    from fastembed import TextEmbedding
    print("[expert_eval] Loading embedding model...")
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed_query(query: str, emb_model) -> np.ndarray:
    return np.array(list(emb_model.embed([query]))[0], dtype=np.float32)


# ── Expert QA pairs loading ────────────────────────────────────────────────────

def load_expert_queries(n: int) -> list[dict]:
    """Load wixqa_expertwritten QA pairs from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets --break-system-packages")

    print(f"[expert_eval] Loading {HF_DATASET}/{EW_CONFIG}...")
    ds = load_dataset(HF_DATASET, EW_CONFIG, split="train")
    pairs = []
    for i, row in enumerate(ds):
        if n and i >= n:
            break
        raw_ids = row.get("article_ids", [])
        if isinstance(raw_ids, str):
            raw_ids = re.findall(r"'([a-f0-9]{64})'", raw_ids)
        # Normalise: full SHA-256 IDs → first 16 chars to match chunk_id prefix
        gt_ids_full  = list(raw_ids)
        gt_ids_short = [aid[:16] for aid in gt_ids_full]
        pairs.append({
            "query_id":     f"expert_{i:04d}",
            "question":     row.get("question", "").strip(),
            "answer":       row.get("answer", "").strip(),
            "gt_ids_full":  gt_ids_full,   # full SHA-256 article IDs
            "gt_ids_short": gt_ids_short,  # first 16 chars — matches chunk_id suffix
        })
    print(f"[expert_eval] Loaded {len(pairs)} expert QA pairs")
    return pairs


# ── Retrieval modes ────────────────────────────────────────────────────────────

def retrieve_bm25(query: str, bm25, chunk_dicts: list[dict], k: int) -> list[dict]:
    scores   = bm25.get_scores(_tokenize(query))
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"article_id": chunk_dicts[i].get("chunk_id", ""),
              "text": chunk_dicts[i].get("text", "")[:1200],
              "score": float(scores[i]), "rank": r + 1}
            for r, i in enumerate(top_idxs)]


def retrieve_dense(query: str, vectors: np.ndarray, chunk_dicts: list[dict],
                   emb_model, k: int) -> list[dict]:
    q_vec = embed_query(query, emb_model)
    norms = np.linalg.norm(vectors, axis=1)
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-9:
        return []
    sims = (vectors @ q_vec) / (norms * q_norm + 1e-9)
    top_idxs = np.argsort(sims)[::-1][:k]
    return [{"article_id": chunk_dicts[int(i)].get("chunk_id", ""),
              "text": chunk_dicts[int(i)].get("text", "")[:1200],
              "score": float(sims[i]), "rank": r + 1}
            for r, i in enumerate(top_idxs)]


def retrieve_hybrid(query: str, bm25, vectors: np.ndarray, chunk_dicts: list[dict],
                    emb_model, k: int, rrf_k: int = 60) -> list[dict]:
    fetch_k = min(k * 4, len(chunk_dicts))
    b_res = retrieve_bm25(query, bm25, chunk_dicts, fetch_k)
    d_res = retrieve_dense(query, vectors, chunk_dicts, emb_model, fetch_k)
    b_ranks = {r["article_id"]: r["rank"] for r in b_res}
    d_ranks = {r["article_id"]: r["rank"] for r in d_res}
    all_ids = set(b_ranks) | set(d_ranks)
    def rrf(aid):
        return 1/(rrf_k + b_ranks.get(aid, fetch_k+1)) + 1/(rrf_k + d_ranks.get(aid, fetch_k+1))
    ranked = sorted(all_ids, key=rrf, reverse=True)[:k]
    id_info = {r["article_id"]: r for r in b_res}
    id_info.update({r["article_id"]: r for r in d_res if r["article_id"] not in id_info})
    return [{"article_id": aid,
              "text": id_info.get(aid, {}).get("text", ""),
              "rrf_score": round(rrf(aid), 6), "rank": r + 1}
            for r, aid in enumerate(ranked)]


# ── Precision@K calculation ────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], gt_ids_short: list[str], k: int) -> float:
    """
    Precision@K: fraction of top-K retrieved chunks whose article_id matches
    any ground-truth article ID (using first-16-char short form for comparison).

    chunk_id format: "wixqa_{article_id[:16]}"
    gt_ids_short: first 16 chars of full SHA-256 ground-truth article IDs
    """
    if not gt_ids_short:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(
        1 for aid in top_k
        # strip "wixqa_" prefix to get the 16-char article ID
        if aid.replace(CHUNK_ID_PREFIX, "")[:16] in gt_ids_short
    )
    return round(hits / max(k, 1), 4)


def recall_at_k(retrieved_ids: list[str], gt_ids_short: list[str], k: int) -> float:
    """Recall@K: fraction of ground-truth articles found in top-K retrieved."""
    if not gt_ids_short:
        return 0.0
    top_k = retrieved_ids[:k]
    retrieved_short = {aid.replace(CHUNK_ID_PREFIX, "")[:16] for aid in top_k}
    hits = sum(1 for gt in gt_ids_short if gt in retrieved_short)
    return round(hits / len(gt_ids_short), 4)


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
            chunk for chunk in chunks[:TOP_K_VERIFY]
            if (r := _checker.check(claim, chunk.get("text", ""), chunk.get("article_id", "")))
            and r.verdict == "SUPPORTS" and r.confidence >= NLI_CONFIDENCE_THRESHOLD
        ]
        return {
            "decision":   "ANSWER_WITH_CITATION" if verified else "ABSTAIN",
            "precision":  round(len(verified) / max(min(len(chunks), TOP_K_VERIFY), 1), 3),
            "n_verified": len(verified),
        }
    except Exception as e:
        return {"decision": "ERROR", "precision": None, "n_verified": 0, "error": str(e)}


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_expert_eval(n: int = DEFAULT_N) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_dicts, bm25, vectors = load_cache()
    emb_model = load_emb_model()
    qa_pairs  = load_expert_queries(n)

    MODES = ["bm25_only", "dense_only", "hybrid_rrf"]
    K_VALUES = [1, 3, 10]

    all_results: dict[str, list[dict]] = {m: [] for m in MODES}

    print(f"\n[expert_eval] {len(qa_pairs)} expert questions × 3 modes")
    print("=" * 60)

    for i, qa in enumerate(qa_pairs):
        question     = qa["question"]
        gt_ids_short = qa["gt_ids_short"]

        for mode in MODES:
            t0 = time.time()

            if mode == "bm25_only":
                chunks = retrieve_bm25(question, bm25, chunk_dicts, TOP_K_RETRIEVE)
            elif mode == "dense_only":
                chunks = retrieve_dense(question, vectors, chunk_dicts, emb_model, TOP_K_RETRIEVE)
            else:
                chunks = retrieve_hybrid(question, bm25, vectors, chunk_dicts, emb_model, TOP_K_RETRIEVE)

            retrieved_ids = [c["article_id"] for c in chunks]
            nli    = run_nli(question, chunks)
            elapsed = round(time.time() - t0, 3)

            record = {
                "query_id":    qa["query_id"],
                "question":    question,
                "gt_ids_short": gt_ids_short,
                "mode":        mode,
                "route":       nli["decision"],
                "latency_s":   elapsed,
            }
            for k in K_VALUES:
                record[f"precision_at_{k}"] = precision_at_k(retrieved_ids, gt_ids_short, k)
                record[f"recall_at_{k}"]    = recall_at_k(retrieved_ids, gt_ids_short, k)

            all_results[mode].append(record)

        if (i + 1) % 20 == 0 or (i + 1) == len(qa_pairs):
            print(f"[expert_eval] {i+1}/{len(qa_pairs)} done")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def agg(mode_results: list[dict]) -> dict:
        total    = len(mode_results)
        answered = sum(1 for r in mode_results if r["route"] == "ANSWER_WITH_CITATION")
        lats     = sorted(r["latency_s"] for r in mode_results)
        out = {
            "n":           total,
            "answer_rate": round(answered / total, 3),
            "mean_latency_s": round(sum(lats) / len(lats), 3),
            "p50_latency_s":  lats[len(lats) // 2],
            "p95_latency_s":  lats[int(0.95 * len(lats))],
        }
        for k in K_VALUES:
            p_vals = [r[f"precision_at_{k}"] for r in mode_results]
            r_vals = [r[f"recall_at_{k}"]    for r in mode_results]
            out[f"mean_precision_at_{k}"] = round(sum(p_vals) / len(p_vals), 4)
            out[f"mean_recall_at_{k}"]    = round(sum(r_vals) / len(r_vals), 4)
        return out

    summary = {
        "eval_date":     datetime.now(timezone.utc).isoformat(),
        "eval_set":      f"{HF_DATASET}/{EW_CONFIG}",
        "n_queries":     len(qa_pairs),
        "nli_threshold": NLI_CONFIDENCE_THRESHOLD,
        "top_k_retrieve": TOP_K_RETRIEVE,
        "top_k_verify":   TOP_K_VERIFY,
        "k_values":       K_VALUES,
        "modes":         {m: agg(all_results[m]) for m in MODES},
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    DETAIL_PATH.write_text(json.dumps(all_results, indent=2))

    # ── Print comparison table ─────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  EXPERT EVAL — PRECISION@K / RECALL@K vs GROUND-TRUTH ARTICLE IDs")
    print("=" * 72)
    print(f"  {'Metric':<30} {'BM25-only':>12} {'Dense-only':>12} {'Hybrid RRF':>12}")
    print("  " + "-" * 68)

    def prow(label, key):
        vals = [summary["modes"][m].get(key) for m in MODES]
        cells = [f"{v:.4f}" if v is not None else "  N/A" for v in vals]
        print(f"  {label:<30} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")

    for k in K_VALUES:
        prow(f"Precision@{k}", f"mean_precision_at_{k}")
        prow(f"Recall@{k}",    f"mean_recall_at_{k}")
        print()

    prow("Answer rate", "answer_rate")
    prow("Mean latency (s)", "mean_latency_s")
    prow("P95 latency (s)", "p95_latency_s")
    print("=" * 72)
    print(f"  N={len(qa_pairs)}, eval_set={EW_CONFIG}")
    print(f"  Summary → {SUMMARY_PATH}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Number of expert QA pairs to evaluate")
    args = parser.parse_args()
    run_expert_eval(n=args.n)
