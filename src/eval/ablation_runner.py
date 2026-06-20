"""
ablation_runner.py — Retrieval ablation: BM25-only vs Dense-only vs Hybrid RRF

Compares three retrieval modes on N=50 sampled WixQA article titles.
For each mode, runs NLI verification and records answer_rate + citation_precision + latency.

Usage:
    python3 src/eval/ablation_runner.py
    python3 src/eval/ablation_runner.py --n 100

Output:
    outputs/ablation_summary.json   — comparison table (all three modes)
    outputs/ablation_detail.json    — per-query results for all three modes
"""

from __future__ import annotations
import sys, json, time, pickle, random, argparse, re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PA_ROOT))

from config import TOP_K_RETRIEVE, TOP_K_VERIFY, NLI_CONFIDENCE_THRESHOLD

CACHE_DIR      = PA_ROOT / ".cache"
CHUNKS_CACHE   = CACHE_DIR / "chunks.pkl"
BM25_CACHE     = CACHE_DIR / "bm25.pkl"
VECTORS_CACHE  = CACHE_DIR / "vectors.npy"
OUTPUTS_DIR    = PA_ROOT / "outputs"
SUMMARY_PATH   = OUTPUTS_DIR / "ablation_summary.json"
DETAIL_PATH    = OUTPUTS_DIR / "ablation_detail.json"

DEFAULT_N = 50
SEED      = 42
RRF_K     = 60


# ── Tokenizer (must match indexer.py) ─────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ── Cache loading ──────────────────────────────────────────────────────────────

def load_cache():
    """Load chunks, BM25, and embedding vectors from .cache/"""
    print("[ablation] Loading cache...")
    with open(CHUNKS_CACHE, "rb") as f:
        chunk_dicts = pickle.load(f)
    with open(BM25_CACHE, "rb") as f:
        bm25 = pickle.load(f)
    vectors = np.load(str(VECTORS_CACHE))   # shape: (N_docs, 384)
    print(f"[ablation] {len(chunk_dicts)} chunks, vectors shape: {vectors.shape}")
    return chunk_dicts, bm25, vectors


def embed_query(query: str, emb_model) -> np.ndarray:
    """Embed a single query string using the fastembed model."""
    return np.array(list(emb_model.embed([query]))[0], dtype=np.float32)


def load_emb_model():
    """Load BAAI/bge-small-en-v1.5 via fastembed for query embedding."""
    from fastembed import TextEmbedding
    print("[ablation] Loading embedding model (BAAI/bge-small-en-v1.5)...")
    return TextEmbedding("BAAI/bge-small-en-v1.5")


# ── Retrieval modes ────────────────────────────────────────────────────────────

def bm25_only(query: str, bm25, chunk_dicts: list[dict], k: int) -> list[dict]:
    """BM25-only retrieval."""
    q_tokens = _tokenize(query)
    scores   = bm25.get_scores(q_tokens)
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "article_id": chunk_dicts[i].get("chunk_id", f"chunk_{i}"),
            "text":       chunk_dicts[i].get("text", "")[:1200],
            "score":      float(scores[i]),
            "rank":       rank + 1,
            "source":     chunk_dicts[i].get("source", ""),
            "mode":       "bm25",
        }
        for rank, i in enumerate(top_idxs)
    ]


def dense_only(query: str, vectors: np.ndarray, chunk_dicts: list[dict],
               emb_model, k: int) -> list[dict]:
    """Dense-only retrieval via cosine similarity (numpy)."""
    q_vec = embed_query(query, emb_model)          # (384,)
    # Cosine similarity: dot product of unit vectors
    norms = np.linalg.norm(vectors, axis=1)        # (N_docs,)
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-9 or np.any(norms < 1e-9):
        return []
    sims = (vectors @ q_vec) / (norms * q_norm)    # (N_docs,)
    top_idxs = np.argsort(sims)[::-1][:k]
    return [
        {
            "article_id": chunk_dicts[int(i)].get("chunk_id", f"chunk_{i}"),
            "text":       chunk_dicts[int(i)].get("text", "")[:1200],
            "score":      float(sims[i]),
            "rank":       rank + 1,
            "source":     chunk_dicts[int(i)].get("source", ""),
            "mode":       "dense",
        }
        for rank, i in enumerate(top_idxs)
    ]


def hybrid_rrf(query: str, bm25, vectors: np.ndarray, chunk_dicts: list[dict],
               emb_model, k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Hybrid RRF: fuse BM25 and dense rankings."""
    fetch_k = min(k * 4, len(chunk_dicts))
    bm25_results  = bm25_only(query, bm25, chunk_dicts, fetch_k)
    dense_results = dense_only(query, vectors, chunk_dicts, emb_model, fetch_k)

    # Build rank maps by article_id
    bm25_ranks  = {r["article_id"]: r["rank"] for r in bm25_results}
    dense_ranks = {r["article_id"]: r["rank"] for r in dense_results}

    all_ids = set(bm25_ranks) | set(dense_ranks)

    def rrf_score(aid: str) -> float:
        b = 1.0 / (rrf_k + bm25_ranks.get(aid, fetch_k + 1))
        d = 1.0 / (rrf_k + dense_ranks.get(aid, fetch_k + 1))
        return b + d

    ranked_ids = sorted(all_ids, key=rrf_score, reverse=True)[:k]

    id_to_info = {r["article_id"]: r for r in bm25_results}
    id_to_info.update({r["article_id"]: r for r in dense_results
                       if r["article_id"] not in id_to_info})

    return [
        {
            **id_to_info.get(aid, {"article_id": aid, "text": "", "source": ""}),
            "rrf_score":   round(rrf_score(aid), 6),
            "bm25_rank":   bm25_ranks.get(aid),
            "dense_rank":  dense_ranks.get(aid),
            "rank":        rank + 1,
            "mode":        "hybrid_rrf",
        }
        for rank, aid in enumerate(ranked_ids)
    ]


# ── NLI gate (reuse from nli_tool) ────────────────────────────────────────────

_checker = None

def _get_checker():
    global _checker
    if _checker is None:
        from src.citation.entailment import NLICitationChecker
        _checker = NLICitationChecker()
    return _checker


def run_nli(query: str, chunks: list[dict]) -> dict:
    """Run NLI on top-3 chunks. Returns {decision, precision, n_verified}."""
    try:
        checker    = _get_checker()
        raw        = query.strip().rstrip("?")
        claim      = f"This article provides information about: {raw}"
        candidates = chunks[:TOP_K_VERIFY]
        verified   = []
        for chunk in candidates:
            result = checker.check(
                claim=claim,
                chunk_text=chunk.get("text", ""),
                chunk_id=chunk.get("article_id", ""),
            )
            if result.verdict == "SUPPORTS" and result.confidence >= NLI_CONFIDENCE_THRESHOLD:
                verified.append(chunk)
        decision  = "ANSWER_WITH_CITATION" if verified else "ABSTAIN"
        precision = round(len(verified) / max(len(candidates), 1), 3)
        return {"decision": decision, "precision": precision,
                "n_verified": len(verified), "error": None}
    except Exception as e:
        return {"decision": "ERROR", "precision": None, "n_verified": 0, "error": str(e)}


# ── Query sampling ─────────────────────────────────────────────────────────────

def sample_queries(chunk_dicts: list[dict], n: int, seed: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in chunk_dicts:
        src   = c.get("source", "").strip()
        title = c.get("title", "").strip()
        if src and title and src not in seen:
            seen[src] = {"query": title, "source": src}
    articles = list(seen.values())
    random.seed(seed)
    return random.sample(articles, min(n, len(articles)))


# ── Main ablation loop ─────────────────────────────────────────────────────────

def run_ablation(n: int = DEFAULT_N) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_dicts, bm25, vectors = load_cache()
    emb_model = load_emb_model()
    queries   = sample_queries(chunk_dicts, n, SEED)

    MODES = ["bm25_only", "dense_only", "hybrid_rrf"]
    results: dict[str, list[dict]] = {m: [] for m in MODES}

    print(f"\n[ablation] Running {len(queries)} queries × 3 modes")
    print("=" * 60)

    for i, q in enumerate(queries):
        query = q["query"]
        row   = {"query_id": i, "query": query, "source": q["source"]}

        for mode in MODES:
            t0 = time.time()

            if mode == "bm25_only":
                chunks = bm25_only(query, bm25, chunk_dicts, TOP_K_RETRIEVE)
            elif mode == "dense_only":
                chunks = dense_only(query, vectors, chunk_dicts, emb_model, TOP_K_RETRIEVE)
            else:
                chunks = hybrid_rrf(query, bm25, vectors, chunk_dicts, emb_model, TOP_K_RETRIEVE)

            nli   = run_nli(query, chunks)
            elapsed = round(time.time() - t0, 3)

            results[mode].append({
                **row,
                "route":      nli["decision"],
                "precision":  nli["precision"],
                "n_retrieved": len(chunks),
                "n_verified": nli["n_verified"],
                "latency_s":  elapsed,
                "error":      nli["error"],
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"[ablation] {i+1}/{len(queries)} done")

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    def aggregate(mode_results: list[dict]) -> dict:
        total  = len(mode_results)
        answered = sum(1 for r in mode_results if r["route"] == "ANSWER_WITH_CITATION")
        errors   = sum(1 for r in mode_results if r["route"] == "ERROR")
        precs    = [r["precision"] for r in mode_results if r["precision"] is not None]
        lats     = sorted(r["latency_s"] for r in mode_results)
        return {
            "n":                   total,
            "answer_rate":         round(answered / total, 3),
            "abstain_rate":        round((total - answered - errors) / total, 3),
            "error_rate":          round(errors / total, 3),
            "mean_citation_prec":  round(sum(precs) / len(precs), 3) if precs else None,
            "mean_latency_s":      round(sum(lats) / len(lats), 3),
            "p50_latency_s":       lats[len(lats) // 2],
            "p95_latency_s":       lats[int(0.95 * len(lats))],
        }

    summary = {
        "eval_date":  datetime.now(timezone.utc).isoformat(),
        "n_queries":  n,
        "seed":       SEED,
        "nli_threshold": NLI_CONFIDENCE_THRESHOLD,
        "top_k_retrieve": TOP_K_RETRIEVE,
        "top_k_verify":   TOP_K_VERIFY,
        "modes": {m: aggregate(results[m]) for m in MODES},
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    DETAIL_PATH.write_text(json.dumps(results, indent=2))

    # ── Print comparison table ─────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  RETRIEVAL ABLATION — COMPARISON TABLE")
    print("=" * 68)
    hdr = f"  {'Metric':<28} {'BM25-only':>12} {'Dense-only':>12} {'Hybrid RRF':>12}"
    print(hdr)
    print("  " + "-" * 64)

    def row(label, key, fmt="{:.3f}"):
        vals = [summary["modes"][m].get(key) for m in MODES]
        cells = [fmt.format(v) if v is not None else "  N/A  " for v in vals]
        print(f"  {label:<28} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")

    row("Answer rate",       "answer_rate")
    row("Abstain rate",      "abstain_rate")
    row("Error rate",        "error_rate")
    row("Mean citation prec","mean_citation_prec")
    row("Mean latency (s)",  "mean_latency_s")
    row("P50 latency (s)",   "p50_latency_s")
    row("P95 latency (s)",   "p95_latency_s")
    print("=" * 68)
    print(f"  N={n}, seed={SEED}, NLI threshold={NLI_CONFIDENCE_THRESHOLD}")
    print(f"  Summary → {SUMMARY_PATH}")
    print(f"  Detail  → {DETAIL_PATH}")
    print("=" * 68)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    run_ablation(n=args.n)
