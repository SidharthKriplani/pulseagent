"""
eval_runner.py — PulseAgent evaluation over 200 WixQA queries

Runs retriever + NLI pipeline (no LLM required — LM Studio not needed).
Uses cached index — no re-embedding.

Output:
  outputs/evidence/<query_id>.json   per-query evidence record
  outputs/eval_summary.json          aggregate metrics

Usage:
  python3 src/eval/eval_runner.py
  python3 src/eval/eval_runner.py --n 50   # smaller sample
"""

from __future__ import annotations
import sys, json, time, pickle, random, argparse
from pathlib import Path
from datetime import datetime, timezone

PA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PA_ROOT))

from src.tools.retriever_tool import retrieve_chunks
from src.tools.nli_tool import verify_citations
from config import TOP_K_VERIFY

CACHE_DIR    = PA_ROOT / ".cache"
EVIDENCE_DIR = PA_ROOT / "outputs" / "evidence"
SUMMARY_PATH = PA_ROOT / "outputs" / "eval_summary.json"
DEFAULT_N    = 200
SEED         = 42


# ── Query sampling ─────────────────────────────────────────────────────────────

def load_eval_queries(n: int = DEFAULT_N, seed: int = SEED) -> list[dict]:
    """Sample N unique articles from cached chunks.pkl. Uses title as query."""
    chunks_path = CACHE_DIR / "chunks.pkl"
    assert chunks_path.exists(), f"Cache not found: {chunks_path}. Run main.py once first."

    with open(chunks_path, "rb") as f:
        chunk_dicts = pickle.load(f)

    seen_sources: dict[str, dict] = {}
    for c in chunk_dicts:
        src = c.get("source", "").strip()
        title = c.get("title", "").strip()
        if src and title and src not in seen_sources:
            seen_sources[src] = {"query": title, "source": src,
                                  "chunk_id": c.get("chunk_id", "")}

    articles = list(seen_sources.values())
    random.seed(seed)
    sampled = random.sample(articles, min(n, len(articles)))
    print(f"[eval] Sampled {len(sampled)} queries from {len(articles)} unique articles.")
    return sampled


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_eval(n: int = DEFAULT_N) -> dict:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_eval_queries(n)
    results: list[dict] = []

    print(f"[eval] Starting evaluation — {len(queries)} queries")
    print(f"[eval] Evidence → {EVIDENCE_DIR}")
    print()

    t_total_start = time.time()

    for i, q in enumerate(queries):
        query_text = q["query"]
        t0 = time.time()

        # ── Retrieval ──────────────────────────────────────────────────────────
        try:
            chunks = retrieve_chunks.invoke({"query": query_text})
        except Exception as e:
            chunks = []
            retrieval_error = str(e)
        else:
            retrieval_error = None

        # ── NLI citation verification ──────────────────────────────────────────
        raw   = query_text.strip().rstrip("?")
        claim = f"This article provides information about: {raw}"
        try:
            nli_result = verify_citations.invoke({
                "query": claim,
                "chunks": chunks[:TOP_K_VERIFY],
            })
        except Exception as e:
            nli_result = {"contract_decision": "ERROR", "citation_precision": None,
                          "verified_chunks": [], "error": str(e)}

        elapsed = round(time.time() - t0, 3)

        route     = nli_result.get("contract_decision", "ERROR")
        precision = nli_result.get("citation_precision")
        n_verified = len(nli_result.get("verified_chunks", []))

        record = {
            "query_id":          f"eval_{i:04d}",
            "query":             query_text,
            "source":            q["source"],
            "route":             route,
            "citation_precision": precision,
            "n_retrieved":       len(chunks),
            "n_verified":        n_verified,
            "latency_s":         elapsed,
            "retrieval_error":   retrieval_error,
            "nli_error":         nli_result.get("error"),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

        # Save per-query evidence (includes chunk ids for traceability)
        evidence = {**record, "retrieved_chunks": [
            {"article_id": c.get("article_id"), "rrf_score": c.get("rrf_score"),
             "heading": c.get("heading", "")}
            for c in chunks
        ]}
        (EVIDENCE_DIR / f"{record['query_id']}.json").write_text(
            json.dumps(evidence, indent=2))

        results.append(record)

        if (i + 1) % 25 == 0 or (i + 1) == len(queries):
            answered  = sum(1 for r in results if r["route"] == "ANSWER_WITH_CITATION")
            abstained = sum(1 for r in results if r["route"] == "ABSTAIN")
            errors    = sum(1 for r in results if r["route"] == "ERROR")
            elapsed_total = round(time.time() - t_total_start, 1)
            print(f"[eval] {i+1:3d}/{len(queries)}  "
                  f"ANSWER={answered}  ABSTAIN={abstained}  ERROR={errors}  "
                  f"({elapsed_total}s elapsed)")

    # ── Summary metrics ────────────────────────────────────────────────────────
    total     = len(results)
    answered  = sum(1 for r in results if r["route"] == "ANSWER_WITH_CITATION")
    abstained = sum(1 for r in results if r["route"] == "ABSTAIN")
    errors    = sum(1 for r in results if r["route"] == "ERROR")

    precisions = [r["citation_precision"] for r in results
                  if r["citation_precision"] is not None]
    latencies  = sorted(r["latency_s"] for r in results)

    summary = {
        "eval_date":        datetime.now(timezone.utc).isoformat(),
        "total_queries":    total,
        "route_distribution": {
            "ANSWER_WITH_CITATION": answered,
            "ABSTAIN":              abstained,
            "ERROR":                errors,
        },
        "answer_rate":              round(answered / total, 3),
        "abstain_rate":             round(abstained / total, 3),
        "error_rate":               round(errors / total, 3),
        "mean_citation_precision":  round(sum(precisions) / len(precisions), 3)
                                    if precisions else None,
        "mean_latency_s":           round(sum(latencies) / len(latencies), 3),
        "p50_latency_s":            latencies[len(latencies) // 2],
        "p95_latency_s":            latencies[int(0.95 * len(latencies))],
        "total_wall_time_s":        round(time.time() - t_total_start, 1),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 55)
    print("  PULSEAGENT EVAL SUMMARY")
    print("=" * 55)
    print(f"  Queries evaluated:       {total}")
    print(f"  ANSWER_WITH_CITATION:    {answered:3d}  ({100*answered/total:.1f}%)")
    print(f"  ABSTAIN:                 {abstained:3d}  ({100*abstained/total:.1f}%)")
    print(f"  ERROR:                   {errors:3d}  ({100*errors/total:.1f}%)")
    print(f"  Mean citation precision: {summary['mean_citation_precision']}")
    print(f"  Mean latency:            {summary['mean_latency_s']}s")
    print(f"  P95 latency:             {summary['p95_latency_s']}s")
    print(f"  Total wall time:         {summary['total_wall_time_s']}s")
    print("=" * 55)
    print(f"  Evidence:  {EVIDENCE_DIR}")
    print(f"  Summary:   {SUMMARY_PATH}")
    print("=" * 55)

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulseAgent eval runner")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of queries to evaluate (default: {DEFAULT_N})")
    args = parser.parse_args()
    run_eval(n=args.n)
