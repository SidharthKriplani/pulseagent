"""
indexer.py — Build Qdrant dense index + BM25 index over the accepted corpus
PulseKnowledge · G4 Retrieval Baseline

Embedding model: BAAI/bge-small-en-v1.5 via fastembed (ONNX, 384-dim)
Vector store:    Qdrant in-memory (local dev; swap to qdrant-client(url=...) for deployed)
BM25:            rank_bm25.BM25Okapi with whitespace tokenization

Design choices (interview-defensible):
  - BAAI/bge-small-en-v1.5: state-of-the-art small embedding model. ONNX via fastembed
    avoids PyTorch overhead. 384-dim vectors are efficient without sacrificing quality
    for this corpus size (392 chunks).
  - Qdrant pre-filter: the store supports filtering by doc_type/version/corpus_tag BEFORE
    cosine ranking. This is the core reason Qdrant was chosen over FAISS — version-aware
    retrieval in G5 requires filtering to current-version documents, not post-filtering
    after ranking.
  - BM25Okapi: standard Okapi BM25 (Robertson et al. 1994). Strong for exact technical
    terms (e.g. "AC-2", "OAuth 2.0", "NIST 800-53"). Dense embeddings may miss exact
    identifiers that BM25 handles natively.

Claim status: [BUILT] on accepted synthetic + public corpus
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .corpus import CorpusChunk

# Silence ONNX CPU vendor warning
os.environ.setdefault("FASTEMBED_CACHE_PATH",
                      str(Path.home() / ".fastembed_cache"))

EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM    = 384
QDRANT_COLLECTION = "pulseknowledge_g4"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class RetrievalIndex:
    """
    Holds both the Qdrant dense index and the BM25 index over the same corpus.
    Single source of truth for chunk ordering.
    """

    def __init__(self, chunks: list[CorpusChunk]) -> None:
        self.chunks = chunks
        self._qdrant = None
        self._bm25   = None
        self._emb_model = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, verbose: bool = True) -> "RetrievalIndex":
        """Build both indexes. Returns self for chaining."""
        self._build_bm25(verbose)
        self._build_qdrant(verbose)
        return self

    def _build_bm25(self, verbose: bool) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)
        if verbose:
            print(f"  BM25 index: {len(self.chunks)} docs")

    def _build_qdrant(self, verbose: bool) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from fastembed import TextEmbedding

        if verbose:
            print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self._emb_model = TextEmbedding(EMBEDDING_MODEL)

        texts = [c.text for c in self.chunks]
        if verbose:
            print(f"  Embedding {len(texts)} chunks...")
        vectors = list(self._emb_model.embed(texts))

        self._qdrant = QdrantClient(":memory:")
        self._qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id":       c.chunk_id,
                    "source":         c.source,
                    "doc_type":       c.doc_type,
                    "version":        c.version,
                    "effective_date": c.effective_date,
                    "corpus_tag":     c.corpus_tag,
                    "section_heading": c.section_heading,
                    "text_preview":   c.text[:200],
                },
            )
            for i, c in enumerate(self.chunks)
        ]
        self._qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        if verbose:
            print(f"  Qdrant index: {len(points)} vectors (cosine, {EMBEDDING_DIM}-dim)")

    # ── Query embedding ────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return list(self._emb_model.embed([query]))[0].tolist()

    # ── Dense retrieval ────────────────────────────────────────────────────────

    def dense_search(
        self, query: str, k: int = 5,
        filter_doc_type: Optional[str] = None,
        filter_version:  Optional[str] = None,
        filter_corpus_tag: Optional[str] = None,
    ) -> list[dict]:
        """
        Dense retrieval via Qdrant cosine similarity.

        Supports pre-filter on doc_type, version, or corpus_tag — applied BEFORE
        ranking (not post-filter). This is the core Qdrant advantage for version-aware
        retrieval in G5.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Query

        must_conditions = []
        if filter_doc_type:
            must_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=filter_doc_type))
            )
        if filter_version:
            must_conditions.append(
                FieldCondition(key="version", match=MatchValue(value=filter_version))
            )
        if filter_corpus_tag:
            must_conditions.append(
                FieldCondition(key="corpus_tag", match=MatchValue(value=filter_corpus_tag))
            )

        q_filter = Filter(must=must_conditions) if must_conditions else None
        q_vec    = self.embed_query(query)

        result = self._qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_vec,
            limit=k,
            query_filter=q_filter,
            with_payload=True,
        )
        hits = result.points

        return [
            {
                "rank":           i + 1,
                "score":          round(h.score, 4),
                "chunk_id":       h.payload["chunk_id"],
                "source":         h.payload["source"],
                "doc_type":       h.payload["doc_type"],
                "version":        h.payload["version"],
                "corpus_tag":     h.payload["corpus_tag"],
                "section_heading": h.payload["section_heading"],
                "text_preview":   h.payload["text_preview"],
                "chunk_index":    h.id,
            }
            for i, h in enumerate(hits)
        ]

    # ── BM25 retrieval ─────────────────────────────────────────────────────────

    def bm25_search(self, query: str, k: int = 5) -> list[dict]:
        """BM25 keyword retrieval using BM25Okapi."""
        q_tokens = _tokenize(query)
        scores   = self._bm25.get_scores(q_tokens)

        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "rank":           i + 1,
                "score":          round(float(scores[idx]), 4),
                "chunk_id":       self.chunks[idx].chunk_id,
                "source":         self.chunks[idx].source,
                "doc_type":       self.chunks[idx].doc_type,
                "version":        self.chunks[idx].version,
                "corpus_tag":     self.chunks[idx].corpus_tag,
                "section_heading": self.chunks[idx].section_heading,
                "text_preview":   self.chunks[idx].text[:200],
                "chunk_index":    idx,
            }
            for i, idx in enumerate(top_idxs)
        ]

    # ── Hybrid (RRF) ──────────────────────────────────────────────────────────

    def hybrid_search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[dict]:
        """
        Hybrid retrieval: Reciprocal Rank Fusion (RRF) of BM25 + dense rankings.

        RRF score for document d:
            rrf(d) = Σ_r  1 / (k + rank_r(d))

        where k=60 is the RRF smoothing constant (standard value from Cormack et al.).
        RRF is normalization-free — BM25 and cosine scores are not directly comparable,
        but ranks are. This avoids the score calibration problem of linear combination.

        Interview note: "I chose RRF over score-based fusion because BM25 scores and
        cosine similarity scores are on incomparable scales. Normalizing them introduces
        a hyperparameter (the mixing weight). RRF sidesteps this by operating only on
        ranks — and its effectiveness is well-supported in the literature."
        """
        # Fetch more candidates for fusion
        fetch_k = min(k * 4, len(self.chunks))
        dense_results = self.dense_search(query, k=fetch_k)
        bm25_results  = self.bm25_search(query,  k=fetch_k)

        # Build rank maps: chunk_index → rank
        dense_ranks = {r["chunk_index"]: r["rank"] for r in dense_results}
        bm25_ranks  = {r["chunk_index"]: r["rank"] for r in bm25_results}

        # Union of candidate documents
        all_idxs = set(dense_ranks) | set(bm25_ranks)

        def rrf_score(idx: int) -> float:
            d = 1.0 / (rrf_k + dense_ranks.get(idx, fetch_k + 1))
            b = 1.0 / (rrf_k + bm25_ranks.get(idx,  fetch_k + 1))
            return d + b

        ranked = sorted(all_idxs, key=rrf_score, reverse=True)[:k]

        # Look up full info from dense results map (fallback to BM25 map)
        idx_to_info: dict[int, dict] = {r["chunk_index"]: r for r in dense_results}
        idx_to_info.update({r["chunk_index"]: r for r in bm25_results
                            if r["chunk_index"] not in idx_to_info})

        results = []
        for i, idx in enumerate(ranked):
            info = idx_to_info.get(idx, {})
            results.append({
                "rank":           i + 1,
                "rrf_score":      round(rrf_score(idx), 6),
                "dense_rank":     dense_ranks.get(idx),
                "bm25_rank":      bm25_ranks.get(idx),
                "chunk_id":       info.get("chunk_id", self.chunks[idx].chunk_id if idx < len(self.chunks) else ""),
                "source":         info.get("source", ""),
                "doc_type":       info.get("doc_type", ""),
                "version":        info.get("version", ""),
                "corpus_tag":     info.get("corpus_tag", ""),
                "section_heading": info.get("section_heading", ""),
                "text_preview":   info.get("text_preview", ""),
                "chunk_index":    idx,
            })
        return results
