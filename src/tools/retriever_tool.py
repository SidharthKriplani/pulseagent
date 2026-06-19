from __future__ import annotations
import sys, pickle
import numpy as np
from pathlib import Path
from typing import List
from langchain_core.tools import tool

# PK modules are now bundled inside src/ — no cross-repo sys.path needed
PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from config import TOP_K_RETRIEVE, CONTEXT_WINDOW

CACHE_DIR     = Path(__file__).parent.parent.parent / ".cache"
CHUNKS_CACHE  = CACHE_DIR / "chunks.pkl"
BM25_CACHE    = CACHE_DIR / "bm25.pkl"
VECTORS_CACHE = CACHE_DIR / "vectors.npy"

_index = None


def _load_from_cache():
    """Reload index from 3-part cache — no re-embedding (~10-15s)."""
    from src.retrieval.indexer import (
        RetrievalIndex, QDRANT_COLLECTION, EMBEDDING_DIM, EMBEDDING_MODEL
    )
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from fastembed import TextEmbedding

    print("[retriever] Loading chunks from cache...")
    with open(CHUNKS_CACHE, "rb") as f:
        chunk_dicts = pickle.load(f)
    from src.retrieval.corpus import CorpusChunk
    chunks = [CorpusChunk(**d) for d in chunk_dicts]

    print("[retriever] Loading BM25 from cache...")
    with open(BM25_CACHE, "rb") as f:
        bm25 = pickle.load(f)

    print("[retriever] Loading vectors from cache (numpy)...")
    vectors = np.load(str(VECTORS_CACHE))

    print(f"[retriever] Rebuilding Qdrant from {len(chunks)} saved vectors (no embedding)...")
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id":        c.chunk_id,
                "source":          c.source,
                "doc_type":        c.doc_type,
                "version":         c.version,
                "effective_date":  c.effective_date,
                "corpus_tag":      c.corpus_tag,
                "section_heading": c.section_heading,
                "text_preview":    c.text[:200],
            },
        )
        for i, c in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

    print("[retriever] Loading embedding model for query-time use...")
    emb_model = TextEmbedding(EMBEDDING_MODEL)

    idx            = RetrievalIndex(chunks)
    idx._bm25      = bm25
    idx._qdrant    = qdrant
    idx._emb_model = emb_model
    return idx


def _save_cache(idx, vectors: np.ndarray) -> None:
    """Persist 3 serializable parts to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    # Save chunks as plain dicts — avoids back-reference to idx (which holds ONNX emb_model)
    chunk_dicts = [vars(c) for c in idx.chunks]
    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(chunk_dicts, f)
    with open(BM25_CACHE, "wb") as f:
        pickle.dump(idx._bm25, f)
    np.save(str(VECTORS_CACHE), vectors)
    print(f"[retriever] Cache saved → {CACHE_DIR}  (next run loads in ~10-15s)")


def _build_fresh():
    """Full cold build: load corpus, embed 6221 chunks, index. ONE-TIME ~5-8 min."""
    import re
    from rank_bm25 import BM25Okapi
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from src.corpus.wixqa_adapter import load_kb_articles, articles_to_corpus_chunks
    from src.retrieval.indexer import (
        RetrievalIndex, QDRANT_COLLECTION, EMBEDDING_DIM, EMBEDDING_MODEL
    )

    def tokenize(text: str):
        return re.findall(r"[a-z0-9]+", text.lower())

    print("[retriever] Loading WixQA corpus (6,221 articles)...")
    articles = load_kb_articles()
    chunks   = articles_to_corpus_chunks(articles)
    print(f"[retriever] {len(chunks)} chunks loaded.")

    print("[retriever] Building BM25 index...")
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    print(f"[retriever] Loading embedding model: {EMBEDDING_MODEL}")
    emb_model = TextEmbedding(EMBEDDING_MODEL)

    texts = [c.text for c in chunks]
    print(f"[retriever] Embedding {len(texts)} chunks (ONE-TIME slow step)...")
    vectors = np.array(list(emb_model.embed(texts)))
    print(f"[retriever] Embeddings done: {vectors.shape}")

    print("[retriever] Building Qdrant index...")
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id":        c.chunk_id,
                "source":          c.source,
                "doc_type":        c.doc_type,
                "version":         c.version,
                "effective_date":  c.effective_date,
                "corpus_tag":      c.corpus_tag,
                "section_heading": c.section_heading,
                "text_preview":    c.text[:200],
            },
        )
        for i, c in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

    idx            = RetrievalIndex(chunks)
    idx._bm25      = bm25
    idx._qdrant    = qdrant
    idx._emb_model = emb_model

    try:
        _save_cache(idx, vectors)
    except Exception as e:
        print(f"[retriever] WARNING: cache save failed ({e}) — next run will rebuild again")

    return idx


def _get_index():
    global _index
    if _index is not None:
        return _index

    cache_ok = (CHUNKS_CACHE.exists() and BM25_CACHE.exists() and VECTORS_CACHE.exists())
    if cache_ok:
        print("[retriever] Cache found — loading without re-embedding...")
        try:
            _index = _load_from_cache()
            print("[retriever] Index ready (from cache).")
            return _index
        except Exception as e:
            print(f"[retriever] Cache load failed ({e}) — rebuilding from scratch...")

    _index = _build_fresh()
    print("[retriever] Index ready.")
    return _index


@tool
def retrieve_chunks(query: str) -> List[dict]:
    """Retrieve top-K evidence chunks from WixQA corpus via hybrid RRF retrieval."""
    idx     = _get_index()
    results = idx.hybrid_search(query=query, k=TOP_K_RETRIEVE, rrf_k=60)
    chunks  = idx.chunks  # full-text access by index position

    output = []
    for i, r in enumerate(results):
        chunk_idx = r.get("chunk_index", -1)
        # Use full text (up to CONTEXT_WINDOW) — NOT the 200-char text_preview
        if 0 <= chunk_idx < len(chunks):
            full_text = chunks[chunk_idx].text[:CONTEXT_WINDOW]
        else:
            full_text = r.get("text_preview", "")[:CONTEXT_WINDOW]

        output.append({
            "article_id": r.get("chunk_id", f"chunk_{i}"),
            "text":       full_text,
            "rrf_score":  r.get("rrf_score", 0.0),
            "rank":       r.get("rank", i + 1),
            "source":     r.get("source", ""),
            "heading":    r.get("section_heading", ""),
        })
    return output
