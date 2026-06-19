"""
extract_cache.py  —  ONE-TIME cache migration script
Run once from pulseagent root:  python3 extract_cache.py

Reads:   .cache/retrieval_index.pkl  (old monolithic pickle)
Writes:  .cache/chunks.pkl           (list of CorpusChunk)
         .cache/bm25.pkl             (BM25Okapi)
         .cache/vectors.npy          (6221×384 float32 numpy array)

After this runs, delete retrieval_index.pkl — it is no longer needed.
Every subsequent run of main.py loads the 3-part cache in ~10-15s.
"""

import sys, pickle, numpy as np
from pathlib import Path

# ── sys.path: PK_SRC must come first so 'retrieval.corpus' resolves ──────────
PA_ROOT = Path(__file__).parent
PK_SRC  = PA_ROOT.parent / "pulseknowledge" / "src"
assert PK_SRC.exists(), f"PulseKnowledge src not found: {PK_SRC}"
sys.path.insert(0, str(PK_SRC))

CACHE_DIR      = PA_ROOT / ".cache"
OLD_PICKLE     = CACHE_DIR / "retrieval_index.pkl"
CHUNKS_CACHE   = CACHE_DIR / "chunks.pkl"
BM25_CACHE     = CACHE_DIR / "bm25.pkl"
VECTORS_CACHE  = CACHE_DIR / "vectors.npy"

assert OLD_PICKLE.exists(), f"Old pickle not found: {OLD_PICKLE}"

# ── Load old pickle (real qdrant-client + rank-bm25 must be installed) ────────
print(f"[extract] Loading {OLD_PICKLE.name}  ({OLD_PICKLE.stat().st_size/1e6:.1f} MB)...")
with open(OLD_PICKLE, "rb") as f:
    idx = pickle.load(f)

print(f"[extract] Type: {type(idx)}")
print(f"[extract] Attrs: {list(vars(idx).keys())}")

# ── Chunks ────────────────────────────────────────────────────────────────────
chunks = idx.chunks
print(f"[extract] {len(chunks)} chunks found.")

# ── BM25 ──────────────────────────────────────────────────────────────────────
bm25 = idx._bm25
print(f"[extract] BM25: {type(bm25)}")

# ── Vectors: scroll all points from in-memory Qdrant ─────────────────────────
from retrieval.indexer import QDRANT_COLLECTION, EMBEDDING_DIM

qdrant = idx._qdrant
print(f"[extract] Extracting vectors from Qdrant (scrolling all {len(chunks)} points)...")

all_points = []
offset = None
while True:
    result, next_offset = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=500,
        offset=offset,
        with_vectors=True,
    )
    all_points.extend(result)
    if next_offset is None:
        break
    offset = next_offset

print(f"[extract] Retrieved {len(all_points)} points from Qdrant.")
assert len(all_points) == len(chunks), \
    f"Mismatch: {len(all_points)} points vs {len(chunks)} chunks"

# Sort by ID so vectors[i] corresponds to chunks[i]
all_points.sort(key=lambda p: p.id)
vectors = np.array([p.vector for p in all_points], dtype=np.float32)
print(f"[extract] Vectors shape: {vectors.shape}  (expected ({len(chunks)}, {EMBEDDING_DIM}))")

# ── Save 3-part cache ─────────────────────────────────────────────────────────
CACHE_DIR.mkdir(exist_ok=True)

print(f"[extract] Saving chunks.pkl ...")
with open(CHUNKS_CACHE, "wb") as f:
    pickle.dump(chunks, f)

print(f"[extract] Saving bm25.pkl ...")
with open(BM25_CACHE, "wb") as f:
    pickle.dump(bm25, f)

print(f"[extract] Saving vectors.npy ...")
np.save(str(VECTORS_CACHE), vectors)

print()
print("=" * 60)
print("  Cache migration complete.")
print(f"  chunks.pkl  : {CHUNKS_CACHE.stat().st_size/1e6:.1f} MB")
print(f"  bm25.pkl    : {BM25_CACHE.stat().st_size/1e6:.1f} MB")
print(f"  vectors.npy : {VECTORS_CACHE.stat().st_size/1e6:.1f} MB")
print()
print("  You can now delete .cache/retrieval_index.pkl")
print("  All future runs will load in ~10-15s.")
print("=" * 60)
