"""
corpus.py — Load and tag accepted chunks from G3/G3.6 JSONL files
PulseKnowledge · G4 Retrieval Baseline

Loads chunks from two accepted sources:
  SYNTHETIC — outputs/chunks/*.jsonl (G3 synthetic corpus, 6 docs, 203 chunks)
  PUBLIC    — outputs/chunks_public/nist_sp800-53r5_ac_ia_au_sc_chunks.jsonl (G3.6, 189 chunks)

Quarantined sources (do NOT load):
  outputs/chunks_public/nist_sp800-63b_chunks.jsonl  ← FAILED QA (57.7% bad split rate)

Data classification:
  corpus_tags: ["SYNTHETIC", "PUBLIC"]  (distinct — never merged without labeling)
  quarantined: ["PUBLIC (SP 800-63B — QA FAIL)"]

Claim status: [BUILT] on accepted corpus · no retrieval quality claim yet
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CorpusChunk:
    """A single chunk from the accepted corpus with full provenance."""
    chunk_id: str
    text: str
    source: str
    page: int
    doc_type: str
    version: str
    effective_date: str
    title: str
    section_heading: str
    chunk_index: int
    corpus_tag: str          # "SYNTHETIC" or "PUBLIC"
    corpus_source_file: str  # JSONL file this chunk came from
    metadata: dict = field(default_factory=dict)


# QUARANTINED — must not appear in retrieval index
QUARANTINED_FILES = frozenset({"nist_sp800-63b_chunks.jsonl"})


def load_accepted_corpus(repo_root: Path) -> list[CorpusChunk]:
    """
    Load all chunks from accepted JSONL files.
    Explicitly skips quarantined SP 800-63B chunks.

    Returns list of CorpusChunk, tagged SYNTHETIC or PUBLIC.
    """
    chunks: list[CorpusChunk] = []

    # ── Synthetic corpus (G3) ─────────────────────────────────────────────────
    syn_dir = repo_root / "outputs" / "chunks"
    for jsonl in sorted(syn_dir.glob("*.jsonl")):
        if jsonl.name in QUARANTINED_FILES:
            continue
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            chunks.append(_parse(raw, corpus_tag="SYNTHETIC", source_file=jsonl.name))

    # ── Public corpus (G3.6) ─────────────────────────────────────────────────
    pub_dir = repo_root / "outputs" / "chunks_public"
    accepted_public = {"nist_sp800-53r5_ac_ia_au_sc_chunks.jsonl"}
    for jsonl in sorted(pub_dir.glob("*.jsonl")):
        if jsonl.name not in accepted_public:
            # Log quarantine skip
            continue
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            chunks.append(_parse(raw, corpus_tag="PUBLIC", source_file=jsonl.name))

    return chunks


def _parse(raw: dict, corpus_tag: str, source_file: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=raw.get("chunk_id", ""),
        text=raw.get("text", ""),
        source=raw.get("source", ""),
        page=raw.get("page", 0),
        doc_type=raw.get("doc_type", ""),
        version=raw.get("version", ""),
        effective_date=raw.get("effective_date", ""),
        title=raw.get("title", ""),
        section_heading=raw.get("section_heading", ""),
        chunk_index=raw.get("chunk_index", 0),
        corpus_tag=corpus_tag,
        corpus_source_file=source_file,
    )


def corpus_stats(chunks: list[CorpusChunk]) -> dict:
    """Summary statistics over the loaded corpus."""
    from collections import Counter
    tags   = Counter(c.corpus_tag for c in chunks)
    dtypes = Counter(c.doc_type for c in chunks)
    return {
        "total_chunks": len(chunks),
        "by_corpus_tag": dict(tags),
        "by_doc_type": dict(dtypes),
        "sources": sorted({c.source for c in chunks}),
        "quarantined_note": "nist_sp800-63b_chunks.jsonl excluded — QA FAIL (57.7% bad split rate)",
    }
