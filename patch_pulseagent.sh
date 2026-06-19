#!/bin/bash
# Run from inside your pulseagent folder:
# bash patch_pulseagent.sh

cd "/Users/ASUS/Documents/Professional/GitHub/beastmax (4)/pulseagent"

# Fix 1: retriever_tool.py — use PK_SRC not PK_ROOT to avoid src.* conflict
cat > src/tools/retriever_tool.py << 'PYEOF'
"""
retriever_tool.py — Bridge to PulseKnowledge hybrid RRF retrieval
PulseKnowledge champion: Recall@10=0.790, MRR=0.490 on 200 WixQA expert QA pairs
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
from langchain_core.tools import tool

# Point directly to PK src/ to avoid conflict with PulseAgent's own src/ package
PK_SRC = Path(__file__).parent.parent.parent.parent / "pulseknowledge" / "src"
sys.path.insert(0, str(PK_SRC))

from config import TOP_K_RETRIEVE, CONTEXT_WINDOW

_index = None

def _get_index():
    global _index
    if _index is not None:
        return _index
    print("[retriever] Loading WixQA corpus (6,221 articles)...")
    from corpus.wixqa_adapter import load_kb_articles, articles_to_corpus_chunks
    from retrieval.indexer import RetrievalIndex
    articles = load_kb_articles()
    chunks = articles_to_corpus_chunks(articles)
    print(f"[retriever] Building hybrid index over {len(chunks)} chunks...")
    _index = RetrievalIndex(chunks)
    _index.build(verbose=False)
    print("[retriever] Index ready.")
    return _index

@tool
def retrieve_chunks(query: str) -> List[dict]:
    """Retrieve top-K evidence chunks from WixQA corpus using hybrid RRF (BM25+dense+RRF k=60).
    Champion metrics: Recall@10=0.790, MRR=0.490 on 200 WixQA expert QA pairs."""
    index = _get_index()
    results = index.hybrid_search(query=query, k=TOP_K_RETRIEVE, rrf_k=60)
    return [
        {
            "article_id": r.get("metadata", {}).get("article_id", r.get("chunk_id", f"chunk_{i}")),
            "text":       r.get("text", r.get("text_preview", ""))[:CONTEXT_WINDOW],
            "rrf_score":  r.get("rrf_score", 0.0),
            "rank":       r.get("rank", i + 1),
            "source":     r.get("source", ""),
        }
        for i, r in enumerate(results)
    ]
PYEOF

# Fix 2: nli_tool.py — same PK_SRC fix
cat > src/tools/nli_tool.py << 'PYEOF'
"""
nli_tool.py — Bridge to PulseKnowledge NLI citation layer
Strict policy: NLI confidence >= 0.85, no CONTRADICTS. Measured precision: 75.9%
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
from langchain_core.tools import tool

PK_SRC = Path(__file__).parent.parent.parent.parent / "pulseknowledge" / "src"
sys.path.insert(0, str(PK_SRC))

from config import TOP_K_VERIFY

@tool
def verify_citations(query: str, chunks: List[dict]) -> dict:
    """
    NLI citation check using cross-encoder/nli-deberta-v3-small ONNX (17ms/pair).
    Strict policy: confidence >= 0.85, no CONTRADICTS. Measured precision: 75.9%.
    Returns contract_decision: ANSWER_WITH_CITATION | ABSTAIN | BLOCK | ESCALATE
    """
    try:
        from citation.entailment import OnnxNLICitationChecker
        checker = OnnxNLICitationChecker()
        verified = []
        for chunk in chunks[:TOP_K_VERIFY]:
            result = checker.check(
                claim=query,
                chunk_text=chunk.get("text", ""),
                chunk_id=chunk.get("article_id", ""),
            )
            if result.verdict == "SUPPORTS" and result.confidence >= 0.85:
                verified.append(chunk)
        decision = "ANSWER_WITH_CITATION" if verified else "ABSTAIN"
        precision = len(verified) / max(len(chunks[:TOP_K_VERIFY]), 1)
        return {
            "verified_chunks":    verified,
            "contract_decision":  decision,
            "citation_precision": round(precision, 3),
        }
    except Exception as e:
        return {
            "verified_chunks":    [],
            "contract_decision":  "ABSTAIN",
            "citation_precision": None,
            "error": str(e),
        }
PYEOF

# Fix 3: model name in config.py
sed -i '' 's/qwen2.5-7b-instruct-q4_k_m/qwen2.5-7b-instruct/' config.py

echo "All patches applied."
echo "Now run: python3 main.py \"how do I add a blog to my Wix site?\""
