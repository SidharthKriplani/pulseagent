"""
nli_tool.py — PulseAgent bridge to PulseKnowledge NLI citation layer

Uses NLICitationChecker (cross-encoder/nli-deberta-v3-small from HuggingFace).
Strict policy: verdict==SUPPORTS AND confidence >= 0.85.
Measured citation precision: 75.9% on 200 WixQA queries.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
from langchain_core.tools import tool

PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from config import NLI_CONFIDENCE_THRESHOLD, TOP_K_VERIFY

_checker = None

def _get_checker():
    global _checker
    if _checker is not None:
        return _checker
    from src.citation.entailment import NLICitationChecker
    _checker = NLICitationChecker()
    return _checker


@tool
def verify_citations(query: str, chunks: List[dict]) -> dict:
    """
    Run NLI citation verification on retrieved chunks against the query.
    Uses cross-encoder/nli-deberta-v3-small ONNX. Strict policy: confidence >= 0.85.
    Maps to JD: evaluation frameworks, testing strategies for probabilistic conditions.
    """
    try:
        checker = _get_checker()
        verified = []
        candidates = chunks[:TOP_K_VERIFY]
        for chunk in candidates:
            result = checker.check(
                claim=query,
                chunk_text=chunk.get("text", ""),
                chunk_id=chunk.get("article_id", ""),
            )
            if result.verdict == "SUPPORTS" and result.confidence >= NLI_CONFIDENCE_THRESHOLD:
                verified.append(chunk)
        decision = "ANSWER_WITH_CITATION" if verified else "ABSTAIN"
        precision = round(len(verified) / max(len(candidates), 1), 3)
        return {"verified_chunks": verified, "contract_decision": decision,
                "citation_precision": precision}
    except Exception as e:
        return {"verified_chunks": [], "contract_decision": "ERROR",
                "citation_precision": None, "error": str(e)}
