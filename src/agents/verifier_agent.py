"""
src/agents/verifier_agent.py — VerifierAgent specialist subgraph

Responsibility: given aggregated chunks and the original query, run NLI
citation verification (cross-encoder/nli-deberta-v3-small) and return
verified chunks with ANSWER_WITH_CITATION or ABSTAIN decision.

NLI claim conversion: questions fail entailment; convert to declarative
claim: "This article provides information about: {query}".

Input:  VerifierState{"query": str, "chunks": [...]}
Output: VerifierState{"verified_chunks": [...], "contract_decision": str, ...}
"""
from __future__ import annotations
import sys
from pathlib import Path

PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from langgraph.graph import StateGraph, END
from src.agents.state import VerifierState
from src.tools.nli_tool import verify_citations
from config import TOP_K_VERIFY


def verify_node(state: VerifierState) -> dict:
    """Run NLI entailment check on top-K chunks."""
    # Convert question → declarative claim for correct NLI entailment
    raw   = state["query"].strip().rstrip("?")
    claim = f"This article provides information about: {raw}"

    result = verify_citations.invoke({
        "query":  claim,
        "chunks": state["chunks"][:TOP_K_VERIFY],
    })

    decision  = result.get("contract_decision", "ABSTAIN")
    precision = result.get("citation_precision")
    verified  = result.get("verified_chunks", [])
    print(f"[verifier_agent] decision={decision}, precision={precision}, "
          f"verified={len(verified)}/{min(len(state['chunks']), TOP_K_VERIFY)}")

    return {
        "verified_chunks":   verified,
        "contract_decision": decision,
        "citation_precision": precision,
    }


def build_verifier_agent():
    """Build and compile the VerifierAgent subgraph."""
    g = StateGraph(VerifierState)
    g.add_node("verify", verify_node)
    g.set_entry_point("verify")
    g.add_edge("verify", END)
    return g.compile()
