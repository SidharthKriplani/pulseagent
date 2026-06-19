"""
src/agents/state.py — State schemas for PulseAgent multi-agent system

SupervisorState  : top-level shared state
RetrievalState   : per-specialist state for RetrievalAgent subgraph
VerifierState    : per-specialist state for VerifierAgent subgraph
SynthesisState   : per-specialist state for SynthesisAgent subgraph

The `all_retrieved_chunks` field uses an Annotated reducer so parallel
Send()-dispatched RetrievalAgent runs accumulate their results correctly.
"""
from __future__ import annotations
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
import operator


class SupervisorState(TypedDict):
    # Input
    query:                str
    # Planner output
    sub_queries:          List[str]
    search_intent:        str
    # RetrievalAgent fan-out: Annotated reducer merges parallel results
    all_retrieved_chunks: Annotated[List[dict], operator.add]
    # VerifierAgent output
    verified_chunks:      List[dict]
    contract_decision:    str
    citation_precision:   Optional[float]
    # SynthesisAgent output
    draft_answer:         str
    final_answer:         str
    cited_article_ids:    List[str]
    reflection_notes:     str
    reflection_passed:    bool
    retry_count:          int
    route:                str


class RetrievalState(TypedDict):
    """State for a single RetrievalAgent run (one sub-query)."""
    query:                str
    # Writes to this key; SupervisorState accumulates via operator.add
    all_retrieved_chunks: List[dict]


class VerifierState(TypedDict):
    """State for VerifierAgent."""
    query:             str
    chunks:            List[dict]
    verified_chunks:   List[dict]
    contract_decision: str
    citation_precision: Optional[float]


class SynthesisState(TypedDict):
    """State for SynthesisAgent (generator + reflector loop)."""
    query:             str
    chunks:            List[dict]
    contract_decision: str
    draft_answer:      str
    final_answer:      str
    cited_article_ids: List[str]
    reflection_notes:  str
    reflection_passed: bool
    retry_count:       int
    route:             str
