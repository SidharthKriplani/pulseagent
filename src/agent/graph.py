"""
graph.py — LangGraph agent graph
planner → retriever → nli_verifier → generator → reflector → [END or retry to generator]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import planner_node, retriever_node, nli_verifier_node, generator_node, reflector_node
from config import MAX_RETRIES

def _should_retry(state: AgentState) -> str:
    if not state.get("reflection_passed") and state.get("retry_count", 0) < MAX_RETRIES:
        return "generator"
    return END

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner",      planner_node)
    g.add_node("retriever",    retriever_node)
    g.add_node("nli_verifier", nli_verifier_node)
    g.add_node("generator",    generator_node)
    g.add_node("reflector",    reflector_node)
    g.set_entry_point("planner")
    g.add_edge("planner",      "retriever")
    g.add_edge("retriever",    "nli_verifier")
    g.add_edge("nli_verifier", "generator")
    g.add_edge("generator",    "reflector")
    g.add_conditional_edges("reflector", _should_retry, {"generator": "generator", END: END})
    return g.compile()

_graph = None

def run_agent(query: str, session_id: str = "default") -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    init = {
        "query": query, "session_id": session_id,
        "decomposed_queries": [], "search_intent": "",
        "retrieved_chunks": [], "verified_chunks": [],
        "contract_decision": "", "citation_precision": None,
        "draft_answer": "", "cited_article_ids": [],
        "reflection_notes": "", "reflection_passed": False,
        "retry_count": 0, "conversation_history": [],
        "final_answer": "", "route": "", "error": None,
    }
    result = _graph.invoke(init)
    return {
        "query":        result["query"],
        "final_answer": result.get("final_answer", ""),
        "route":        result.get("route", "ABSTAIN"),
        "cited_ids":    result.get("cited_article_ids", []),
        "reflection":   result.get("reflection_notes", ""),
    }
