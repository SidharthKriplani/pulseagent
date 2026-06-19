"""
src/agents/supervisor.py — SupervisorAgent (top-level multi-agent graph)

Architecture:
  planner_node       — LLM decomposes query → sub-queries; fans out via Send()
  retrieval_agent    — specialist subgraph (one per sub-query, run in parallel)
  aggregate_node     — deduplicates + ranks all retrieved chunks
  verifier_agent     — specialist subgraph: NLI citation check
  synthesis_agent    — specialist subgraph: generator + reflector loop

Fan-out pattern (LangGraph Send API):
  planner_node returns Command(goto=[Send("retrieval_agent", {q}) for q in sub_queries])
  Each RetrievalAgent writes to all_retrieved_chunks → Annotated reducer accumulates.
  aggregate_node runs once all parallel retrievals complete.
"""
from __future__ import annotations
import sys
from pathlib import Path

PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

import json, threading
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type, before_sleep_log)
import logging

from langgraph.graph import StateGraph, END
from langgraph.types import Command, Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.agents.state import SupervisorState, VerifierState, SynthesisState
from src.agents.retrieval_agent import build_retrieval_agent
from src.agents.verifier_agent import build_verifier_agent
from src.agents.synthesis_agent import build_synthesis_agent
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K_RETRIEVE

logger = logging.getLogger("pulseagent.supervisor")


def _get_llm(json_mode: bool = False):
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
        model=LLM_MODEL, temperature=0.2,
    )
    if json_mode:
        llm = llm.bind(response_format={"type": "json_object"})
    return llm


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm_with_retry(llm, messages):
    return llm.invoke(messages)


# ── Node 1: Planner + fan-out dispatcher ──────────────────────────────────────

def planner_node(state: SupervisorState) -> Command:
    """
    Decompose query into sub-queries, then fan out to RetrievalAgent
    instances in parallel using LangGraph's Send() API.
    Uses JSON mode for structured output — no fragile string parsing.
    Retries up to 3x with exponential backoff on LLM errors.
    """
    llm = _get_llm(json_mode=True)
    response = _call_llm_with_retry(llm, [HumanMessage(content=
        f"""You are a planning agent. Given this user query, return a JSON object with:
- "intent": string (1 sentence describing the search intent)
- "sub_queries": array of up to 3 focused sub-queries that together cover the full information need

Query: {state['query']}

Return only valid JSON, no other text."""
    )])

    intent, sub_queries = "", [state["query"]]
    try:
        data = json.loads(response.content)
        intent = data.get("intent", "")
        parsed = data.get("sub_queries", [])
        if isinstance(parsed, list) and parsed:
            sub_queries = [q for q in parsed if isinstance(q, str)][:3]
    except (json.JSONDecodeError, AttributeError):
        logger.warning("[supervisor.planner] JSON parse failed — using original query")

    sub_queries = sub_queries or [state["query"]]
    logger.info(f"[supervisor.planner] intent='{intent}' | {len(sub_queries)} sub-queries")

    # Fan out: each sub-query goes to a separate RetrievalAgent instance
    return Command(
        update={
            "sub_queries":          sub_queries,
            "search_intent":        intent,
            "all_retrieved_chunks": [],   # reset accumulator
        },
        goto=[
            Send("retrieval_agent", {
                "query":                q,
                "all_retrieved_chunks": [],
            })
            for q in sub_queries
        ],
    )


# ── Node 2: Aggregate (runs after all parallel retrievals complete) ────────────

def aggregate_node(state: SupervisorState) -> dict:
    """
    Deduplicate and rank chunks accumulated from parallel RetrievalAgent runs.
    Routes to VerifierAgent with deduplicated top-K chunks.
    """
    all_chunks = state.get("all_retrieved_chunks", [])

    # Deduplicate by article_id, keep highest rrf_score
    seen: dict[str, dict] = {}
    for c in all_chunks:
        aid = c.get("article_id", "")
        if aid not in seen or c.get("rrf_score", 0) > seen[aid].get("rrf_score", 0):
            seen[aid] = c

    ranked = sorted(seen.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
    top_chunks = ranked[:TOP_K_RETRIEVE]

    print(f"[supervisor.aggregate] {len(all_chunks)} raw → {len(top_chunks)} unique chunks")
    # Overwrite all_retrieved_chunks with deduplicated result
    return {"all_retrieved_chunks": top_chunks}


# ── Node 3: VerifierAgent wrapper ──────────────────────────────────────────────

def verifier_wrapper_node(state: SupervisorState) -> dict:
    """
    Run compiled VerifierAgent subgraph and merge result into SupervisorState.
    """
    verifier = build_verifier_agent()
    result   = verifier.invoke({
        "query":             state["query"],
        "chunks":            state["all_retrieved_chunks"],
        "verified_chunks":   [],
        "contract_decision": "ABSTAIN",
        "citation_precision": None,
    })
    return {
        "verified_chunks":   result["verified_chunks"],
        "contract_decision": result["contract_decision"],
        "citation_precision": result["citation_precision"],
    }


# ── Node 4: SynthesisAgent wrapper ────────────────────────────────────────────

def synthesis_wrapper_node(state: SupervisorState) -> dict:
    """
    Run compiled SynthesisAgent subgraph and merge result into SupervisorState.
    Uses verified chunks if available; falls back to all retrieved chunks.
    """
    synthesis = build_synthesis_agent()
    chunks    = state.get("verified_chunks") or state.get("all_retrieved_chunks", [])

    result = synthesis.invoke({
        "query":             state["query"],
        "chunks":            chunks,
        "contract_decision": state.get("contract_decision", "ABSTAIN"),
        "draft_answer":      "",
        "final_answer":      "",
        "cited_article_ids": [],
        "reflection_notes":  "",
        "reflection_passed": False,
        "retry_count":       0,
        "route":             "",
    })
    return {
        "final_answer":      result.get("final_answer", ""),
        "draft_answer":      result.get("draft_answer", ""),
        "cited_article_ids": result.get("cited_article_ids", []),
        "reflection_notes":  result.get("reflection_notes", ""),
        "reflection_passed": result.get("reflection_passed", True),
        "route":             result.get("route", state.get("contract_decision", "ABSTAIN")),
    }


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_supervisor():
    """
    Build and compile the SupervisorAgent multi-agent graph.

    Graph structure:
      planner ──Send()──► retrieval_agent (×N, parallel)
                              │ (all converge via Annotated reducer)
                              ▼
                         aggregate
                              │
                              ▼
                         verifier_wrapper
                              │
                              ▼
                         synthesis_wrapper
                              │
                              ▼
                             END
    """
    retrieval_agent = build_retrieval_agent()

    g = StateGraph(SupervisorState)

    # Specialist subgraph as a node
    g.add_node("retrieval_agent",    retrieval_agent)
    # Supervisor orchestration nodes
    g.add_node("planner",            planner_node)
    g.add_node("aggregate",          aggregate_node)
    g.add_node("verifier_wrapper",   verifier_wrapper_node)
    g.add_node("synthesis_wrapper",  synthesis_wrapper_node)

    g.set_entry_point("planner")
    # planner uses Command/Send — edges are implicit in Command.goto
    g.add_edge("retrieval_agent",   "aggregate")
    g.add_edge("aggregate",         "verifier_wrapper")
    g.add_edge("verifier_wrapper",  "synthesis_wrapper")
    g.add_edge("synthesis_wrapper", END)

    return g.compile()


_supervisor = None
_supervisor_lock = threading.Lock()

def run_supervisor(query: str) -> dict:
    """Run the multi-agent supervisor and return structured result."""
    global _supervisor
    if _supervisor is None:
        with _supervisor_lock:
            if _supervisor is None:
                _supervisor = build_supervisor()

    init = {
        "query":                query,
        "sub_queries":          [],
        "search_intent":        "",
        "all_retrieved_chunks": [],
        "verified_chunks":      [],
        "contract_decision":    "",
        "citation_precision":   None,
        "draft_answer":         "",
        "final_answer":         "",
        "cited_article_ids":    [],
        "reflection_notes":     "",
        "reflection_passed":    False,
        "retry_count":          0,
        "route":                "",
    }

    result = _supervisor.invoke(init)
    return {
        "query":        result["query"],
        "final_answer": result.get("final_answer", ""),
        "route":        result.get("route", "ABSTAIN"),
        "cited_ids":    result.get("cited_article_ids", []),
        "reflection":   result.get("reflection_notes", ""),
        "sub_queries":  result.get("sub_queries", []),
    }
