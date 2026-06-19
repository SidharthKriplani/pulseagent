"""
src/agents/synthesis_agent.py — SynthesisAgent specialist subgraph

Responsibility: grounded answer generation + self-reflective critique loop.
  generator_node  — produces citation-grounded answer from verified chunks
  reflector_node  — critiques answer; PASS → END, RETRY → generator (max 2)

Always runs even on ABSTAIN: if no verified chunks, falls back to
top retrieved chunks with a "best-effort" confidence note.

Input:  SynthesisState{"query", "chunks", "contract_decision", ...}
Output: SynthesisState{"final_answer", "cited_article_ids", "route", ...}
"""
from __future__ import annotations
import sys
from pathlib import Path

PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

import logging
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type, before_sleep_log)

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agents.state import SynthesisState
from config import (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
                    MAX_RETRIES, TOP_K_VERIFY)

logger = logging.getLogger("pulseagent.synthesis")


def _get_llm():
    return ChatOpenAI(
        base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
        model=LLM_MODEL, temperature=0.2,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm_with_retry(llm, messages):
    return llm.invoke(messages)


def generator_node(state: SynthesisState) -> dict:
    """Generate grounded answer. Falls back to retrieved chunks if none verified."""
    chunks    = state.get("chunks") or []
    decision  = state.get("contract_decision", "ABSTAIN")
    confidence_note = (
        "Citation-verified (NLI SUPPORTS ≥ 0.85)"
        if decision == "ANSWER_WITH_CITATION"
        else "Best-effort (NLI threshold not met — answer with care)"
    )
    print(f"[synthesis_agent.generator] decision={decision}, "
          f"chunks={len(chunks)}, calling LLM...")

    context = "\n\n".join(
        f"[Source {i+1} | ID: {c['article_id']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    response = _call_llm_with_retry(_get_llm(), [HumanMessage(content=
        f"""You are a precise knowledge assistant for Wix help articles.
Answer ONLY using the provided sources. If insufficient, say so.

QUERY: {state['query']}

SOURCES ({confidence_note}):
{context}

Instructions: answer concisely, cite sources as [Source N], no hallucination."""
    )])

    print(f"[synthesis_agent.generator] answer generated ({len(response.content)} chars)")
    return {
        "draft_answer":      response.content,
        "cited_article_ids": [c["article_id"] for c in chunks],
        "route":             decision,
    }


def reflector_node(state: SynthesisState) -> dict:
    """Self-critique draft. PASS → finalise. RETRY → back to generator."""
    if not state.get("draft_answer"):
        return {
            "reflection_passed": True,
            "reflection_notes":  "No draft produced.",
            "final_answer":      "[No answer generated]",
            "route":             state.get("contract_decision", "ABSTAIN"),
        }

    if state.get("retry_count", 0) >= MAX_RETRIES:
        return {
            "reflection_passed": True,
            "reflection_notes":  f"Max retries ({MAX_RETRIES}) reached.",
            "final_answer":      state["draft_answer"],
            "route":             "ANSWER",
        }

    text = _call_llm_with_retry(_get_llm(), [HumanMessage(content=
        f"""Review this answer. Is it grounded in sources, specific, and answers the query?
QUERY: {state['query']}
ANSWER: {state['draft_answer']}
Respond: PASS or RETRY\nREASON: <one sentence>"""
    )]).content.strip()

    passed = text.upper().startswith("PASS")
    print(f"[synthesis_agent.reflector] {'PASS' if passed else 'RETRY'} — {text[:80]}")

    if passed:
        return {
            "reflection_passed": True,
            "reflection_notes":  text,
            "final_answer":      state["draft_answer"],
            "route":             "ANSWER",
        }
    return {
        "reflection_passed": False,
        "reflection_notes":  text,
        "retry_count":       state.get("retry_count", 0) + 1,
    }


def _should_retry(state: SynthesisState) -> str:
    if not state.get("reflection_passed") and state.get("retry_count", 0) < MAX_RETRIES:
        return "generator"
    return END


def build_synthesis_agent():
    """Build and compile the SynthesisAgent subgraph."""
    g = StateGraph(SynthesisState)
    g.add_node("generator",  generator_node)
    g.add_node("reflector",  reflector_node)
    g.set_entry_point("generator")
    g.add_edge("generator", "reflector")
    g.add_conditional_edges(
        "reflector", _should_retry,
        {"generator": "generator", END: END},
    )
    return g.compile()
