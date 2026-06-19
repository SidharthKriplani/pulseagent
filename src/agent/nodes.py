"""
nodes.py — LangGraph node functions for PulseAgent
Graph flow: planner -> retriever -> nli_verifier -> generator -> reflector -> [END or retry]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import (LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL,
                    MAX_RETRIES, TOP_K_VERIFY)
from src.agent.state import AgentState
from src.tools.retriever_tool import retrieve_chunks
from src.tools.nli_tool import verify_citations


def _get_llm():
    return ChatOpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY,
                      model=LM_STUDIO_MODEL, temperature=0.2)


def planner_node(state: AgentState) -> dict:
    """Decompose query into sub-queries + intent. Maps to: task decomposition, planning."""
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=f"""You are a planning agent. Given this user query, output:
1. A brief search intent (1 sentence)
2. Up to 3 focused sub-queries that together cover the full information need

Query: {state['query']}

Respond in this exact format:
INTENT: <intent>
SUB_QUERIES:
- <sub-query 1>
- <sub-query 2>
- <sub-query 3>""")])
    intent, sub_queries = "", [state["query"]]
    for line in response.content.split("\n"):
        if line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip()
        elif line.strip().startswith("- "):
            sub_queries.append(line.strip()[2:])
    print(f"[planner] intent='{intent}' | sub_queries={sub_queries[:3]}")
    return {"search_intent": intent,
            "decomposed_queries": sub_queries[:3] if sub_queries else [state["query"]]}


def retriever_node(state: AgentState) -> dict:
    """Hybrid RRF retrieval per sub-query, deduplicated. PK champion: R@10=0.790."""
    seen_ids, all_chunks = set(), []
    for q in (state.get("decomposed_queries") or [state["query"]]):
        for chunk in retrieve_chunks.invoke({"query": q}):
            if chunk["article_id"] not in seen_ids:
                seen_ids.add(chunk["article_id"])
                all_chunks.append(chunk)
    all_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
    print(f"[retriever] {len(all_chunks)} unique chunks retrieved")
    return {"retrieved_chunks": all_chunks[:10]}


def nli_verifier_node(state: AgentState) -> dict:
    """NLI citation check. Converts question -> declarative claim for correct entailment.
    Maps to: evaluation frameworks, probabilistic condition testing."""
    raw = state["query"].strip().rstrip("?")
    claim = f"This article provides information about: {raw}"
    result = verify_citations.invoke({"query": claim,
                                      "chunks": state["retrieved_chunks"][:TOP_K_VERIFY]})
    decision = result.get("contract_decision", "ABSTAIN")
    precision = result.get("citation_precision")
    error = result.get("error")
    print(f"[nli] decision={decision}, precision={precision}" + (f", error={error}" if error else ""))
    return {"verified_chunks": result.get("verified_chunks", []),
            "contract_decision": decision,
            "citation_precision": precision}


def generator_node(state: AgentState) -> dict:
    """Generate grounded answer. Always runs — verified chunks if available,
    falls back to retrieved. Maps to: LLM integration, function calling."""
    decision = state.get("contract_decision", "ABSTAIN")
    verified  = state.get("verified_chunks") or []
    retrieved = state.get("retrieved_chunks") or []
    chunks = verified if verified else retrieved[:TOP_K_VERIFY]
    confidence_note = ("Citation-verified (NLI SUPPORTS >= 0.85)" if verified
                       else "Best-effort (NLI threshold not met — answer with care)")
    print(f"[generator] decision={decision}, using {len(chunks)} chunks, calling LLM...")

    context = "\n\n".join(
        f"[Source {i+1} | ID: {c['article_id']}]\n{c['text']}"
        for i, c in enumerate(chunks))

    response = _get_llm().invoke([HumanMessage(content=f"""You are a precise knowledge assistant for Wix help articles.
Answer ONLY using the provided sources. If insufficient, say so.

QUERY: {state['query']}

SOURCES ({confidence_note}):
{context}

Instructions: answer concisely, cite sources as [Source N], no hallucination.""")])

    print(f"[generator] answer generated ({len(response.content)} chars)")
    return {"draft_answer": response.content,
            "cited_article_ids": [c["article_id"] for c in chunks],
            "route": decision}


def reflector_node(state: AgentState) -> dict:
    """Self-critique draft answer. Retry if unfaithful. Maps to: self-reflective loops."""
    if not state.get("draft_answer"):
        return {"reflection_passed": True, "reflection_notes": "No draft.",
                "final_answer": "[No answer generated]",
                "route": state.get("contract_decision", "ABSTAIN")}

    if state.get("retry_count", 0) >= MAX_RETRIES:
        return {"reflection_passed": True,
                "reflection_notes": f"Max retries ({MAX_RETRIES}) reached.",
                "final_answer": state["draft_answer"], "route": "ANSWER"}

    text = _get_llm().invoke([HumanMessage(content=f"""Review this answer. Is it grounded in sources, specific, and answers the query?
QUERY: {state['query']}
ANSWER: {state['draft_answer']}
Respond: PASS or RETRY\nREASON: <one sentence>""")]).content.strip()

    passed = text.upper().startswith("PASS")
    print(f"[reflector] {'PASS' if passed else 'RETRY'} — {text[:80]}")
    if passed:
        return {"reflection_passed": True, "reflection_notes": text,
                "final_answer": state["draft_answer"], "route": "ANSWER"}
    return {"reflection_passed": False, "reflection_notes": text,
            "retry_count": state.get("retry_count", 0) + 1}
