"""
src/agents/retrieval_agent.py — RetrievalAgent specialist subgraph

Responsibility: given one sub-query, run hybrid BM25+dense RRF retrieval
and return ranked chunks. Compiled as an independent StateGraph and used
as a node in the SupervisorAgent via LangGraph's Send() fan-out.

Input:  RetrievalState{"query": str, "all_retrieved_chunks": []}
Output: RetrievalState{"all_retrieved_chunks": [chunk, ...]}

The `all_retrieved_chunks` key matches the Annotated reducer in
SupervisorState, so parallel RetrievalAgent runs accumulate automatically.
"""
from __future__ import annotations
import sys
from pathlib import Path

PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from langgraph.graph import StateGraph, END
from src.agents.state import RetrievalState
from src.tools.retriever_tool import retrieve_chunks


def retrieve_node(state: RetrievalState) -> dict:
    """Run hybrid RRF retrieval for one sub-query."""
    chunks = retrieve_chunks.invoke({"query": state["query"]})
    print(f"[retrieval_agent] query='{state['query'][:60]}...' → {len(chunks)} chunks")
    return {"all_retrieved_chunks": chunks}


def build_retrieval_agent():
    """Build and compile the RetrievalAgent subgraph."""
    g = StateGraph(RetrievalState)
    g.add_node("retrieve", retrieve_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()
