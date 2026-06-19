"""
Run this from inside your pulseagent folder:
  python3 setup_pulseagent.py
Creates all PulseAgent source files.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent

files = {}

files["config.py"] = '''"""
config.py — PulseAgent configuration
"""
from pathlib import Path

ROOT = Path(__file__).parent
PULSEKNOWLEDGE_ROOT = ROOT.parent / "pulseknowledge"
OUTPUTS_DIR = ROOT / "outputs" / "evidence"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"
LM_STUDIO_MODEL    = "qwen2.5-7b-instruct-q4_k_m"

MAX_RETRIES       = 2
TOP_K_RETRIEVE    = 10
TOP_K_VERIFY      = 3
CONTEXT_WINDOW    = 1200
NLI_CONFIDENCE_THRESHOLD = 0.85
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MCP_HOST = "localhost"
MCP_PORT = 8765
'''

files["src/__init__.py"] = ""
files["src/agent/__init__.py"] = ""
files["src/tools/__init__.py"] = ""
files["src/mcp_server/__init__.py"] = ""
files["src/eval/__init__.py"] = ""

files["src/agent/state.py"] = '''from __future__ import annotations
from typing import Annotated, List, Optional, TypedDict
import operator

class AgentState(TypedDict):
    query: str
    decomposed_queries: List[str]
    search_intent: str
    retrieved_chunks: List[dict]
    verified_chunks: List[dict]
    contract_decision: str
    citation_precision: Optional[float]
    draft_answer: str
    cited_article_ids: List[str]
    reflection_notes: str
    reflection_passed: bool
    retry_count: int
    conversation_history: Annotated[List[dict], operator.add]
    session_id: str
    final_answer: str
    route: str
    error: Optional[str]
'''

files["src/tools/retriever_tool.py"] = '''"""
retriever_tool.py — Bridge to PulseKnowledge hybrid RRF retrieval
PulseKnowledge champion: Recall@10=0.790, MRR=0.490 on 200 WixQA expert QA pairs
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
from langchain_core.tools import tool

PK_ROOT = Path(__file__).parent.parent.parent.parent / "pulseknowledge"
sys.path.insert(0, str(PK_ROOT))

from config import TOP_K_RETRIEVE, CONTEXT_WINDOW

_index = None
_chunks = None

def _get_index():
    global _index, _chunks
    if _index is not None:
        return _index
    print("[retriever] Loading WixQA corpus (6,221 articles)...")
    from src.corpus.wixqa_adapter import load_kb_articles, articles_to_corpus_chunks
    from src.retrieval.indexer import RetrievalIndex
    articles = load_kb_articles()
    _chunks = articles_to_corpus_chunks(articles)
    print(f"[retriever] Building hybrid index over {len(_chunks)} chunks...")
    _index = RetrievalIndex(_chunks)
    _index.build(verbose=False)
    print("[retriever] Index ready.")
    return _index

@tool
def retrieve_chunks(query: str) -> List[dict]:
    """Retrieve top-K evidence chunks from WixQA corpus using hybrid RRF (BM25+dense+RRF k=60)."""
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
'''

files["src/tools/nli_tool.py"] = '''"""
nli_tool.py — Bridge to PulseKnowledge NLI citation layer
Strict policy: NLI confidence >= 0.85, no CONTRADICTS. Measured precision: 75.9%
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
from langchain_core.tools import tool

PK_ROOT = Path(__file__).parent.parent.parent.parent / "pulseknowledge"
sys.path.insert(0, str(PK_ROOT))

from config import TOP_K_VERIFY, NLI_CONFIDENCE_THRESHOLD

@tool
def verify_citations(query: str, chunks: List[dict]) -> dict:
    """
    NLI citation check using cross-encoder/nli-deberta-v3-small ONNX (17ms/pair).
    Strict policy: confidence >= 0.85, no CONTRADICTS allowed.
    Returns: contract_decision in ANSWER_WITH_CITATION | ABSTAIN | BLOCK | ESCALATE
    """
    try:
        from src.citation.answer_contract import AnswerContractRouter
        router = AnswerContractRouter(policy="strict")
        result = router.evaluate(query=query, chunks=chunks[:TOP_K_VERIFY])
        return {
            "verified_chunks":    getattr(result, "verified_chunks", []),
            "contract_decision":  getattr(result, "decision", "ABSTAIN"),
            "citation_precision": getattr(result, "citation_precision", None),
        }
    except Exception as e:
        return {
            "verified_chunks":    [],
            "contract_decision":  "ABSTAIN",
            "citation_precision": None,
            "error": str(e),
        }
'''

files["src/agent/nodes.py"] = '''"""
nodes.py — LangGraph node functions
Graph: planner → retriever → nli_verifier → generator → reflector → [END or retry]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL, MAX_RETRIES, TOP_K_VERIFY
from src.agent.state import AgentState
from src.tools.retriever_tool import retrieve_chunks
from src.tools.nli_tool import verify_citations

def _llm():
    return ChatOpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY,
                      model=LM_STUDIO_MODEL, temperature=0.2)

def planner_node(state: AgentState) -> dict:
    """Decomposes query into sub-queries and identifies search intent. (JD: task decomposition, planning)"""
    resp = _llm().invoke([HumanMessage(content=f"""Decompose this query for retrieval.
Query: {state["query"]}
Output format:
INTENT: <one sentence>
SUB_QUERIES:
- <query 1>
- <query 2>
- <query 3>""")])
    intent, sub_queries = "", [state["query"]]
    for line in resp.content.split("\\n"):
        if line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip()
        elif line.strip().startswith("- "):
            sub_queries.append(line.strip()[2:])
    return {"search_intent": intent, "decomposed_queries": sub_queries[:3] or [state["query"]]}

def retriever_node(state: AgentState) -> dict:
    """Hybrid RRF retrieval per sub-query, dedup by article_id. (JD: RAG pipelines, vector DB)"""
    seen, all_chunks = set(), []
    for q in (state.get("decomposed_queries") or [state["query"]]):
        for chunk in retrieve_chunks.invoke({"query": q}):
            if chunk["article_id"] not in seen:
                seen.add(chunk["article_id"])
                all_chunks.append(chunk)
    all_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
    return {"retrieved_chunks": all_chunks[:10]}

def nli_verifier_node(state: AgentState) -> dict:
    """NLI citation gate — strict policy. (JD: evaluation frameworks, probabilistic testing)"""
    result = verify_citations.invoke({"query": state["query"],
                                      "chunks": state["retrieved_chunks"][:TOP_K_VERIFY]})
    return {
        "verified_chunks":    result.get("verified_chunks", []),
        "contract_decision":  result.get("contract_decision", "ABSTAIN"),
        "citation_precision": result.get("citation_precision"),
    }

def generator_node(state: AgentState) -> dict:
    """LLM answer generation grounded in verified chunks only. (JD: LLM integration, function calling)"""
    if state.get("contract_decision") != "ANSWER_WITH_CITATION":
        return {
            "draft_answer": "",
            "final_answer": f"[{state.get('contract_decision','ABSTAIN')}] Cannot provide a verified answer.",
            "cited_article_ids": [],
            "route": state.get("contract_decision", "ABSTAIN"),
        }
    chunks = state.get("verified_chunks") or state.get("retrieved_chunks", [])
    context = "\\n\\n".join(f"[Source {i+1} | ID: {c[\'article_id\']}]\\n{c[\'text\']}"
                             for i, c in enumerate(chunks[:3]))
    resp = _llm().invoke([HumanMessage(content=f"""Answer using ONLY these verified sources.
QUERY: {state["query"]}
SOURCES:\\n{context}
Rules: cite inline as [Source N], no info beyond sources.""")])
    return {"draft_answer": resp.content, "cited_article_ids": [c["article_id"] for c in chunks[:3]]}

def reflector_node(state: AgentState) -> dict:
    """Self-critique loop — PASS or RETRY. (JD: self-reflective behavior loops)"""
    if not state.get("draft_answer"):
        return {"reflection_passed": True, "reflection_notes": "No answer (non-ANSWER route).",
                "final_answer": state.get("final_answer", ""), "route": state.get("contract_decision", "ABSTAIN")}
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return {"reflection_passed": True, "reflection_notes": "Max retries reached.",
                "final_answer": state["draft_answer"], "route": "ANSWER_WITH_CITATION"}
    resp = _llm().invoke([HumanMessage(content=f"""Review this answer.
QUERY: {state["query"]}
ANSWER: {state["draft_answer"]}
Is it grounded, specific, and answers the query? Reply PASS or RETRY, then REASON: <one sentence>""")])
    passed = resp.content.strip().upper().startswith("PASS")
    if passed:
        return {"reflection_passed": True, "reflection_notes": resp.content,
                "final_answer": state["draft_answer"], "route": "ANSWER_WITH_CITATION"}
    return {"reflection_passed": False, "reflection_notes": resp.content,
            "retry_count": state.get("retry_count", 0) + 1}
'''

files["src/agent/graph.py"] = '''"""
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
'''

files["main.py"] = '''"""
main.py — PulseAgent entry point
Usage:
  python3 main.py "how do I add a blog?"   # single query
  python3 main.py                           # interactive
"""
import sys

def run_query(query: str):
    from src.agent.graph import run_agent
    print(f"\\nQuery: {query}")
    print("Running agent...\\n")
    result = run_agent(query)
    print(f"Route:  {result[\'route\']}")
    print(f"Answer: {result[\'final_answer\']}")
    if result["cited_ids"]:
        print(f"Cited:  {result[\'cited_ids\'][:3]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]))
    else:
        print("PulseAgent interactive. Ctrl+C to exit.\\n")
        while True:
            try:
                q = input("Query > ").strip()
                if q: run_query(q)
            except (KeyboardInterrupt, EOFError):
                break
'''

files["requirements.txt"] = """langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
fastmcp>=0.1.0
qdrant-client>=1.7.0
fastembed>=0.2.0
rank-bm25>=0.2.2
sentence-transformers>=2.7.0
onnxruntime>=1.17.0
datasets>=2.18.0
numpy>=1.24.0
rich>=13.0.0
"""

# Write all files
for rel_path, content in files.items():
    fpath = ROOT / rel_path
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(content)
    print(f"  created: {rel_path}")

# Create output dirs
(ROOT / "outputs" / "evidence").mkdir(parents=True, exist_ok=True)
(ROOT / "docs").mkdir(exist_ok=True)
print("\nDone. Run: python3 main.py \"how do I add a blog to my Wix site?\"")
