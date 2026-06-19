"""
src/mcp_server/server.py — PulseAgent MCP Tool Server

Exposes PulseAgent's retrieval and NLI layers as MCP tools using fastmcp.
Any MCP-compatible client (Claude Desktop, LangChain MCP adapter, etc.)
can call these tools via the Model Context Protocol.

Tools:
  retrieve_passages(query)           — hybrid BM25+dense RRF over 6,221 WixQA articles
  verify_citation(claim, passage)    — NLI entailment check via cross-encoder

Usage:
  python3 src/mcp_server/server.py       # stdio transport (Claude Desktop)
  python3 src/mcp_server/server.py --sse # SSE transport (web clients)

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "pulseagent": {
        "command": "python3",
        "args": ["/path/to/pulseagent/src/mcp_server/server.py"]
      }
    }
  }
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

PA_ROOT = Path(__file__).parent.parent.parent
if str(PA_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_ROOT))

from fastmcp import FastMCP

mcp = FastMCP(
    name="PulseAgent",
    instructions=(
        "PulseAgent provides hybrid RAG retrieval and NLI citation verification "
        "over a 6,221-article Wix help center knowledge base. "
        "Use retrieve_passages to find relevant articles, then verify_citation "
        "to check if a passage actually supports your claim."
    ),
)


@mcp.tool()
def retrieve_passages(query: str) -> list[dict]:
    """
    Retrieve the most relevant passages from the WixQA corpus using
    hybrid BM25 + dense vector RRF retrieval (BAAI/bge-small-en-v1.5, 384-dim).

    Args:
        query: The question or search query.

    Returns:
        List of dicts with keys: article_id, text, rrf_score, rank, source, heading.
        Ranked by RRF score (higher = more relevant). Top 10 results returned.
    """
    from src.tools.retriever_tool import retrieve_chunks
    return retrieve_chunks.invoke({"query": query})


@mcp.tool()
def verify_citation(claim: str, passage: str) -> dict:
    """
    Verify whether a passage supports a claim using NLI entailment
    (cross-encoder/nli-deberta-v3-small, confidence threshold 0.85).

    IMPORTANT: claim must be a declarative statement, not a question.
    Convert "How do I add a blog?" → "This article provides information about adding a blog."

    Args:
        claim:   Declarative statement to verify.
        passage: Article text to check against the claim.

    Returns:
        Dict with keys:
          verdict    : "SUPPORTS" | "CONTRADICTS" | "NEUTRAL"
          confidence : float 0–1
          passes     : bool (True if verdict==SUPPORTS and confidence>=0.85)
    """
    from src.citation.entailment import NLICitationChecker
    checker = NLICitationChecker()
    result  = checker.check(
        claim=claim,
        chunk_text=passage,
        chunk_id="mcp_verify",
    )
    return {
        "verdict":    result.verdict,
        "confidence": round(float(result.confidence), 4),
        "passes":     result.verdict == "SUPPORTS" and result.confidence >= 0.85,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulseAgent MCP Server")
    parser.add_argument("--sse", action="store_true",
                        help="Run with SSE transport instead of stdio")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port for SSE transport (default: 8765)")
    args = parser.parse_args()

    if args.sse:
        print(f"[mcp_server] Starting SSE server on port {args.port}...")
        mcp.run(transport="sse", port=args.port)
    else:
        print("[mcp_server] Starting stdio server (Claude Desktop mode)...")
        mcp.run(transport="stdio")
