from __future__ import annotations
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
