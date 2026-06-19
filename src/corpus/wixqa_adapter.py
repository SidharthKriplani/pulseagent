"""
wixqa_adapter.py — Load WixQA from HuggingFace for PulseKnowledge G6.3
PulseKnowledge · Gate G6.3 · Primary Corpus Ingestion

Corpus: Wix/WixQA (MIT license, arXiv:2505.08643)
  wix_kb_corpus       — 6,221 Wix Help Center articles
  wixqa_expertwritten — 200 expert-authored QA pairs (primary eval set)
  wixqa_simulated     — 200 simulated QA pairs
  wixqa_synthetic     — 6,221 LLM-generated QA pairs

Observed article schema:
  id           : str  (SHA-256 of article URL path — matches article_ids in QA pairs)
  url          : str  (https://support.wix.com/en/article/...)
  contents     : str  (plain text — title line + article body)
  title        : str  (article title, also appears as first line of contents)
  html_content : str  (not used — PulseKnowledge works on plain text)
  article_type : str  (article | feature_request | known_issue)

Observed QA schema:
  question    : str
  answer      : str  (may contain markdown links)
  article_ids : list[str]  (ground-truth source article IDs)

Design notes:
  - Adapter produces article-level WixQAArticle objects and CorpusChunk objects.
  - One CorpusChunk per article for retrieval indexing (article-level BM25 + dense).
  - Paragraph-level chunking is available separately for citation/NLI evaluation.
  - The existing chunk_document() from src/ingestion/chunker.py is NOT used here
    because it assumes a synthetic document header (===...===). WixQA articles have
    no such header — using that chunker would strip the first 20 lines of content.
"""

from __future__ import annotations

import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Chunking thresholds for WixQA article paragraph chunking
WIXQA_MIN_CHUNK_CHARS = 80
WIXQA_MAX_CHUNK_CHARS = 1800

# HuggingFace dataset path and config names
HF_DATASET_PATH = "Wix/WixQA"
KB_CONFIG       = "wix_kb_corpus"
EW_CONFIG       = "wixqa_expertwritten"
SIM_CONFIG      = "wixqa_simulated"
SYN_CONFIG      = "wixqa_synthetic"

CORPUS_TAG = "WIXQA"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WixQAArticle:
    """A single Wix Help Center article from the WixQA KB corpus."""
    article_id: str          # SHA-256 ID (matches article_ids in QA pairs)
    title: str
    url: str
    text: str                # plain text (contents field)
    article_type: str        # article | feature_request | known_issue
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class WixQAQuery:
    """A single QA pair from WixQA-ExpertWritten (or simulated/synthetic)."""
    question: str
    answer: str
    article_ids: list[str]   # ground-truth article IDs (1 or more)
    query_index: int = 0


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------

def load_kb_articles(max_articles: Optional[int] = None) -> list[WixQAArticle]:
    """
    Load WixQA KB articles from HuggingFace (Wix/WixQA, wix_kb_corpus).

    Parameters
    ----------
    max_articles : int or None
        If set, limit to first N articles (useful for testing).

    Returns
    -------
    list[WixQAArticle]
        All articles with metadata.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` library required. "
            "Install with: pip install datasets --break-system-packages"
        ) from e

    logger.info("Loading WixQA KB corpus from HuggingFace (%s, %s)...", HF_DATASET_PATH, KB_CONFIG)
    ds = load_dataset(HF_DATASET_PATH, KB_CONFIG, split="train")

    articles = []
    for i, row in enumerate(ds):
        if max_articles is not None and i >= max_articles:
            break
        articles.append(WixQAArticle(
            article_id   = row["id"],
            title        = row.get("title", "").strip(),
            url          = row.get("url", ""),
            text         = row.get("contents", "").strip(),
            article_type = row.get("article_type", "article"),
        ))

    logger.info("Loaded %d KB articles", len(articles))
    return articles


def load_qa_pairs(config: str = EW_CONFIG) -> list[WixQAQuery]:
    """
    Load WixQA QA pairs from HuggingFace.

    Parameters
    ----------
    config : str
        One of: 'wixqa_expertwritten', 'wixqa_simulated', 'wixqa_synthetic'

    Returns
    -------
    list[WixQAQuery]
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` library required."
        ) from e

    logger.info("Loading WixQA QA pairs from HuggingFace (%s, %s)...", HF_DATASET_PATH, config)
    ds = load_dataset(HF_DATASET_PATH, config, split="train")

    queries = []
    for i, row in enumerate(ds):
        # article_ids may be stored as a list[str] or as a string repr of a list
        raw_ids = row.get("article_ids", [])
        if isinstance(raw_ids, str):
            # Parse string repr: "['id1', 'id2']"
            raw_ids = re.findall(r"'([a-f0-9]{64})'", raw_ids)
        queries.append(WixQAQuery(
            question    = row.get("question", "").strip(),
            answer      = row.get("answer", "").strip(),
            article_ids = list(raw_ids),
            query_index = i,
        ))

    logger.info("Loaded %d QA pairs from config '%s'", len(queries), config)
    return queries


# ---------------------------------------------------------------------------
# Article-level CorpusChunk production (for retrieval indexing)
# ---------------------------------------------------------------------------

def article_to_corpus_chunk(article: WixQAArticle, chunk_index: int = 0):
    """
    Convert a WixQAArticle to a single CorpusChunk (article-level).

    One chunk per article — used for article-level BM25 and dense retrieval.
    The BM25 index text combines the title with the full article body.

    Returns a CorpusChunk from src/retrieval/corpus.py.
    """
    try:
        from src.retrieval.corpus import CorpusChunk
    except ModuleNotFoundError:
        from retrieval.corpus import CorpusChunk

    # Index text: title + body (gives BM25 the title keywords as well)
    index_text = article.title + "\n\n" + article.text if article.title else article.text

    return CorpusChunk(
        chunk_id           = f"wixqa_{article.article_id[:16]}",
        text               = index_text,
        source             = article.url,
        page               = 1,
        doc_type           = article.article_type,
        version            = "",
        effective_date     = "",
        title              = article.title,
        section_heading    = "Document",
        chunk_index        = chunk_index,
        corpus_tag         = CORPUS_TAG,
        corpus_source_file = KB_CONFIG,
        metadata           = {
            "article_id":   article.article_id,
            "article_type": article.article_type,
            "url":          article.url,
            "char_count":   article.char_count,
        },
    )


def articles_to_corpus_chunks(articles: list[WixQAArticle]) -> list:
    """
    Convert all WixQA articles to article-level CorpusChunks for retrieval indexing.
    Returns list[CorpusChunk].
    """
    return [article_to_corpus_chunk(art, i) for i, art in enumerate(articles)]


# ---------------------------------------------------------------------------
# Paragraph-level chunking (for NLI citation evaluation)
# ---------------------------------------------------------------------------

def chunk_article_paragraphs(article: WixQAArticle) -> list[dict]:
    """
    Split a WixQA article into paragraph-level chunks for NLI citation checking.

    Unlike the retrieval index (article-level), the NLI entailment checker
    works best on focused passages (not the full article text). This function
    produces sub-article passages suitable for entailment evaluation.

    Returns list of dicts with keys: text, article_id, title, chunk_index.
    """
    if not article.text.strip():
        return []

    # Split on paragraph boundaries (double newlines)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", article.text) if p.strip()]

    # Merge very short paragraphs with the preceding one
    merged: list[str] = []
    for para in paragraphs:
        if merged and len(para) < WIXQA_MIN_CHUNK_CHARS:
            combined = merged[-1] + "\n\n" + para
            if len(combined) <= WIXQA_MAX_CHUNK_CHARS:
                merged[-1] = combined
                continue
        merged.append(para)

    # Split overlength paragraphs at sentence boundary
    chunks: list[str] = []
    for para in merged:
        if len(para) > WIXQA_MAX_CHUNK_CHARS:
            chunks.extend(_split_long_text(para))
        else:
            chunks.append(para)

    # Filter noise (< MIN_CHUNK_CHARS)
    chunks = [c for c in chunks if len(c) >= WIXQA_MIN_CHUNK_CHARS]

    return [
        {
            "text":        chunk_text,
            "article_id":  article.article_id,
            "title":       article.title,
            "chunk_index": i,
            "chunk_id":    str(uuid.uuid4()),
        }
        for i, chunk_text in enumerate(chunks)
    ]


def _split_long_text(text: str) -> list[str]:
    """Split at sentence boundary for paragraphs > WIXQA_MAX_CHUNK_CHARS."""
    sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_end.split(text)
    parts: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= WIXQA_MAX_CHUNK_CHARS:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sent
    if current:
        parts.append(current)
    return parts or [text[:WIXQA_MAX_CHUNK_CHARS]]


# ---------------------------------------------------------------------------
# Ingestion QA stats
# ---------------------------------------------------------------------------

def compute_ingestion_stats(articles: list[WixQAArticle]) -> dict:
    """
    Compute ingestion QA statistics for the WixQA KB corpus.

    Returns a dict suitable for serialization to g6_3_wixqa_ingestion_report.json.
    """
    from collections import Counter

    total_articles = len(articles)
    char_counts    = [a.char_count for a in articles]
    type_counts    = Counter(a.article_type for a in articles)

    empty_articles = sum(1 for a in articles if a.char_count < WIXQA_MIN_CHUNK_CHARS)
    very_short     = sum(1 for a in articles if 0 < a.char_count < 200)

    # Paragraph chunk stats
    all_chunks: list[dict] = []
    for article in articles:
        all_chunks.extend(chunk_article_paragraphs(article))

    total_chunks   = len(all_chunks)
    chunk_lengths  = [len(c["text"]) for c in all_chunks]

    return {
        "corpus":                  "Wix/WixQA (wix_kb_corpus)",
        "corpus_tag":              CORPUS_TAG,
        "total_articles":          total_articles,
        "by_article_type":         dict(type_counts),
        "empty_articles":          empty_articles,
        "very_short_articles_lt200": very_short,
        "char_count_avg":          round(sum(char_counts) / max(total_articles, 1), 1),
        "char_count_median":       sorted(char_counts)[len(char_counts) // 2] if char_counts else 0,
        "char_count_max":          max(char_counts, default=0),
        "char_count_min":          min(char_counts, default=0),
        "total_corpus_chars":      sum(char_counts),
        "total_paragraph_chunks":  total_chunks,
        "avg_chunks_per_article":  round(total_chunks / max(total_articles, 1), 2),
        "chunk_char_avg":          round(sum(chunk_lengths) / max(total_chunks, 1), 1),
        "chunk_char_max":          max(chunk_lengths, default=0),
        "chunk_char_min":          min(chunk_lengths, default=0),
        "ingestion_pipeline":      "src/corpus/wixqa_adapter.py :: article_to_corpus_chunk()",
        "index_unit":              "article-level (one CorpusChunk per article for retrieval)",
        "citation_unit":           "paragraph-level chunks (used for NLI citation evaluation)",
        "ingestion_complete":      True,
    }
