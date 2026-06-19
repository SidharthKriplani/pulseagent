"""
src/citation/entailment.py — NLI-based Citation Entailment Checker
PulseKnowledge · G6 · Citation Entailment Gate

Moves beyond keyword/lexical citation proxy (G4) into entailment-level
citation checking using a cross-encoder NLI model.

Model: cross-encoder/nli-deberta-v3-small
  - Labels: contradiction / entailment / neutral
  - Mapped to: CONTRADICTS / SUPPORTS / NOT_ENOUGH_INFO
  - ~180MB download, CPU inference ~15ms/pair

NLI convention (following SNLI/NLI standard):
  premise   = chunk_text  (the retrieved evidence)
  hypothesis = claim      (the statement to be verified)

Hard boundaries (G6):
  - Do NOT claim this proves full RAG reliability.
  - Do NOT claim NLI is infallible. Model confidence < 0.5 is a low-confidence zone.
  - Do NOT claim this replaces human citation review.
  - lexical keyword proxy (G4) is a different layer — this does not erase it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Confidence threshold below which we degrade to NOT_ENOUGH_INFO regardless
# of highest-scoring label — prevents overconfident CONTRADICTS on ambiguous
# inputs where all label scores are near uniform.
LOW_CONFIDENCE_THRESHOLD = 0.45

# Label order for cross-encoder/nli-deberta-v3-small
# Source: HuggingFace model card + sentence-transformers CrossEncoder label convention
# Verified at runtime via _get_label_map()
_DEFAULT_LABEL_ORDER = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntailmentResult:
    """Full NLI result for a single (claim, chunk) pair."""

    claim: str
    chunk_id: str
    chunk_text_preview: str          # first 150 chars for logging
    verdict: str                     # SUPPORTS | CONTRADICTS | NOT_ENOUGH_INFO
    confidence: float                # probability of winning label after softmax
    label_scores: dict               # {SUPPORTS: float, CONTRADICTS: float, NOT_ENOUGH_INFO: float}
    model_used: str
    low_confidence: bool = False     # True if confidence < LOW_CONFIDENCE_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "chunk_id": self.chunk_id,
            "chunk_text_preview": self.chunk_text_preview,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "label_scores": {k: round(v, 4) for k, v in self.label_scores.items()},
            "model_used": self.model_used,
            "low_confidence": self.low_confidence,
        }


@dataclass
class BatchEntailmentResult:
    """NLI results for one claim against multiple chunks."""

    claim: str
    results: list[EntailmentResult] = field(default_factory=list)

    @property
    def best(self) -> Optional[EntailmentResult]:
        """Chunk with highest SUPPORTS confidence."""
        supports = [r for r in self.results if r.verdict == "SUPPORTS"]
        if supports:
            return max(supports, key=lambda r: r.confidence)
        return self.results[0] if self.results else None

    @property
    def any_contradicts(self) -> bool:
        return any(r.verdict == "CONTRADICTS" for r in self.results)


# ---------------------------------------------------------------------------
# NLI Citation Checker
# ---------------------------------------------------------------------------

class OnnxNLICitationChecker:
    """
    ONNX-runtime NLI citation checker.

    Faster than CrossEncoder in environments where PyTorch is not available or
    the model download would exceed the available timeout.  Uses onnxruntime
    directly with the quantized ONNX export of cross-encoder/nli-deberta-v3-small.

    Usage:
        checker = OnnxNLICitationChecker(model_dir="/path/to/nli_model")
        result = checker.check(claim, chunk_text, chunk_id="c001")

    model_dir must contain:
        model.onnx        — quantized ONNX export (onnx/model_qint8_avx512.onnx from HF)
        tokenizer.json    — DeBERTa tokenizer
        tokenizer_config.json
        spm.model
        config.json       — must have id2label with entailment/contradiction/neutral
        special_tokens_map.json
    """

    def __init__(self, model_dir: str, max_length: int = 512):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime and transformers are required for OnnxNLICitationChecker. "
                "Run: pip install onnxruntime transformers --break-system-packages"
            ) from e

        import json
        from pathlib import Path as _Path

        self._model_dir = str(model_dir)
        self._max_length = max_length
        self._model_name = NLI_MODEL_NAME

        onnx_path = str(_Path(model_dir) / "model.onnx")
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Read label order from config.json
        config_path = _Path(model_dir) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        id2label_raw = config.get("id2label", {})
        self._label_map: dict[int, str] = {}
        for k, v in id2label_raw.items():
            raw = v.lower()
            if "entail" in raw:
                self._label_map[int(k)] = "SUPPORTS"
            elif "contradict" in raw:
                self._label_map[int(k)] = "CONTRADICTS"
            else:
                self._label_map[int(k)] = "NOT_ENOUGH_INFO"

    def check(
        self,
        claim: str,
        chunk_text: str,
        chunk_id: str = "",
    ) -> "EntailmentResult":
        import numpy as np
        from scipy.special import softmax

        enc = self._tokenizer(
            chunk_text,
            claim,
            return_tensors="np",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )
        logits = self._sess.run(
            ["logits"],
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        )[0][0]
        probs = softmax(logits)

        consolidated: dict[str, float] = {
            "SUPPORTS": 0.0,
            "CONTRADICTS": 0.0,
            "NOT_ENOUGH_INFO": 0.0,
        }
        for i, p in enumerate(probs):
            label = self._label_map.get(i, "NOT_ENOUGH_INFO")
            consolidated[label] = consolidated.get(label, 0.0) + float(p)

        verdict = max(consolidated, key=consolidated.get)
        confidence = consolidated[verdict]
        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        if low_confidence and verdict in ("SUPPORTS", "CONTRADICTS"):
            verdict = "NOT_ENOUGH_INFO"

        return EntailmentResult(
            claim=claim,
            chunk_id=chunk_id,
            chunk_text_preview=chunk_text[:150].replace("\n", " ").strip(),
            verdict=verdict,
            confidence=round(confidence, 4),
            label_scores=consolidated,
            model_used=self._model_name + " [onnx]",
            low_confidence=low_confidence,
        )

    def check_batch(
        self,
        claim: str,
        chunks: list,
        text_key: str = "text",
        id_key: str = "chunk_id",
    ) -> "BatchEntailmentResult":
        results = [
            self.check(
                claim,
                chunk.get(text_key, chunk.get("chunk_text", "")),
                chunk.get(id_key, f"chunk_{i}"),
            )
            for i, chunk in enumerate(chunks)
        ]
        return BatchEntailmentResult(claim=claim, results=results)


class NLICitationChecker:
    """
    Cross-encoder NLI citation checker (sentence-transformers backend).

    Usage:
        checker = NLICitationChecker()        # downloads model on first init
        result = checker.check(claim, chunk_text, chunk_id="c001")
        print(result.verdict)  # SUPPORTS | CONTRADICTS | NOT_ENOUGH_INFO

    G6 contract:
      Input:  (claim: str, chunk_text: str, chunk_id: str)
      Output: EntailmentResult with verdict + confidence + per-label scores
      Failure: RuntimeError if model cannot be loaded (environment issue)

    For environments without PyTorch, use OnnxNLICitationChecker instead.
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        max_length: int = 512,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers --break-system-packages"
            ) from e

        self._model_name = model_name
        self._max_length = max_length
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
        )
        self._label_map = self._get_label_map()

    def _get_label_map(self) -> dict[int, str]:
        """
        Map model output index → PulseKnowledge verdict label.

        cross-encoder/nli-deberta-v3-small label order (from model config):
          0 → contradiction → CONTRADICTS
          1 → entailment   → SUPPORTS
          2 → neutral      → NOT_ENOUGH_INFO
        """
        # Try to read label order from model config
        try:
            id2label = self._model.model.config.id2label
            raw_labels = [id2label[i].lower() for i in sorted(id2label.keys())]
        except (AttributeError, KeyError):
            raw_labels = _DEFAULT_LABEL_ORDER

        mapping = {}
        for i, raw in enumerate(raw_labels):
            if "entail" in raw:
                mapping[i] = "SUPPORTS"
            elif "contradict" in raw:
                mapping[i] = "CONTRADICTS"
            else:
                mapping[i] = "NOT_ENOUGH_INFO"
        return mapping

    def check(
        self,
        claim: str,
        chunk_text: str,
        chunk_id: str = "",
    ) -> EntailmentResult:
        """
        Run NLI on a single (chunk_text, claim) pair.

        NLI convention: premise = chunk_text, hypothesis = claim.
        Returns EntailmentResult with SUPPORTS / CONTRADICTS / NOT_ENOUGH_INFO.
        """
        import numpy as np
        from scipy.special import softmax

        raw_scores = self._model.predict(
            [(chunk_text, claim)],
            apply_softmax=False,
        )
        # raw_scores shape: (1, num_labels) or (num_labels,)
        scores_1d = raw_scores[0] if raw_scores.ndim > 1 else raw_scores
        probs = softmax(scores_1d)

        label_scores = {
            self._label_map[i]: float(probs[i])
            for i in range(len(probs))
        }

        # Aggregate multi-index (model may have >3 classes in some configs)
        # — collapse to three canonical verdicts
        consolidated: dict[str, float] = {
            "SUPPORTS": 0.0,
            "CONTRADICTS": 0.0,
            "NOT_ENOUGH_INFO": 0.0,
        }
        for verdict, score in label_scores.items():
            consolidated[verdict] = consolidated.get(verdict, 0.0) + score

        verdict = max(consolidated, key=consolidated.get)
        confidence = consolidated[verdict]
        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        # Downgrade to NOT_ENOUGH_INFO if confidence is too low to commit
        if low_confidence and verdict in ("SUPPORTS", "CONTRADICTS"):
            verdict = "NOT_ENOUGH_INFO"

        return EntailmentResult(
            claim=claim,
            chunk_id=chunk_id,
            chunk_text_preview=chunk_text[:150].replace("\n", " ").strip(),
            verdict=verdict,
            confidence=round(confidence, 4),
            label_scores=consolidated,
            model_used=self._model_name,
            low_confidence=low_confidence,
        )

    def check_batch(
        self,
        claim: str,
        chunks: list[dict],
        text_key: str = "text",
        id_key: str = "chunk_id",
    ) -> BatchEntailmentResult:
        """
        Check a claim against multiple chunks.

        chunks: list of dicts with at least a text field (default key: 'text').
        Returns BatchEntailmentResult with individual results per chunk.
        """
        results = [
            self.check(
                claim,
                chunk.get(text_key, chunk.get("chunk_text", "")),
                chunk.get(id_key, f"chunk_{i}"),
            )
            for i, chunk in enumerate(chunks)
        ]
        return BatchEntailmentResult(claim=claim, results=results)


# ---------------------------------------------------------------------------
# Keyword overlap proxy (G4 baseline — for side-by-side comparison in G6)
# ---------------------------------------------------------------------------

def _tokenize_for_overlap(text: str) -> set[str]:
    """Same tokenizer used in G4 citation.py — for fair apples-to-apples comparison."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def keyword_overlap_verdict(
    claim: str,
    chunk_text: str,
    threshold: float = 0.6,
) -> dict:
    """
    G4-style keyword overlap proxy verdict.

    Returns dict with:
      verdict: SUPPORTED | PARTIAL | PHANTOM | UNVERIFIABLE
      overlap_score: float (Jaccard-like: claim_tokens ∩ chunk_tokens / claim_tokens)
    """
    claim_tokens = _tokenize_for_overlap(claim)
    chunk_tokens = _tokenize_for_overlap(chunk_text)

    if not claim_tokens:
        return {"verdict": "UNVERIFIABLE", "overlap_score": 0.0}

    overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)

    if overlap >= threshold:
        verdict = "SUPPORTED"
    elif overlap > 0.0:
        verdict = "PARTIAL"
    else:
        verdict = "PHANTOM"

    return {"verdict": verdict, "overlap_score": round(overlap, 4)}


# ---------------------------------------------------------------------------
# Numeric Policy Verifier (G6.1 addition)
# ---------------------------------------------------------------------------
#
# NLI alone CANNOT be trusted for numeric/threshold/date policy claims when
# hedging language ("up to", "minimum of", "at least") is present. The
# int8 quantized model returns NOT_ENOUGH_INFO with high confidence on
# hedged numeric claims even when the numbers agree.
#
# This verifier runs BEFORE NLI on any claim that contains a number, and
# its result is fed to CitationRouter to determine whether to escalate.
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\$?[\d,]+(?:\.\d+)?(?:%|k|m|b)?", re.IGNORECASE)
_HEDGE_LOWER = re.compile(
    # Match explicit hedge phrases only — NOT bare "minimum" (which is a policy noun in
    # "minimum length of 12 characters") or bare "maximum".
    r"\b(at least|minimum of|at minimum|at a minimum|no less than|not less than|≥)\b",
    re.IGNORECASE,
)
_HEDGE_UPPER = re.compile(
    r"\b(up to|at most|maximum of|at most|no more than|not more than|no greater than|≤)\b",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text, normalising $1,234 → 1234.0."""
    nums = []
    for tok in _NUMBER_RE.findall(text):
        clean = tok.replace("$", "").replace(",", "").replace("%", "").lower()
        clean = clean.rstrip("kmb")
        try:
            nums.append(float(clean))
        except ValueError:
            pass
    return nums


@dataclass
class NumericVerifierResult:
    """
    Result of deterministic numeric comparison between claim and chunk.

    verdict values:
        NUMERIC_MATCH       — claim number(s) found in chunk; hedging is consistent
        NUMERIC_MISMATCH    — claim states N, chunk states M ≠ N with no covering hedge
        NUMERIC_HEDGED      — numbers differ but chunk hedging may cover claim value;
                              cannot determine match/mismatch without domain knowledge
        NUMERIC_NOT_PRESENT — no numbers in claim; numeric check not applicable
        NUMERIC_UNCERTAIN   — numbers present but cannot be reliably compared
    """

    verdict: str
    claim_numbers: list[float]
    chunk_numbers: list[float]
    hedge_lower_in_chunk: bool
    hedge_upper_in_chunk: bool
    explanation: str

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "claim_numbers": self.claim_numbers,
            "chunk_numbers": self.chunk_numbers,
            "hedge_lower_in_chunk": self.hedge_lower_in_chunk,
            "hedge_upper_in_chunk": self.hedge_upper_in_chunk,
            "explanation": self.explanation,
        }


def numeric_policy_verify(claim: str, chunk_text: str) -> NumericVerifierResult:
    """
    Deterministic numeric comparison for policy claims.

    Used when a claim contains a specific numeric value (dollar amount, rate,
    count, percentage, day-count, etc.) and the chunk may confirm or contradict it.

    Logic:
      1. If no numbers in claim → NOT_PRESENT (NLI handles it alone)
      2. If any claim number appears in chunk → MATCH
      3. If chunk has hedging covering the claim number → HEDGED
      4. If claim number absent and non-matching chunk numbers present → MISMATCH
      5. Otherwise → UNCERTAIN

    Hard rule: on MISMATCH, the CitationRouter must escalate or block —
    NLI alone cannot be trusted for this case.
    """
    claim_nums = _extract_numbers(claim)
    chunk_nums = _extract_numbers(chunk_text)
    hedge_lower = bool(_HEDGE_LOWER.search(chunk_text))
    hedge_upper = bool(_HEDGE_UPPER.search(chunk_text))

    if not claim_nums:
        return NumericVerifierResult(
            verdict="NUMERIC_NOT_PRESENT",
            claim_numbers=[],
            chunk_numbers=chunk_nums,
            hedge_lower_in_chunk=hedge_lower,
            hedge_upper_in_chunk=hedge_upper,
            explanation="No numeric values in claim; NLI handles entailment alone.",
        )

    # Check if any claim number appears exactly in chunk
    for cn in claim_nums:
        if cn in chunk_nums:
            return NumericVerifierResult(
                verdict="NUMERIC_MATCH",
                claim_numbers=claim_nums,
                chunk_numbers=chunk_nums,
                hedge_lower_in_chunk=hedge_lower,
                hedge_upper_in_chunk=hedge_upper,
                explanation=f"Claim number {cn} found in chunk.",
            )

    # Numbers differ — check if hedging covers the claim number
    if chunk_nums and (hedge_lower or hedge_upper):
        # "up to M" covers any claim N where N ≤ M
        # "at least M" covers any claim N where N ≥ M
        for cn in claim_nums:
            for chn in chunk_nums:
                if hedge_upper and cn <= chn:
                    return NumericVerifierResult(
                        verdict="NUMERIC_HEDGED",
                        claim_numbers=claim_nums,
                        chunk_numbers=chunk_nums,
                        hedge_lower_in_chunk=hedge_lower,
                        hedge_upper_in_chunk=hedge_upper,
                        explanation=(
                            f"Claim number {cn} ≤ chunk upper bound {chn} ('up to' hedge). "
                            f"May be consistent — requires domain knowledge to confirm."
                        ),
                    )
                if hedge_lower and cn >= chn:
                    return NumericVerifierResult(
                        verdict="NUMERIC_HEDGED",
                        claim_numbers=claim_nums,
                        chunk_numbers=chunk_nums,
                        hedge_lower_in_chunk=hedge_lower,
                        hedge_upper_in_chunk=hedge_upper,
                        explanation=(
                            f"Claim number {cn} ≥ chunk lower bound {chn} ('at least' hedge). "
                            f"May be consistent — requires domain knowledge to confirm."
                        ),
                    )

    if chunk_nums:
        # Numbers in chunk but none match claim and hedging doesn't cover it
        return NumericVerifierResult(
            verdict="NUMERIC_MISMATCH",
            claim_numbers=claim_nums,
            chunk_numbers=chunk_nums,
            hedge_lower_in_chunk=hedge_lower,
            hedge_upper_in_chunk=hedge_upper,
            explanation=(
                f"Claim number(s) {claim_nums} not found in chunk number(s) {chunk_nums}. "
                f"No covering hedge. Deterministic MISMATCH — escalation required."
            ),
        )

    # Claim has numbers but chunk has none
    return NumericVerifierResult(
        verdict="NUMERIC_UNCERTAIN",
        claim_numbers=claim_nums,
        chunk_numbers=[],
        hedge_lower_in_chunk=hedge_lower,
        hedge_upper_in_chunk=hedge_upper,
        explanation="Claim has numbers but chunk has none — cannot confirm or deny.",
    )


# ---------------------------------------------------------------------------
# Entailment Routing Policy (G6.1 addition)
# ---------------------------------------------------------------------------
#
# Maps (EntailmentResult, NumericVerifierResult) → routing decision.
# This is the policy layer above the NLI classifier.
# ---------------------------------------------------------------------------

@dataclass
class CitationRoutingDecision:
    """
    Routing decision produced by route_citation().

    action values:
        CITE        — citation supports the claim; safe to use
        BLOCK       — citation contradicts the claim; must not cite; surface conflict
        ABSTAIN     — not enough evidence; do not answer from this citation
        CLARIFY     — answer is hedged; recommend clarification with the user
        ESCALATE    — uncertain case requiring stronger model or human review
    """

    action: str
    reason: str
    nli_verdict: str
    numeric_verdict: str
    confidence: float
    requires_human_review: bool = False

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "nli_verdict": self.nli_verdict,
            "numeric_verdict": self.numeric_verdict,
            "confidence": self.confidence,
            "requires_human_review": self.requires_human_review,
        }


def route_citation(
    nli_result: "EntailmentResult",
    numeric_result: Optional["NumericVerifierResult"] = None,
) -> CitationRoutingDecision:
    """
    Apply entailment routing policy.

    Priority order:
      1. Numeric MISMATCH  → BLOCK (deterministic; NLI cannot override)
      2. Numeric HEDGED    → ESCALATE (NLI cannot resolve hedged bounds reliably)
      3. NLI CONTRADICTS   → BLOCK
      4. NLI NOT_ENOUGH_INFO (low confidence) → ABSTAIN
      5. NLI SUPPORTS + Numeric MATCH or NOT_PRESENT → CITE
      6. NLI SUPPORTS + Numeric UNCERTAIN → CLARIFY
      7. NLI low_confidence → ESCALATE
    """
    nv = numeric_result.verdict if numeric_result else "NUMERIC_NOT_PRESENT"
    conf = nli_result.confidence
    nli = nli_result.verdict

    # Rule 1: Deterministic numeric mismatch always blocks — NLI cannot override
    if nv == "NUMERIC_MISMATCH":
        return CitationRoutingDecision(
            action="BLOCK",
            reason="Numeric mismatch detected by deterministic verifier. NLI cannot be trusted for this case.",
            nli_verdict=nli,
            numeric_verdict=nv,
            confidence=conf,
            requires_human_review=True,
        )

    # Rule 2: Hedged numeric — NLI may be unreliable; escalate for human/stronger model
    if nv == "NUMERIC_HEDGED":
        return CitationRoutingDecision(
            action="ESCALATE",
            reason="Hedged numeric bound ('up to N', 'at least N') — NLI cannot reliably resolve; requires domain-aware check.",
            nli_verdict=nli,
            numeric_verdict=nv,
            confidence=conf,
            requires_human_review=True,
        )

    # Rule 3: NLI CONTRADICTS
    if nli == "CONTRADICTS":
        return CitationRoutingDecision(
            action="BLOCK",
            reason="NLI cross-encoder returned CONTRADICTS. Citation must not be used to support this claim.",
            nli_verdict=nli,
            numeric_verdict=nv,
            confidence=conf,
            requires_human_review=(conf < 0.7),
        )

    # Rule 4: NOT_ENOUGH_INFO — abstain
    if nli == "NOT_ENOUGH_INFO":
        if nli_result.low_confidence:
            return CitationRoutingDecision(
                action="ESCALATE",
                reason=f"NLI returned NOT_ENOUGH_INFO with low confidence ({conf:.3f}). Routing to stronger model.",
                nli_verdict=nli,
                numeric_verdict=nv,
                confidence=conf,
                requires_human_review=False,
            )
        return CitationRoutingDecision(
            action="ABSTAIN",
            reason="Retrieved chunk does not provide enough evidence for this claim. Do not answer from this citation.",
            nli_verdict=nli,
            numeric_verdict=nv,
            confidence=conf,
            requires_human_review=False,
        )

    # Rule 5 / 6 / 7: SUPPORTS path
    if nli == "SUPPORTS":
        if nli_result.low_confidence:
            return CitationRoutingDecision(
                action="ESCALATE",
                reason=f"NLI SUPPORTS but confidence {conf:.3f} is below threshold. Escalating for verification.",
                nli_verdict=nli,
                numeric_verdict=nv,
                confidence=conf,
                requires_human_review=False,
            )
        if nv == "NUMERIC_UNCERTAIN":
            return CitationRoutingDecision(
                action="CLARIFY",
                reason="NLI SUPPORTS but claim has numeric values not confirmed in chunk. Recommend clarification.",
                nli_verdict=nli,
                numeric_verdict=nv,
                confidence=conf,
                requires_human_review=False,
            )
        # Clean SUPPORTS with NUMERIC_MATCH or NUMERIC_NOT_PRESENT
        return CitationRoutingDecision(
            action="CITE",
            reason=f"NLI SUPPORTS with confidence {conf:.3f}. Numeric check: {nv}.",
            nli_verdict=nli,
            numeric_verdict=nv,
            confidence=conf,
            requires_human_review=False,
        )

    # Fallback
    return CitationRoutingDecision(
        action="ESCALATE",
        reason=f"Unhandled NLI verdict '{nli}'. Routing for safety.",
        nli_verdict=nli,
        numeric_verdict=nv,
        confidence=conf,
        requires_human_review=True,
    )
