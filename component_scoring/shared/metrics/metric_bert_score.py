# component_scoring/shared/metrics/metric_bert_score.py
# BERTScore wrapper for semantic similarity scoring.
# Requires: pip install bert-score
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default model for BERTScore.  "roberta-large" gives the best scores but is
# slow; "distilbert-base-uncased" is much faster and adequate for ranking.
_DEFAULT_BERT_MODEL = "roberta-large"


def bert_score_f1(
    candidate: str,
    expected: str,
    model: str = _DEFAULT_BERT_MODEL,
) -> float:
    """Compute BERTScore F1 between candidate and expected text.

    Uses contextual embeddings to measure semantic similarity rather than
    surface token overlap.  Paraphrase-correct outputs score near 1.0 even
    when they share no literal tokens with the expected answer.

    Returns float in [0, 1] (higher = more semantically similar).
    Returns 0.5 on import error (graceful degradation: caller can detect by
    checking whether all scores equal 0.5).

    Parameters
    ----------
    candidate:
        The model's candidate output text.
    expected:
        The reference / expected answer text.
    model:
        HuggingFace model identifier for BERTScore (default: roberta-large).
    """
    if not candidate.strip() or not expected.strip():
        return 0.0

    try:
        from bert_score import score as bs_score  # type: ignore[import]
    except ImportError:
        logger.warning(
            "bert-score not installed; returning 0.5 placeholder.  "
            "Install with: pip install bert-score"
        )
        return 0.5

    try:
        # bert_score.score returns (P, R, F1) tensors
        _, _, f1 = bs_score(
            [candidate],
            [expected],
            model_type=model,
            verbose=False,
        )
        return float(f1[0].item())
    except Exception as exc:
        logger.warning("bert_score_f1 failed: %s", exc)
        return 0.5
