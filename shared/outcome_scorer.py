# online/outcome_scorer.py
# Compute a scalar outcome quality [0, 1] from a logged turn's signals.
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b([1-9]|10)\b")


def compute_outcome_quality(turn: dict) -> float:
    """Return a quality score in [0, 1] for a turn's outcome.

    Signal priority:
    1. Explicit user rating (strongest) — "positive" → 1.0, "negative" → 0.0
    2. LLM judge score (if stored in outcome["llm_judge_score"]) — already in [0, 1]
    3. Implicit signals (additive on a 0.5 baseline):
       +0.2  task_completed
       -0.3  follow_up_correction
       +max(0, 0.1 - 0.02 × conversation_length)   (shorter = better)

    Returns a value clamped to [0, 1].
    """
    outcome = turn.get("outcome", {})
    rating = outcome.get("explicit_rating")

    # Priority 1: explicit user rating
    if rating == "positive":
        return 1.0
    if rating == "negative":
        return 0.0

    # Priority 2: LLM judge score (stored by update_outcome when judge was available)
    judge_score = outcome.get("llm_judge_score")
    if judge_score is not None:
        return float(max(0.0, min(1.0, judge_score)))

    # Priority 3: implicit signals
    signals = outcome.get("implicit_signals", {})
    score = 0.5

    if signals.get("task_completed", False):
        score += 0.2
    if signals.get("follow_up_correction", False):
        score -= 0.3

    length = signals.get("conversation_length", 1)
    score += max(0.0, 0.1 - 0.02 * length)

    return max(0.0, min(1.0, score))


def score_from_llm_judge(
    query: str,
    agent_output: str,
    reference_output: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str = "https://api.openai.com/v1",
    timeout: float = 30.0,
) -> float | None:
    """Call an LLM judge to score the agent's output against a reference.

    The judge rates semantic correctness on a 1–10 scale.  The result is
    normalised to [0, 1] so it can be stored directly as ``llm_judge_score``
    in a turn record and consumed by ``compute_outcome_quality``.

    Parameters
    ----------
    query:
        The original user query.
    agent_output:
        The agent's response to evaluate.
    reference_output:
        A reference or expected answer to compare against.
    model:
        OpenAI-compatible model to use as judge (default: gpt-4o-mini).
    api_key:
        API key.  Falls back to the OPENAI_API_KEY environment variable.
    api_base:
        API base URL (default: https://api.openai.com/v1).
    timeout:
        HTTP request timeout in seconds (default: 30).

    Returns
    -------
    float in [0, 1] or None if the judge call fails.
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        logger.warning(
            "score_from_llm_judge: no API key provided and OPENAI_API_KEY not set; returning None."
        )
        return None

    prompt = (
        "You are evaluating the quality of an AI agent's response.\n\n"
        f'User query: "{query}"\n\n'
        f'Reference answer: "{reference_output}"\n\n'
        f'Agent response: "{agent_output}"\n\n'
        "Rate the agent's response for semantic correctness and completeness "
        "relative to the reference answer.\n"
        "A score of 10 means the response is semantically equivalent or better.\n"
        "A score of 1 means the response is completely wrong or irrelevant.\n"
        "Reply with only a single integer from 1 to 10."
    )

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        match = _SCORE_RE.search(text)
        if match:
            raw = float(match.group(1))
            return (raw - 1.0) / 9.0  # normalise [1, 10] → [0, 1]
        logger.warning("score_from_llm_judge: could not parse score from %r", text)
        return None
    except Exception as exc:
        logger.warning("score_from_llm_judge: request failed: %s", exc)
        return None


class OutcomeScorer:
    """Wrapper around compute_outcome_quality / score_from_llm_judge for batch use."""

    def __init__(
        self,
        *,
        judge_model: str = "gpt-4o-mini",
        judge_api_key: str | None = None,
        judge_api_base: str = "https://api.openai.com/v1",
    ):
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_api_base = judge_api_base

    def score(self, turn: dict) -> float:
        return compute_outcome_quality(turn)

    def score_batch(self, turns: list[dict]) -> list[float]:
        return [compute_outcome_quality(t) for t in turns]

    def judge(
        self,
        query: str,
        agent_output: str,
        reference_output: str,
        timeout: float = 30.0,
    ) -> float | None:
        """Convenience wrapper around score_from_llm_judge using this scorer's config."""
        return score_from_llm_judge(
            query,
            agent_output,
            reference_output,
            model=self._judge_model,
            api_key=self._judge_api_key,
            api_base=self._judge_api_base,
            timeout=timeout,
        )
