# component_scoring/shared/metrics/metric_llm_judge.py
# LLM-judge semantic correctness metric.
# Makes one OpenAI-compatible chat completion call per (candidate, expected) pair.
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b([1-9]|10)\b")


def llm_judge_score(
    candidate: str,
    expected: str,
    *,
    query: str = "",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str = "https://api.openai.com/v1",
    timeout: float = 30.0,
) -> float:
    """Rate the semantic correctness of a candidate output against a reference.

    Calls an LLM judge that scores 1–10 based on how semantically correct
    and complete the candidate is relative to the expected answer.  The raw
    score is normalised to [0, 1] before returning.

    Returns 0.5 if the API call fails (graceful degradation to neutral score).

    Parameters
    ----------
    candidate:
        The model's candidate output text.
    expected:
        The reference / expected answer text.
    query:
        The original user query (optional context for the judge).
    model:
        OpenAI-compatible model identifier (default: gpt-4o-mini).
    api_key:
        API key; falls back to the OPENAI_API_KEY environment variable.
    api_base:
        API base URL (default: https://api.openai.com/v1).
    timeout:
        HTTP request timeout in seconds (default: 30).
    """
    if not candidate.strip() or not expected.strip():
        return 0.0

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        logger.warning(
            "llm_judge_score: no API key set; returning 0.5 placeholder."
        )
        return 0.5

    context_line = f'Original query: "{query}"\n\n' if query else ""
    prompt = (
        "You are evaluating the semantic correctness of an AI model's output.\n\n"
        f"{context_line}"
        f'Reference answer: "{expected}"\n\n'
        f'Model output: "{candidate}"\n\n'
        "Rate the model's output for semantic correctness and completeness relative "
        "to the reference answer.\n"
        "A score of 10 means the output is semantically equivalent or superior.\n"
        "A score of 1 means the output is completely wrong or irrelevant.\n"
        "Ignore stylistic differences — focus only on factual/semantic correctness.\n"
        "Reply with only a single integer from 1 to 10."
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    }).encode("utf-8")

    url = f"{api_base.rstrip('/')}/chat/completions"
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
            return (raw - 1.0) / 9.0   # normalise [1, 10] → [0, 1]
        logger.warning("llm_judge_score: could not parse score from %r", text)
        return 0.5
    except Exception as exc:
        logger.warning("llm_judge_score: request failed: %s", exc)
        return 0.5
