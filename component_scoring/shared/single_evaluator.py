# recommendation_matrix/shared/single_evaluator.py
# SingleEvaluator: evaluates one component against all its (query, answer) pairs.
from __future__ import annotations

import asyncio
import logging
import random

from openjiuwen.core.foundation.llm import Model, SystemMessage, UserMessage

from .metrics.metric_bag_of_words import bag_of_words
from .metrics.metric_bigram_f1 import bigram_f1
from .metrics.metric_length_ratio import length_ratio
from .metrics.metric_token_f1 import token_f1
from .metrics.metrics_list import FITNESS_METRICS

logger = logging.getLogger(__name__)


class SingleEvaluator:
    """Evaluate how helpful a component is for a single prompt.

    Given:
      - component_body: the component content (SKILL.md body, memory section, tool description)
      - query: a user prompt
      - expected: the expected/desired output

    Runs one LLM call with the component body as system message and the query as user message.

    Optional semantic metrics can be enabled:
    - ``"bert_score"`` — requires ``pip install bert-score``
    - ``"llm_judge"`` — requires ``judge_api_key`` (or OPENAI_API_KEY env var)

    Combination scoring mode (``eval_combination_size > 1``):
        Instead of evaluating each component in isolation, assembles random N-component
        combinations and estimates each component's contribution by leave-one-out delta.
        This detects components that are only useful in combination with others.
    """

    def __init__(
        self,
        model: Model,
        model_name: str,
        timeout: float = 3600.0,
        temperature: float = 0.2,
        max_tokens: int = 57000,
        metrics: list[str] | None = None,
        judge_model: str = "gpt-4o-mini",
        judge_api_key: str | None = None,
        judge_api_base: str = "https://api.openai.com/v1",
        eval_combination_size: int = 1,
    ):
        self._model = model
        self._model_name = model_name
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._metrics = metrics if metrics is not None else FITNESS_METRICS
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_api_base = judge_api_base
        self._eval_combination_size = max(1, eval_combination_size)

    # ── batch evaluation ───────────────────────────────────────────────────────

    async def evaluate_component(
        self,
        component_body: str,
        pairs: list[dict],
        sem: asyncio.Semaphore,
        all_component_bodies: list[str] | None = None,
    ) -> list[dict]:
        """Evaluate all (query, answer) pairs for one component in parallel.

        Parameters
        ----------
        component_body:
            The component's text body (system prompt).
        pairs:
            List of {"query": ..., "answer": ...} dicts.
        sem:
            Semaphore for concurrency control.
        all_component_bodies:
            When provided and ``eval_combination_size > 1``, used to assemble
            random N-component combinations for leave-one-out contribution scoring.
        """
        if self._eval_combination_size > 1 and all_component_bodies:
            return await self._evaluate_combination_mode(
                component_body, pairs, sem, all_component_bodies
            )

        async def eval_one(pair: dict) -> dict:
            async with sem:
                return await self.evaluate_pair(
                    component_body=component_body,
                    query=pair["query"],
                    expected=pair["answer"],
                )

        return list(await asyncio.gather(*[eval_one(p) for p in pairs]))

    # Backward-compatible alias
    async def evaluate_skill(
        self,
        skill_body: str,
        pairs: list[dict],
        sem: asyncio.Semaphore,
    ) -> list[dict]:
        return await self.evaluate_component(skill_body, pairs, sem)

    async def evaluate_pair(self, component_body: str, query: str, expected: str) -> dict:
        """Run one (component, query) pair through the LLM and score the result."""
        actual = await self._invoke(component_body, query)
        scores = self._compute_scores(actual, expected, query=query)
        return {
            "example_input": query,
            "example_expected": expected,
            "candidate_output": actual,
            "scores": scores,
        }

    # ── combination mode ───────────────────────────────────────────────────────

    async def _evaluate_combination_mode(
        self,
        component_body: str,
        pairs: list[dict],
        sem: asyncio.Semaphore,
        all_component_bodies: list[str],
    ) -> list[dict]:
        """Leave-one-out contribution scoring within N-component combinations.

        For each pair:
        1. Sample (N-1) random other components.
        2. Evaluate the combination WITH component_body.
        3. Evaluate the combination WITHOUT component_body.
        4. Contribution = score_with - score_without (delta).

        The returned rows use the contribution as the effective score.
        """
        N = min(self._eval_combination_size, len(all_component_bodies) + 1)
        others = [b for b in all_component_bodies if b != component_body]

        async def eval_one(pair: dict) -> dict:
            async with sem:
                # Pick N-1 random other components
                sample = random.sample(others, min(N - 1, len(others)))

                # System prompt WITH component_body
                body_with = "\n\n---\n\n".join([component_body] + sample)
                output_with = await self._invoke(body_with, pair["query"])
                scores_with = self._compute_scores(output_with, pair["answer"], query=pair["query"])

                # System prompt WITHOUT component_body
                body_without = "\n\n---\n\n".join(sample) if sample else ""
                if body_without:
                    output_without = await self._invoke(body_without, pair["query"])
                    scores_without = self._compute_scores(output_without, pair["answer"], query=pair["query"])
                else:
                    # No other components — treat baseline as 0
                    scores_without = {m: 0.0 for m in scores_with}

                # Contribution delta (clamped to [0, 1])
                contribution_scores = {
                    m: max(0.0, min(1.0, scores_with[m] - scores_without[m]))
                    for m in scores_with
                }
                return {
                    "example_input": pair["query"],
                    "example_expected": pair["answer"],
                    "candidate_output": output_with,
                    "scores": contribution_scores,
                    "combination_delta": True,
                    "n_combination_size": N,
                }

        return list(await asyncio.gather(*[eval_one(p) for p in pairs]))

    # ── scoring ────────────────────────────────────────────────────────────────

    def _compute_scores(
        self,
        candidate_output: str,
        expected: str,
        query: str = "",
    ) -> dict[str, float]:
        """Compute scores for the configured set of metrics.

        Parameters
        ----------
        candidate_output:
            The model's text output to score.
        expected:
            The reference / expected answer.
        query:
            Original query (used by llm_judge metric for context).
        """
        scores: dict[str, float] = {}

        for metric in self._metrics:
            if metric == "f1":
                scores["f1"] = token_f1(candidate_output, expected)
            elif metric == "bigram_f1":
                scores["bigram_f1"] = bigram_f1(candidate_output, expected)
            elif metric == "bag_of_words":
                scores["bag_of_words"] = bag_of_words(candidate_output, expected)
            elif metric == "length_ratio":
                scores["length_ratio"] = length_ratio(candidate_output, expected)
            elif metric == "bert_score":
                from .metrics.metric_bert_score import bert_score_f1
                scores["bert_score"] = bert_score_f1(candidate_output, expected)
            elif metric == "llm_judge":
                from .metrics.metric_llm_judge import llm_judge_score
                scores["llm_judge"] = llm_judge_score(
                    candidate_output,
                    expected,
                    query=query,
                    model=self._judge_model,
                    api_key=self._judge_api_key,
                    api_base=self._judge_api_base,
                )
            else:
                logger.warning("Unknown metric %r — skipping.", metric)

        return scores

    # ── LLM invocation ─────────────────────────────────────────────────────────

    async def _invoke(self, component_body: str, query: str) -> str:
        """Run one (component, query) pair through the LLM. Returns response text."""
        try:
            response = await asyncio.wait_for(
                self._model.invoke(
                    [SystemMessage(content=component_body), UserMessage(content=query)],
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                ),
                timeout=self._timeout,
            )
            return (response.content or "").strip()
        except asyncio.TimeoutError:
            logger.warning("Execution timed out: query=%r", query[:60])
            return ""
        except Exception as e:
            logger.warning("Execution error: %s", e)
            return ""


# Backward-compatible alias
SingleSkillEvaluator = SingleEvaluator
