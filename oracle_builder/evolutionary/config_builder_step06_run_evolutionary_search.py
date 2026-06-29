# oracle_builder/evolutionary/config_builder_step06_run_evolutionary_search.py
# Step 6: run evolutionary search per cluster × budget.
from __future__ import annotations

import logging

import numpy as np

from .evolution.evolutionary_search_runner import EvolutionarySearchRunner
from .evolution.context_genome import ComponentInfo, ContextGenome
from ...shared.query_clusterer import QueryClusterer

logger = logging.getLogger(__name__)


class EvolutionarySearch:
    """Run an evolutionary search for every cluster and budget combination."""

    def __init__(
        self,
        population_size: int,
        n_generations: int,
        mutation_rate: float,
        lambda_: float,
        budgets: dict[str, int],
        validation_config=None,  # ValidationConfig | None
    ):
        self._pop_size = population_size
        self._n_gen = n_generations
        self._mut_rate = mutation_rate
        self._lambda = lambda_
        self._budgets = budgets
        self._validation_config = validation_config

    def run(
        self,
        components: list[ComponentInfo],
        cluster_texts: dict[int, list[str]],
        clusterer: QueryClusterer,
        per_cluster_lambda: dict[int, float] | None = None,
    ) -> list[dict]:
        """Return one result dict per cluster with optimal configs.

        Parameters
        ----------
        per_cluster_lambda:
            Optional mapping from cluster_id → λ value (written by LambdaTuner).
            When provided, the per-cluster value overrides the global self._lambda
            for that cluster's evolutionary search.
        """
        validator = None
        if self._validation_config is not None:
            from .pareto_validator import ParetoValidator
            validator = ParetoValidator(self._validation_config)

        cluster_results: list[dict] = []

        for cluster_id in range(clusterer.n_clusters):
            texts_in_cluster = cluster_texts.get(cluster_id, [])
            centroid = clusterer.centroid(cluster_id)
            optimal_configs: dict[str, dict] = {}

            # Use per-cluster λ when available, fall back to global
            lambda_for_cluster = (
                per_cluster_lambda.get(cluster_id, self._lambda)
                if per_cluster_lambda
                else self._lambda
            )

            for budget_name, max_tokens in self._budgets.items():
                searcher = EvolutionarySearchRunner(
                    components=components,
                    population_size=self._pop_size,
                    n_generations=self._n_gen,
                    mutation_rate=self._mut_rate,
                    lambda_=lambda_for_cluster,
                    max_tokens=max_tokens,
                    seed=cluster_id,
                )
                pareto = searcher.run(centroid)

                if validator is not None and pareto:
                    ranked = validator.validate(pareto, components, texts_in_cluster)
                    config = self._pick_for_budget_validated(
                        ranked, components, max_tokens, centroid
                    )
                else:
                    config = self._pick_for_budget(pareto, components, max_tokens, centroid)

                optimal_configs[f"budget_{budget_name}"] = config

            cluster_results.append({
                "cluster_id":     cluster_id,
                "label":          f"cluster_{cluster_id}",
                "n_queries":      len(texts_in_cluster),
                "example_queries": texts_in_cluster[:5],
                "optimal_configs": optimal_configs,
            })
            logger.info("Cluster %d: %d queries", cluster_id, len(texts_in_cluster))

        return cluster_results

    # ── budget selection ──────────────────────────────────────────────────────

    @staticmethod
    def _pick_for_budget(
        pareto: list[ContextGenome],
        components: list[ComponentInfo],
        max_tokens: int,
        query_embedding: np.ndarray | None = None,
    ) -> dict:
        """Pick the best config within the token budget from a Pareto front.

        Components in the returned config are sorted by individual relevance
        (most relevant first) so callers can apply bookend ordering at runtime.
        """
        if not pareto:
            logger.warning("Empty Pareto front — returning empty config")
            return {"skills": [], "memory": [], "tools": [], "fitness": 0.0, "context_tokens": 0}

        within_budget = [g for g in pareto if g.context_tokens <= max_tokens]
        pool = within_budget if within_budget else pareto
        best = max(pool, key=lambda g: g.fitness if within_budget else -g.context_tokens)

        if query_embedding is not None:
            return best.to_config_sorted(components, query_embedding)
        return best.to_config(components)

    @staticmethod
    def _pick_for_budget_validated(
        ranked: list[tuple[ContextGenome, float]],
        components: list[ComponentInfo],
        max_tokens: int,
        query_embedding: np.ndarray | None = None,
    ) -> dict:
        """Pick the best config from a validated (re-ranked) Pareto list.

        ranked is sorted best-first by combined_score; we honour the token budget.
        """
        if not ranked:
            logger.warning("Empty validated Pareto — returning empty config")
            return {"skills": [], "memory": [], "tools": [], "fitness": 0.0, "context_tokens": 0}

        within_budget = [(g, s) for g, s in ranked if g.context_tokens <= max_tokens]
        pool = within_budget if within_budget else ranked
        best_genome = pool[0][0]  # already sorted best-first

        if query_embedding is not None:
            return best_genome.to_config_sorted(components, query_embedding)
        return best_genome.to_config(components)
