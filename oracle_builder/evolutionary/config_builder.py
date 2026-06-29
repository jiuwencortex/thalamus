# oracle_builder/evolutionary/config_builder.py
# Orchestrates: load matrices → cluster queries → run evolutionary search → write config.
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone

from .config_builder_step01_load_components import ComponentsLoader
from .config_builder_step02_collect_texts import TextsCollector
from .config_builder_step03_fit_clusters import ClusterFitter
from .config_builder_step04_compute_cluster_centroids import ClusterCentroidsComputer
from .config_builder_step05_assign_clusters import ClusterAssigner
from .config_builder_step06_run_evolutionary_search import EvolutionarySearch
from .config_builder_step07_serialize_output import OutputSerializer

logger = logging.getLogger(__name__)


class ContextConfigBuilder:
    """Build context_configs.json from all scoring matrices in oracle_dir."""

    def __init__(
        self,
        oracle_dir: Path,
        n_clusters: int,
        max_features: int,
        population_size: int,
        n_generations: int,
        mutation_rate: float,
        lambda_: float,
        budgets: dict[str, int],
        embedder: str = "tfidf",
        sentence_model: str = "all-MiniLM-L6-v2",
        validation_config=None,  # pareto_validator.ValidationConfig | None
        per_cluster_lambda: dict[int, float] | None = None,
    ):
        self._budgets = budgets
        self._per_cluster_lambda = per_cluster_lambda

        # Step classes instantiated with their persistent dependencies
        self._components_loader = ComponentsLoader(oracle_dir)
        self._texts_collector = TextsCollector()
        self._cluster_fitter = ClusterFitter(n_clusters, max_features, embedder, sentence_model)
        self._centroid_computer = ClusterCentroidsComputer()
        self._cluster_assigner = ClusterAssigner()
        self._evolutionary_runner = EvolutionarySearch(
            population_size, n_generations, mutation_rate, lambda_,
            self._budgets, validation_config=validation_config,
        )
        self._serializer = OutputSerializer()

    def build(self, output_path: Path) -> None:
        """Run the full pipeline and write results to output_path."""
        # Step 1: load components
        components, example_texts_map = self._components_loader.load()
        if not components:
            print("No scoring matrix files found in oracle_dir. Run the matrix builder first.")
            return

        n_loaded = len(components)
        n_skipped = sum(1 for t in example_texts_map.values() if not t)
        print(f"Loaded {n_loaded} component(s)")
        if n_skipped:
            logger.warning("%d component(s) had no example queries and will have zero centroid", n_skipped)

        # Step 2: collect all texts
        all_texts = self._texts_collector.collect(components, example_texts_map)
        if not all_texts:
            print("No example queries found in any matrix file.")
            return

        # Step 3: fit clusterer
        print(f"Clustering {len(all_texts)} example queries into {self._cluster_fitter._n_clusters} group(s)...")
        clusterer = self._cluster_fitter.fit(all_texts)

        # Step 4: compute per-component centroids
        self._centroid_computer.compute(components, example_texts_map, clusterer)

        # Step 5: assign texts to clusters
        cluster_texts = self._cluster_assigner.assign(components, example_texts_map, clusterer)

        # Step 6: run evolutionary search
        validation_note = (
            " + LLM Pareto validation" if self._evolutionary_runner._validation_config else ""
        )
        print(
            f"Running evolutionary search "
            f"({self._evolutionary_runner._n_gen} generations × "
            f"pop={self._evolutionary_runner._pop_size} × "
            f"{clusterer.n_clusters} clusters){validation_note} ..."
        )
        cluster_results = self._evolutionary_runner.run(
            components, cluster_texts, clusterer,
            per_cluster_lambda=self._per_cluster_lambda,
        )

        for r in cluster_results:
            print(f"  Cluster {r['cluster_id']:>2}: {r['n_queries']:>4} queries")

        # Step 7: build result dict and serialize
        result = {"version": 1,
                  "built_at": self.now_iso(),
                  "n_clusters": clusterer.n_clusters,
                  "n_components": n_loaded,
                  "budgets": self._budgets,
                  "embedder": clusterer.embedder,
                  "vocabulary_size": clusterer.n_features,
                  "clusters": cluster_results}
        self._serializer.serialize(result, output_path, clusterer)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
