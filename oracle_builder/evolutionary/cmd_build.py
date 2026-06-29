from __future__ import annotations

import sys
import argparse

from .config_builder import ContextConfigBuilder
from .config_builder_step01_load_components import ComponentsLoader
from .config_builder_step02_collect_texts import TextsCollector


def cmd_build(args: argparse.Namespace) -> None:
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    output = args.output or (args.oracle_dir / "context_configs.json")
    budgets = {"small":  args.budget_small,
               "medium": args.budget_medium,
               "large":  args.budget_large}

    validation_config = None
    if getattr(args, "validate_pareto", False):
        from .pareto_validator import ValidationConfig
        validation_config = ValidationConfig(
            model=getattr(args, "eval_model", "gpt-4o-mini"),
            api_key=getattr(args, "eval_api_key", None),
            api_base=getattr(args, "eval_api_base", "https://api.openai.com/v1"),
            queries_per_cluster=getattr(args, "eval_queries_per_cluster", 3),
        )
        print(
            f"Pareto validation enabled: model={validation_config.model}, "
            f"queries_per_cluster={validation_config.queries_per_cluster}"
        )

    # ── --auto-k: select optimal cluster count from data ─────────────────────
    n_clusters = args.n_clusters
    if getattr(args, "auto_k", False):
        print("Auto-K: loading example texts to select optimal cluster count...")
        loader = ComponentsLoader(args.oracle_dir)
        collector = TextsCollector()
        components, example_texts_map = loader.load()
        all_texts = collector.collect(components, example_texts_map)

        if all_texts:
            from .cluster_count_tuner import ClusterCountTuner
            tuner = ClusterCountTuner(max_features=args.max_features)
            result = tuner.tune(all_texts)
            n_clusters = result.best_k
            print(
                f"Auto-K selected K={n_clusters} "
                f"(method={result.method}) from {len(all_texts)} example texts."
            )
        else:
            print(
                "Auto-K: no example texts found; falling back to --n-clusters "
                f"({n_clusters}).",
                file=sys.stderr,
            )

    # ── --use-cluster-lambda: load per-cluster λ from tune results ────────────
    per_cluster_lambda = None
    if getattr(args, "use_cluster_lambda", False):
        from .lambda_tuner import load_per_cluster_lambda
        per_cluster_lambda = load_per_cluster_lambda(args.oracle_dir)
        if per_cluster_lambda:
            print(
                f"Per-cluster λ loaded from per_cluster_lambda.json "
                f"({len(per_cluster_lambda)} cluster(s) with tuned values)."
            )
        else:
            print(
                "WARNING: --use-cluster-lambda set but per_cluster_lambda.json not found. "
                "Run: oracle_builder tune --oracle-dir <dir>  first.",
                file=sys.stderr,
            )

    builder = ContextConfigBuilder(
        oracle_dir=args.oracle_dir,
        n_clusters=n_clusters,
        max_features=args.max_features,
        population_size=args.population,
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        lambda_=args.lambda_,
        budgets=budgets,
        embedder=getattr(args, "embedder", "tfidf"),
        sentence_model=getattr(args, "sentence_model", "all-MiniLM-L6-v2"),
        validation_config=validation_config,
        per_cluster_lambda=per_cluster_lambda,
    )
    builder.build(output)
