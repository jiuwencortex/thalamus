# oracle_builder/cli_args_parser.py
# Entry point: python -m jiuwenswarm.tools.oracle_builder build
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m jiuwenswarm.tools.oracle_builder",
        description=(
            "Build context_configs.json from pre-computed recommendation matrices.\n\n"
            "Reads all scoring_matrix_skill_*.json, scoring_matrix_mem_*.json, and\n"
            "scoring_matrix_tool_*.json files from --oracle-dir.\n\n"
            "Clusters their example queries, runs an evolutionary search over component\n"
            "combinations (no LLM calls), and writes optimal configs per cluster and budget\n"
            "to context_configs.json. A companion .pkl file stores the TF-IDF model for\n"
            "query-time cluster assignment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    build = sub.add_parser("evolve", help="Run evolutionary search to build context_configs.json")
    build.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring_matrix_*.json files (output of component_scoring build)",
    )
    build.add_argument(
        "--output", type=Path,
        help="Path to write context_configs.json (default: <oracle-dir>/context_configs.json)",
    )
    build.add_argument(
        "--n-clusters", type=int, default=20,
        help="Number of query-type clusters (default: 20)",
    )
    build.add_argument(
        "--max-features", type=int, default=2000,
        help="TF-IDF vocabulary size (default: 2000)",
    )
    build.add_argument(
        "--population", type=int, default=100,
        help="Evolutionary search population size (default: 100)",
    )
    build.add_argument(
        "--generations", type=int, default=200,
        help="Number of evolutionary generations (default: 200)",
    )
    build.add_argument(
        "--mutation-rate", type=float, default=0.05,
        help="Mutation probability per bit (default: 0.05)",
    )
    build.add_argument(
        "--lambda", dest="lambda_", type=float, default=0.1,
        help="Context-size penalty weight (default: 0.1)",
    )
    build.add_argument(
        "--budget-small", type=int, default=2000,
        help="Max tokens for 'small' budget (default: 2000)",
    )
    build.add_argument(
        "--budget-medium", type=int, default=4000,
        help="Max tokens for 'medium' budget (default: 4000)",
    )
    build.add_argument(
        "--budget-large", type=int, default=8000,
        help="Max tokens for 'large' budget (default: 8000)",
    )
    build.add_argument(
        "--embedder", default="tfidf", choices=["tfidf", "sentence"],
        help=(
            "Embedding backend for query clustering: "
            "'tfidf' (default, fast, no extra deps) or "
            "'sentence' (semantic, requires sentence-transformers)"
        ),
    )
    build.add_argument(
        "--sentence-model", dest="sentence_model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (only used when --embedder=sentence, default: all-MiniLM-L6-v2)",
    )
    build.add_argument(
        "--validate-pareto", action="store_true",
        help=(
            "After GA, evaluate each Pareto-front config with LLM calls against "
            "representative cluster queries and re-rank before selecting the best config. "
            "Requires --eval-model and --eval-api-key."
        ),
    )
    build.add_argument(
        "--eval-model", default="gpt-4o-mini",
        help="LLM model to use for Pareto validation (default: gpt-4o-mini)",
    )
    build.add_argument(
        "--eval-api-key", dest="eval_api_key", default=None,
        help="OpenAI-compatible API key for Pareto validation (falls back to OPENAI_API_KEY env var)",
    )
    build.add_argument(
        "--eval-api-base", dest="eval_api_base", default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL (default: https://api.openai.com/v1)",
    )
    build.add_argument(
        "--eval-queries-per-cluster", type=int, default=3, dest="eval_queries_per_cluster",
        help="Number of representative queries per cluster used in Pareto validation (default: 3)",
    )
    build.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    build.add_argument(
        "--auto-k", dest="auto_k", action="store_true",
        help=(
            "Automatically select the optimal number of K-means clusters using an "
            "elbow + silhouette analysis before running the evolutionary search.  "
            "Overrides --n-clusters when set."
        ),
    )
    build.add_argument(
        "--use-cluster-lambda", dest="use_cluster_lambda", action="store_true",
        help=(
            "Load per-cluster lambda values from per_cluster_lambda.json (written by "
            "the 'tune' subcommand) and apply them in the fitness function instead of "
            "the global --lambda value."
        ),
    )
    build.add_argument(
        "--log-dir", dest="log_dir", type=Path, default=None,
        help="Directory with turn logs (default: <oracle-dir>/online_logs). Used with --auto-k.",
    )

    # Subcommand: train the component inclusion classifier
    train = sub.add_parser(
        "train-classifier",
        help="Train a component inclusion classifier from logged agent turns",
    )
    train.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory to write classifier.pkl",
    )
    train.add_argument(
        "--log-dir", type=Path,
        help="Directory containing JSONL turn logs (default: <oracle-dir>/online_logs)",
    )
    train.add_argument(
        "--min-turns", type=int, default=10,
        help="Minimum logged turns required to train (default: 10)",
    )
    train.add_argument(
        "--max-weeks", type=int, default=8,
        help="Number of weekly log files to scan (default: 8)",
    )
    train.add_argument(
        "--C", dest="C", type=float, default=1.0,
        help="Inverse L2 regularisation strength for LogisticRegression (default: 1.0)",
    )
    train.add_argument(
        "--judge-model", dest="judge_model", default="gpt-4o-mini",
        help=(
            "LLM model to use as quality judge when turns carry an llm_judge_score field "
            "(default: gpt-4o-mini). See shared.outcome_scorer.score_from_llm_judge."
        ),
    )
    train.add_argument(
        "--judge-api-key", dest="judge_api_key", default=None,
        help="API key for the LLM judge (falls back to OPENAI_API_KEY env var).",
    )
    train.add_argument(
        "--judge-api-base", dest="judge_api_base", default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL for the LLM judge (default: https://api.openai.com/v1).",
    )
    train.add_argument(
        "--force-promote", dest="force_promote", action="store_true",
        help=(
            "Promote the newly trained model to classifier_current.pkl even if its "
            "validation F1 does not exceed the current model's F1 by the required margin."
        ),
    )

    # Subcommand: list registered classifier versions
    versions = sub.add_parser(
        "list-versions",
        help="List all registered classifier versions and their validation metrics.",
    )
    versions.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Oracle directory containing classifier_registry.json",
    )

    # Subcommand: tune all hyperparameters from logged data
    tune = sub.add_parser(
        "tune",
        help=(
            "Run all hyperparameter tuners (classifier C/threshold, cluster count K, "
            "per-cluster λ) from logged turn data.  Writes classifier_thresholds.json "
            "and per_cluster_lambda.json to oracle-dir.  Does NOT retrain the classifier "
            "or rebuild the oracle — run train-classifier / evolve after tuning."
        ),
    )
    tune.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Oracle directory (must contain context_configs.pkl for λ tuning).",
    )
    tune.add_argument(
        "--log-dir", dest="log_dir", type=Path, default=None,
        help="Directory containing JSONL turn logs (default: <oracle-dir>/online_logs).",
    )
    tune.add_argument(
        "--max-weeks", type=int, default=8,
        help="Number of weekly log files to scan (default: 8).",
    )
    tune.add_argument(
        "--min-turns", type=int, default=10,
        help="Minimum total turns required to run tuning (default: 10).",
    )
    tune.add_argument(
        "--default-lambda", dest="default_lambda", type=float, default=0.1,
        help="Default λ value to use when a cluster has too few turns to tune (default: 0.1).",
    )
    tune.add_argument(
        "--skip-classifier-tune", dest="skip_classifier_tune", action="store_true",
        help="Skip classifier C/threshold grid search (only run K and λ tuners).",
    )
    tune.add_argument(
        "--skip-k-tune", dest="skip_k_tune", action="store_true",
        help="Skip cluster count K search.",
    )
    tune.add_argument(
        "--skip-lambda-tune", dest="skip_lambda_tune", action="store_true",
        help="Skip per-cluster λ tuning.",
    )
    tune.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    # Subcommand: show drift + staleness + model registry status
    status_p = sub.add_parser(
        "status",
        help=(
            "Show the current health of the oracle: drift status, staleness, "
            "and classifier registry.  Runs fresh drift and staleness checks "
            "and saves results to oracle-dir."
        ),
    )
    status_p.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Oracle directory to inspect.",
    )
    status_p.add_argument(
        "--log-dir", dest="log_dir", type=Path, default=None,
        help="Directory containing JSONL turn logs (default: <oracle-dir>/online_logs).",
    )
    status_p.add_argument(
        "--recent-weeks", dest="recent_weeks", type=int, default=1,
        help="Size of the 'recent' window for drift detection (default: 1 week).",
    )
    status_p.add_argument(
        "--baseline-weeks", dest="baseline_weeks", type=int, default=4,
        help="Size of the baseline window for drift detection (default: 4 weeks).",
    )
    status_p.add_argument(
        "--js-threshold", dest="js_threshold", type=float, default=0.15,
        help="Jensen-Shannon divergence threshold for drift detection (default: 0.15).",
    )
    status_p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    # Subcommand: decide whether a rebuild is needed
    check_p = sub.add_parser(
        "check-rebuild",
        help=(
            "Inspect cached drift_status.json and staleness_status.json "
            "(written by the 'status' command) and print whether the classifier "
            "or oracle should be rebuilt.  Exit code 2 if a rebuild is recommended."
        ),
    )
    check_p.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Oracle directory (must contain drift_status.json / staleness_status.json).",
    )
    check_p.add_argument(
        "--log-dir", dest="log_dir", type=Path, default=None,
        help="Directory containing JSONL turn logs (default: <oracle-dir>/online_logs).",
    )
    check_p.add_argument(
        "--new-turns-threshold", dest="new_turns_threshold", type=int, default=200,
        help=(
            "Minimum number of new turns logged since last training to trigger "
            "classifier retraining (default: 200)."
        ),
    )
    check_p.add_argument(
        "--drift-rebuild-min-turns", dest="drift_rebuild_min_turns", type=int, default=50,
        help=(
            "Minimum number of recent turns required for drift alone to trigger "
            "an oracle rebuild (default: 50)."
        ),
    )
    check_p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    return p
