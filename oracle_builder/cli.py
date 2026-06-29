# oracle_builder/cli.py
# Entry point: python -m jiuwenswarm.tools.oracle_builder build
from __future__ import annotations

import logging
import sys

from jiuwenswarm.thalamus.oracle_builder.cli_args_parser import make_parser
from jiuwenswarm.thalamus.oracle_builder.evolutionary.cmd_build import cmd_build
from jiuwenswarm.thalamus.oracle_builder.classifier.cmd_train_classifier import cmd_train_classifier
from jiuwenswarm.thalamus.oracle_builder.classifier.model_registry import ModelRegistry


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(message)s", stream=sys.stderr)

    if args.command == "evolve":
        cmd_build(args)
    elif args.command == "train-classifier":
        cmd_train_classifier(args)
    elif args.command == "list-versions":
        _cmd_list_versions(args)
    elif args.command == "tune":
        _cmd_tune(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "check-rebuild":
        _cmd_check_rebuild(args)


def _cmd_list_versions(args) -> None:
    """Print a table of all registered classifier versions."""
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    registry = ModelRegistry(args.oracle_dir)
    versions = registry.list_versions()

    if not versions:
        print("No classifier versions registered yet.")
        print(f"Run: python -m thalamus.oracle_builder train-classifier --oracle-dir {args.oracle_dir}")
        return

    header = f"{'#':<4}  {'filename':<40}  {'trained_at':<22}  {'train':>6}  {'val':>5}  {'F1':>7}  {'AUC':>7}  active"
    print(header)
    print("-" * len(header))

    for idx, entry in enumerate(versions, start=1):
        active_marker = "  *" if entry.is_current else ""
        print(
            f"{idx:<4}  {entry.filename:<40}  {entry.trained_at:<22}  "
            f"{entry.n_train_turns:>6}  {entry.n_val_turns:>5}  "
            f"{entry.macro_f1:>7.4f}  {entry.macro_auc:>7.4f}{active_marker}"
        )


def _cmd_tune(args) -> None:
    """Run all hyperparameter tuners and save results to oracle_dir."""
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")

    # ── Classifier C / threshold tuning ──────────────────────────────────────
    if not getattr(args, "skip_classifier_tune", False):
        print("=== Classifier hyperparameter search (C × threshold grid) ===")
        from jiuwenswarm.thalamus.oracle_builder.classifier.hyperparameter_search import HyperparameterSearch
        search = HyperparameterSearch(
            log_dir=log_dir,
            max_weeks=args.max_weeks,
            min_turns=args.min_turns,
        )
        train_turns, val_turns = search.split()
        total = len(train_turns) + len(val_turns)
        if total < args.min_turns:
            print(
                f"  Skipping: only {total} turns available (need {args.min_turns})."
            )
        else:
            print(f"  {total} turns: {len(train_turns)} train / {len(val_turns)} val")
            result = search.run(train_turns, val_turns)
            saved = search.save(result, args.oracle_dir)
            print(
                f"  Best C={result.best_C}  threshold={result.best_threshold}  "
                f"macro_F1={result.best_macro_f1:.4f}  → {saved.name}"
            )
    else:
        print("=== Classifier tuning skipped (--skip-classifier-tune) ===")

    # ── Cluster count K tuning ────────────────────────────────────────────────
    if not getattr(args, "skip_k_tune", False):
        print("=== Cluster count K tuning ===")
        from jiuwenswarm.thalamus.oracle_builder.evolutionary.config_builder_step01_load_components import ComponentsLoader
        from jiuwenswarm.thalamus.oracle_builder.evolutionary.config_builder_step02_collect_texts import TextsCollector
        from jiuwenswarm.thalamus.oracle_builder.evolutionary.cluster_count_tuner import ClusterCountTuner

        loader = ComponentsLoader(args.oracle_dir)
        collector = TextsCollector()
        components, example_texts_map = loader.load()
        all_texts = collector.collect(components, example_texts_map)

        if not all_texts:
            print("  Skipping: no example texts found in oracle_dir.")
        else:
            tuner = ClusterCountTuner()
            k_result = tuner.tune(all_texts)
            print(
                f"  Best K={k_result.best_k} (method={k_result.method}) "
                f"from {len(all_texts)} example texts."
            )
            if k_result.entries:
                print(f"  {'K':>4}  {'inertia':>12}  {'silhouette':>10}")
                for e in k_result.entries:
                    marker = "  <-- selected" if e.k == k_result.best_k else ""
                    print(f"  {e.k:>4}  {e.inertia:>12.1f}  {e.silhouette:>10.4f}{marker}")
            print(
                f"  Recommendation: use --n-clusters {k_result.best_k} on the next 'evolve' run."
            )
    else:
        print("=== Cluster count K tuning skipped (--skip-k-tune) ===")

    # ── Per-cluster λ tuning ──────────────────────────────────────────────────
    if not getattr(args, "skip_lambda_tune", False):
        print("=== Per-cluster λ tuning ===")
        from jiuwenswarm.thalamus.oracle_builder.evolutionary.lambda_tuner import LambdaTuner

        lambda_tuner = LambdaTuner(
            log_dir=log_dir,
            oracle_dir=args.oracle_dir,
            max_weeks=args.max_weeks,
        )
        lambda_result = lambda_tuner.tune(default_lambda=getattr(args, "default_lambda", 0.1))
        if not lambda_result.entries:
            print("  Skipping: context_configs.pkl not found or no logged turns.")
        else:
            saved = lambda_tuner.save(lambda_result, args.oracle_dir)
            n_tuned = sum(
                1 for e in lambda_result.entries
                if e.best_lambda != lambda_result.default_lambda
            )
            print(
                f"  {n_tuned}/{len(lambda_result.entries)} clusters have tuned λ values  → {saved.name}"
            )
            print(f"  {'cluster':>8}  {'n_turns':>8}  {'corr':>8}  {'λ':>6}")
            for e in lambda_result.entries:
                print(
                    f"  {e.cluster_id:>8}  {e.n_turns:>8}  "
                    f"{e.token_outcome_correlation:>8.3f}  {e.best_lambda:>6.3f}"
                )
    else:
        print("=== Per-cluster λ tuning skipped (--skip-lambda-tune) ===")


def _cmd_status(args) -> None:
    """Run fresh drift + staleness checks and print a health summary."""
    import sys
    from jiuwenswarm.thalamus.oracle_builder.staleness_checker import StalenessChecker
    from jiuwenswarm.thalamus.oracle_builder.classifier.model_registry import ModelRegistry
    from jiuwenswarm.thalamus.shared.distribution_monitor import DistributionMonitor

    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")

    # ── Drift check ───────────────────────────────────────────────────────────
    print("=== Distribution Drift ===")
    monitor = DistributionMonitor(
        log_dir=log_dir,
        oracle_dir=args.oracle_dir,
        recent_weeks=getattr(args, "recent_weeks", 1),
        baseline_weeks=getattr(args, "baseline_weeks", 4),
        js_threshold=getattr(args, "js_threshold", 0.15),
    )
    drift = monitor.check()
    monitor.save(drift, args.oracle_dir)
    drift_marker = "  [DRIFT DETECTED]" if drift.drift_detected else ""
    print(
        f"  JS divergence : {drift.js_divergence:.4f}  "
        f"(threshold={drift.threshold}){drift_marker}"
    )
    print(f"  Recent turns  : {drift.n_recent}  |  Baseline turns: {drift.n_baseline}")
    print(f"  {drift.message}")

    # ── Staleness check ───────────────────────────────────────────────────────
    print("\n=== Oracle Staleness ===")
    checker = StalenessChecker(args.oracle_dir)
    staleness = checker.check()
    checker.save(staleness, args.oracle_dir)
    stale_marker = "  [STALE]" if staleness.stale else ""
    print(f"  Oracle exists    : {staleness.oracle_exists}{stale_marker}")
    print(f"  Oracle mtime     : {staleness.oracle_mtime}")
    print(f"  Components in oracle   : {staleness.n_oracle_components}")
    print(f"  Scoring matrices found : {staleness.n_current_matrices}")
    if staleness.added_components:
        print(f"  Added (not in oracle)  : {', '.join(staleness.added_components)}")
    if staleness.removed_components:
        print(f"  Removed (in oracle)    : {', '.join(staleness.removed_components)}")
    if staleness.updated_components:
        print(f"  Updated (newer mtime)  : {', '.join(staleness.updated_components)}")
    print(f"  {staleness.message}")

    # ── Classifier registry ───────────────────────────────────────────────────
    print("\n=== Classifier Registry ===")
    registry = ModelRegistry(args.oracle_dir)
    versions = registry.list_versions()
    current = registry.get_current()
    if not versions:
        print("  No classifier versions registered yet.")
    else:
        print(f"  Registered versions: {len(versions)}")
        if current:
            print(
                f"  Current: {current.filename}  "
                f"F1={current.macro_f1:.4f}  AUC={current.macro_auc:.4f}  "
                f"trained={current.trained_at}"
            )


def _cmd_check_rebuild(args) -> None:
    """Check cached drift/staleness status and print rebuild recommendations."""
    import sys
    from jiuwenswarm.thalamus.oracle_builder.retraining_scheduler import RetrainingScheduler

    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")

    scheduler = RetrainingScheduler(
        oracle_dir=args.oracle_dir,
        log_dir=log_dir,
        new_turns_threshold=getattr(args, "new_turns_threshold", 200),
        drift_rebuild_min_turns=getattr(args, "drift_rebuild_min_turns", 50),
    )
    rec = scheduler.check()

    print("=== Rebuild Recommendation ===")

    if rec.retrain_classifier:
        print("\n  [ACTION NEEDED] Retrain the classifier:")
        for reason in rec.retrain_reasons:
            print(f"    • {reason}")
        print(
            "\n  Run: python -m jiuwenswarm.tools.oracle_builder train-classifier "
            f"--oracle-dir {args.oracle_dir}"
        )
    else:
        print("\n  Classifier is up-to-date — no retraining needed.")

    if rec.rebuild_oracle:
        print("\n  [ACTION NEEDED] Rebuild the oracle:")
        for reason in rec.rebuild_reasons:
            print(f"    • {reason}")
        print(
            "\n  Run: python -m jiuwenswarm.tools.oracle_builder evolve "
            f"--oracle-dir {args.oracle_dir}"
        )
    else:
        print("\n  Oracle is up-to-date — no rebuild needed.")

    # Exit code 2 signals that a rebuild is recommended (useful in CI/scripts)
    if rec.retrain_classifier or rec.rebuild_oracle:
        sys.exit(2)


if __name__ == "__main__":
    main()
