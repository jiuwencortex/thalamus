from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from .component_classifier_trainer import ComponentClassifierTrainer
from .log_splitter import LogSplitter
from .classifier_evaluator import ClassifierEvaluator
from .model_registry import ModelRegistry


def cmd_train_classifier(args: argparse.Namespace) -> None:
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = args.log_dir or (args.oracle_dir / "online_logs")
    force_promote = getattr(args, "force_promote", False)

    # ── 1. Split turns into train / validation ────────────────────────────────
    splitter = LogSplitter(log_dir, max_weeks=args.max_weeks)
    train_turns, val_turns = splitter.split()

    total_turns = len(train_turns) + len(val_turns)
    if total_turns < args.min_turns:
        print(
            f"Not enough turns to train (have {total_turns}, need {args.min_turns}). "
            "Collect more interaction data and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Loaded {total_turns} turns: {len(train_turns)} train / {len(val_turns)} validation."
    )

    # ── 2. Train on the training split ───────────────────────────────────────
    # Guard: need at least 2 training turns per component; use a lenient minimum here
    # since the trainer's own guard handles the per-component case.
    min_train = max(1, args.min_turns - len(val_turns))
    trainer = ComponentClassifierTrainer(
        log_dir=log_dir,
        min_turns=min_train,
        max_weeks=args.max_weeks,
        C=args.C,
    )
    classifier = trainer.train_on_turns(train_turns)
    if classifier is None:
        print(
            f"Training failed: not enough training turns after split "
            f"({len(train_turns)} available).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 3. Save the new model under a dated filename ──────────────────────────
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    versioned_path = args.oracle_dir / f"classifier_{timestamp}.pkl"
    classifier.save(versioned_path)
    print(f"Classifier written to {versioned_path}")

    # ── 4. Evaluate on the validation split ──────────────────────────────────
    evaluator = ClassifierEvaluator()
    eval_result = evaluator.evaluate(classifier, val_turns, threshold=0.5)
    eval_path = evaluator.save(eval_result, args.oracle_dir)

    print(
        f"Validation: macro_F1={eval_result.macro_f1:.4f}  "
        f"macro_AUC={eval_result.macro_auc:.4f}  "
        f"({eval_result.n_val_turns} turns)  → {eval_path.name}"
    )

    # ── 5. Register and optionally promote ───────────────────────────────────
    registry = ModelRegistry(args.oracle_dir)
    entry = registry.register(
        model_path=versioned_path,
        eval_result=eval_result,
        n_train_turns=len(train_turns),
        force_promote=force_promote,
    )

    current_path = args.oracle_dir / "classifier_current.pkl"
    if entry.is_current:
        print(f"Promoted to active model → {current_path}")
    else:
        current = registry.get_current()
        current_f1 = current.macro_f1 if current else 0.0
        print(
            f"Not promoted: new macro_F1={entry.macro_f1:.4f}, "
            f"current macro_F1={current_f1:.4f}. "
            "Pass --force-promote to override the gate."
        )
