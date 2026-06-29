# oracle_builder/classifier/hyperparameter_search.py
# Grid search over classifier hyperparameters (C, global threshold, per-component threshold).
# Uses the chronological train/validation split produced by LogSplitter.
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .component_classifier_trainer import ComponentClassifierTrainer
from .classifier_evaluator import ClassifierEvaluator, _roc_auc, _component_included, _f1, _precision, _recall
from .log_splitter import LogSplitter
from ...shared.classifier_model import ComponentInclusionClassifier
from ...shared.outcome_scorer import compute_outcome_quality

logger = logging.getLogger(__name__)

_C_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
_THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
_PER_COMPONENT_THRESHOLD_GRID = [0.3, 0.5, 0.7]

_THRESHOLDS_FILE = "classifier_thresholds.json"


@dataclass
class SearchResult:
    """Best hyperparameters found by grid search."""
    best_C: float
    best_threshold: float            # globally best threshold
    best_macro_f1: float
    per_component_thresholds: dict[str, float] = field(default_factory=dict)
    search_log: list[dict] = field(default_factory=list)


class HyperparameterSearch:
    """Grid search over (C, threshold) using the chronological train/val split.

    Phase 1 — global search:
        For each C in C_GRID:
            Train classifier on train_turns with this C.
            For each threshold in THRESHOLD_GRID:
                Evaluate on val_turns.
                Record macro_F1.
        Select (C, threshold) with highest macro_F1.

    Phase 2 — per-component threshold refinement:
        Using the best-C classifier, sweep per-component thresholds over
        [0.3, 0.5, 0.7] and select the threshold that maximises per-component F1.

    The result is saved to ``classifier_thresholds.json`` in oracle_dir so that
    ClassifierSelector can load it at inference time.

    Usage::

        search = HyperparameterSearch(log_dir, max_weeks=8)
        train_turns, val_turns = search.split()
        result = search.run(train_turns, val_turns)
        search.save(result, oracle_dir)
    """

    def __init__(
        self,
        log_dir: Path,
        max_weeks: int = 8,
        validation_fraction: float = 0.20,
        min_turns: int = 10,
    ):
        self._log_dir = log_dir
        self._max_weeks = max_weeks
        self._validation_fraction = validation_fraction
        self._min_turns = min_turns

    def split(self) -> tuple[list[dict], list[dict]]:
        """Load and split turns into train/val partitions."""
        splitter = LogSplitter(
            self._log_dir,
            max_weeks=self._max_weeks,
            validation_fraction=self._validation_fraction,
        )
        return splitter.split()

    def run(
        self,
        train_turns: list[dict],
        val_turns: list[dict],
    ) -> SearchResult:
        """Run the full grid search.

        Parameters
        ----------
        train_turns:
            80 % chronologically-older turns for training.
        val_turns:
            20 % most-recent turns for validation.

        Returns
        -------
        SearchResult with best (C, threshold) and per-component thresholds.
        """
        if not val_turns:
            logger.warning("No validation turns — returning default hyperparameters.")
            return SearchResult(best_C=1.0, best_threshold=0.5, best_macro_f1=0.0)

        evaluator = ClassifierEvaluator()
        search_log: list[dict] = []

        best_C = 1.0
        best_threshold = 0.5
        best_macro_f1 = -1.0
        best_classifier: ComponentInclusionClassifier | None = None

        # ── Phase 1: global C × threshold grid search ─────────────────────────
        for C in _C_GRID:
            trainer = ComponentClassifierTrainer(
                log_dir=self._log_dir,
                min_turns=self._min_turns,
                max_weeks=self._max_weeks,
                C=C,
            )
            clf = trainer.train_on_turns(train_turns)
            if clf is None:
                logger.debug("C=%.2f: not enough training turns, skipping.", C)
                continue

            for threshold in _THRESHOLD_GRID:
                result = evaluator.evaluate(clf, val_turns, threshold=threshold)
                entry = {
                    "C": C,
                    "threshold": threshold,
                    "macro_f1": result.macro_f1,
                    "macro_auc": result.macro_auc,
                }
                search_log.append(entry)
                logger.debug(
                    "C=%.3f  threshold=%.2f  macro_F1=%.4f  macro_AUC=%.4f",
                    C, threshold, result.macro_f1, result.macro_auc,
                )

                if result.macro_f1 > best_macro_f1:
                    best_macro_f1 = result.macro_f1
                    best_C = C
                    best_threshold = threshold
                    best_classifier = clf

        logger.info(
            "Best global hyperparams: C=%.3f  threshold=%.2f  macro_F1=%.4f",
            best_C, best_threshold, best_macro_f1,
        )

        # ── Phase 2: per-component threshold refinement ───────────────────────
        per_component_thresholds: dict[str, float] = {}

        if best_classifier is not None and val_turns:
            embeddings = np.array(
                [t["query_embedding"] for t in val_turns], dtype=np.float32
            )
            qualities = np.array(
                [compute_outcome_quality(t) for t in val_turns], dtype=np.float32
            )
            y_positive = qualities > 0.5

            for i, comp_name in enumerate(best_classifier.component_names):
                inclusion_mask = np.array(
                    [_component_included(t, comp_name) for t in val_turns], dtype=bool
                )
                y_true = (inclusion_mask & y_positive).astype(int)

                if y_true.sum() == 0 or (1 - y_true).sum() == 0:
                    # Single class — keep global best threshold
                    per_component_thresholds[comp_name] = best_threshold
                    continue

                proba_i = np.array(
                    [float(best_classifier.predict_proba(embeddings[j])[i])
                     for j in range(len(val_turns))],
                    dtype=np.float32,
                )

                best_t = best_threshold
                best_comp_f1 = -1.0
                for t in _PER_COMPONENT_THRESHOLD_GRID:
                    y_pred = (proba_i >= t).astype(int)
                    prec = _precision(y_true, y_pred)
                    rec = _recall(y_true, y_pred)
                    comp_f1 = _f1(prec, rec)
                    if comp_f1 > best_comp_f1:
                        best_comp_f1 = comp_f1
                        best_t = t

                per_component_thresholds[comp_name] = best_t

        return SearchResult(
            best_C=best_C,
            best_threshold=best_threshold,
            best_macro_f1=round(best_macro_f1, 4),
            per_component_thresholds=per_component_thresholds,
            search_log=search_log,
        )

    def save(self, result: SearchResult, oracle_dir: Path) -> Path:
        """Write thresholds to classifier_thresholds.json in oracle_dir.

        ClassifierSelector reads this file at load time to use per-component
        thresholds instead of a global default.
        """
        data = {
            "best_C": result.best_C,
            "best_threshold": result.best_threshold,
            "best_macro_f1": result.best_macro_f1,
            "per_component_thresholds": result.per_component_thresholds,
        }
        path = oracle_dir / _THRESHOLDS_FILE
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Thresholds saved to %s", path)
        return path


def load_thresholds(oracle_dir: Path) -> dict[str, float] | None:
    """Load per-component thresholds from oracle_dir/classifier_thresholds.json.

    Returns None if the file does not exist (caller falls back to global default).
    """
    path = oracle_dir / _THRESHOLDS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("per_component_thresholds")
    except Exception as exc:
        logger.warning("Could not load classifier_thresholds.json: %s", exc)
        return None
