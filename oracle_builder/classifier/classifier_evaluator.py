# oracle_builder/classifier/classifier_evaluator.py
# Evaluate a trained ComponentInclusionClassifier on a held-out validation set.
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...shared.classifier_model import ComponentInclusionClassifier
from ...shared.outcome_scorer import compute_outcome_quality

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetrics:
    """Per-component evaluation metrics."""
    name: str
    n_positive: int  # ground-truth positive examples
    n_negative: int  # ground-truth negative examples
    precision: float
    recall: float
    f1: float
    auc: float        # ROC-AUC; 0.5 = random, 1.0 = perfect


@dataclass
class EvalResult:
    """Aggregate evaluation result for a classifier checkpoint."""
    evaluated_at: str          # ISO timestamp
    n_val_turns: int
    n_components: int
    macro_f1: float            # unweighted mean F1 across components
    macro_auc: float           # unweighted mean AUC across components
    threshold: float
    per_component: list[ComponentMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_component"] = [asdict(c) for c in self.per_component]
        return d


class ClassifierEvaluator:
    """Compute precision, recall, F1, and AUC for each component on validation turns.

    Usage::

        evaluator = ClassifierEvaluator()
        result = evaluator.evaluate(classifier, val_turns, threshold=0.5)
        evaluator.save(result, oracle_dir)
    """

    def evaluate(
        self,
        classifier: ComponentInclusionClassifier,
        val_turns: list[dict],
        threshold: float = 0.5,
    ) -> EvalResult:
        """Evaluate the classifier on held-out validation turns.

        Parameters
        ----------
        classifier:
            Trained ComponentInclusionClassifier.
        val_turns:
            Held-out turn records (raw dicts from TurnLogger).
        threshold:
            Inclusion probability threshold for binary prediction (default: 0.5).

        Returns
        -------
        EvalResult with per-component and aggregate metrics.
        """
        if not val_turns:
            logger.warning("ClassifierEvaluator: no validation turns provided.")
            return EvalResult(
                evaluated_at=_now_iso(),
                n_val_turns=0,
                n_components=classifier.n_components,
                macro_f1=0.0,
                macro_auc=0.5,
                threshold=threshold,
            )

        embeddings = np.array(
            [t["query_embedding"] for t in val_turns], dtype=np.float32
        )

        # Ground-truth: for each turn, which components were in context AND had
        # a positive outcome (quality > 0.5)?  This mirrors the training labels
        # produced by ComponentClassifierTrainer.
        qualities = np.array(
            [compute_outcome_quality(t) for t in val_turns], dtype=np.float32
        )
        y_positive = qualities > 0.5  # shape (n_val,)

        # Build per-component y_true and predicted probabilities
        per_component_metrics: list[ComponentMetrics] = []
        f1_scores: list[float] = []
        auc_scores: list[float] = []

        for i, comp_name in enumerate(classifier.component_names):
            # y_true[j] = 1 if component was included AND turn was positive
            inclusion_mask = np.array(
                [_component_included(t, comp_name) for t in val_turns], dtype=bool
            )
            y_true = (inclusion_mask & y_positive).astype(int)
            n_pos = int(y_true.sum())
            n_neg = int(len(y_true) - n_pos)

            if n_pos == 0 or n_neg == 0:
                # Cannot compute meaningful metrics with a single class
                per_component_metrics.append(ComponentMetrics(
                    name=comp_name,
                    n_positive=n_pos,
                    n_negative=n_neg,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    auc=0.5,
                ))
                f1_scores.append(0.0)
                auc_scores.append(0.5)
                continue

            # Predicted probabilities for this component
            proba_i = np.array(
                [float(classifier.predict_proba(embeddings[j])[i]) for j in range(len(val_turns))],
                dtype=np.float32,
            )
            y_pred = (proba_i >= threshold).astype(int)

            precision = _precision(y_true, y_pred)
            recall = _recall(y_true, y_pred)
            f1 = _f1(precision, recall)
            auc = _roc_auc(y_true, proba_i)

            per_component_metrics.append(ComponentMetrics(
                name=comp_name,
                n_positive=n_pos,
                n_negative=n_neg,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                auc=round(auc, 4),
            ))
            f1_scores.append(f1)
            auc_scores.append(auc)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.5

        return EvalResult(
            evaluated_at=_now_iso(),
            n_val_turns=len(val_turns),
            n_components=classifier.n_components,
            macro_f1=round(macro_f1, 4),
            macro_auc=round(macro_auc, 4),
            threshold=threshold,
            per_component=per_component_metrics,
        )

    def save(self, result: EvalResult, oracle_dir: Path) -> Path:
        """Write evaluation result to a dated JSON file in oracle_dir.

        Returns the path written.
        """
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        path = oracle_dir / f"classifier_eval_{date_str}.json"
        path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Classifier evaluation saved to %s", path)
        return path


# ── metrics helpers ────────────────────────────────────────────────────────────

def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    if (precision + recall) == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC via the Wilcoxon-Mann-Whitney statistic (no sklearn needed)."""
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # Count concordant pairs: positive score > negative score
    concordant = float(np.sum(pos_scores[:, None] > neg_scores[None, :]))
    tied = float(np.sum(pos_scores[:, None] == neg_scores[None, :]))
    total = float(len(pos_scores) * len(neg_scores))
    return (concordant + 0.5 * tied) / total


def _component_included(turn: dict, comp_name: str) -> bool:
    config = turn.get("context_config", {})
    return (
        comp_name in config.get("skills", [])
        or comp_name in config.get("memory_sections", [])
        or comp_name in config.get("tools", [])
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
