# oracle_builder/classifier/component_classifier_trainer.py
# Train ComponentInclusionClassifier on accumulated (embedding, config, outcome) turns.
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ...shared.outcome_scorer import compute_outcome_quality
from ...shared.turn_logger import TurnLogger
from ...shared.classifier_model import ComponentInclusionClassifier

logger = logging.getLogger(__name__)

_MIN_TURNS = 10  # refuse to train on fewer turns than this


class ComponentClassifierTrainer:
    """Train a ComponentInclusionClassifier from logged agent turns.

    Training procedure (one binary classifier per component):

    For each component C:
        X = query_embeddings of turns where C was included in context
        y = outcome_quality(T)   (continuous in [0, 1])
        Binary threshold: y_binary = 1 if quality >= 0.5 else 0

    A logistic regression model is fitted per component.
    Weights and biases are extracted and packaged into a ComponentInclusionClassifier.

    Turns where C was NOT included contribute no training signal for C
    (the plan design: "0 where component was not included (no signal either way)").
    """

    def __init__(
        self,
        log_dir: Path,
        min_turns: int = _MIN_TURNS,
        max_weeks: int = 8,
        C: float = 1.0,  # L2 regularization inverse strength
    ):
        self._log_dir = log_dir
        self._min_turns = min_turns
        self._max_weeks = max_weeks
        self._C = C

    def train(self) -> ComponentInclusionClassifier | None:
        """Load all logged turns, train per-component classifiers, return classifier.

        Returns None if there are too few turns to train.
        """
        turn_logger = TurnLogger(self._log_dir)
        turns = turn_logger.load_turns(self._max_weeks)
        return self.train_on_turns(turns)

    def train_on_turns(
        self, turns: list[dict]
    ) -> ComponentInclusionClassifier | None:
        """Train per-component classifiers on a pre-supplied list of turns.

        Accepts an explicit turn list so the caller can pass a pre-split training
        partition (e.g. from LogSplitter) rather than all available logs.

        Returns None if there are too few turns to train.
        """
        if len(turns) < self._min_turns:
            logger.warning(
                "Only %d turns available (need %d); classifier not trained.",
                len(turns),
                self._min_turns,
            )
            return None

        # Collect all component names across all turns
        all_components: list[str] = _collect_component_names(turns)
        if not all_components:
            logger.warning("No component names found in logged turns.")
            return None

        # Build embedding matrix
        embeddings = np.array([t["query_embedding"] for t in turns], dtype=np.float32)
        d_embed = embeddings.shape[1]
        n_comp = len(all_components)

        weights = np.zeros((n_comp, d_embed), dtype=np.float32)
        biases = np.zeros(n_comp, dtype=np.float32)

        for i, comp_name in enumerate(all_components):
            # Select turns where this component was in context
            mask = _inclusion_mask(turns, comp_name)
            n_included = mask.sum()

            if n_included < 2:
                # Not enough data for this component; leave weights at zero (predict 0.5 always)
                logger.debug("Component %r: only %d turns, skipping.", comp_name, n_included)
                continue

            X = embeddings[mask]
            qualities = np.array(
                [compute_outcome_quality(t) for t, m in zip(turns, mask) if m],
                dtype=np.float32,
            )
            y_binary = (qualities >= 0.5).astype(int)

            # Handle case where all labels are the same class
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                # Only one class; use mean quality as constant bias, zero weight
                biases[i] = float(qualities.mean()) - 0.5
                logger.debug("Component %r: single class, constant bias only.", comp_name)
                continue

            clf = LogisticRegression(C=self._C, max_iter=1000, solver="lbfgs")
            clf.fit(X, y_binary)

            weights[i] = clf.coef_[0]
            biases[i] = float(clf.intercept_[0])
            logger.debug("Component %r: trained on %d turns.", comp_name, n_included)

        classifier = ComponentInclusionClassifier(
            weights=weights,
            biases=biases,
            component_names=all_components,
        )
        logger.info(
            "Classifier trained: %d component(s), %d turns, embed_dim=%d",
            n_comp,
            len(turns),
            d_embed,
        )
        return classifier

    def train_and_save(self, classifier_path: Path) -> bool:
        """Train the classifier on all available turns and save to classifier_path.

        Returns True if training succeeded and classifier was saved, False otherwise.
        Note: does NOT use the held-out split or model registry.  Use cmd_train_classifier
        for the full pipeline including evaluation and promotion gate.
        """
        classifier = self.train()
        if classifier is None:
            return False
        classifier.save(classifier_path)
        logger.info("Classifier saved to %s", classifier_path)
        return True


# ── helpers ───────────────────────────────────────────────────────────────────

def _collect_component_names(turns: list[dict]) -> list[str]:
    """Return a sorted deduplicated list of all component names across turns."""
    names: set[str] = set()
    for turn in turns:
        config = turn.get("context_config", {})
        names.update(config.get("skills", []))
        names.update(config.get("memory_sections", []))
        names.update(config.get("tools", []))
    return sorted(names)


def _inclusion_mask(turns: list[dict], comp_name: str) -> np.ndarray:
    """Return a boolean array: True for turns where comp_name was in context."""
    result = []
    for turn in turns:
        config = turn.get("context_config", {})
        included = (
            comp_name in config.get("skills", [])
            or comp_name in config.get("memory_sections", [])
            or comp_name in config.get("tools", [])
        )
        result.append(included)
    return np.array(result, dtype=bool)
