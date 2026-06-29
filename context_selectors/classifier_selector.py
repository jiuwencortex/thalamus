# context_selectors/classifier_selector.py
# Query-time selection using a trained ComponentInclusionClassifier.
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..shared.classifier_model import ComponentInclusionClassifier

logger = logging.getLogger(__name__)

_THRESHOLDS_FILE = "classifier_thresholds.json"


def _load_per_component_thresholds(oracle_dir: Path) -> dict[str, float] | None:
    """Load per-component thresholds tuned by HyperparameterSearch, or None.

    Returns the ``per_component_thresholds`` dict from
    ``classifier_thresholds.json``, or None if the file is absent or unreadable.
    ClassifierSelector falls back to a global threshold when this returns None.
    """
    path = oracle_dir / _THRESHOLDS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("per_component_thresholds") or None
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


class ClassifierSelector:
    """Context selector driven by a trained ComponentInclusionClassifier.

    Loads classifier_current.pkl (or legacy classifier.pkl) and predicts
    per-component inclusion probabilities from a query embedding.

    When ``classifier_thresholds.json`` is present in the oracle directory
    (written by HyperparameterSearch), per-component thresholds are applied
    instead of a single global threshold.  This improves precision/recall
    trade-offs for components that are rare or common in the training data.

    No fallback logic — caller decides what to do when the classifier is not
    available or not confident enough.

    Usage::

        selector = ClassifierSelector.load(oracle_dir)
        result = selector.select(query_embedding)
        # result keys: skills, memory, tools, probabilities, confidence, source
    """

    def __init__(
        self,
        classifier: ComponentInclusionClassifier,
        per_component_thresholds: dict[str, float] | None = None,
    ) -> None:
        self._classifier = classifier
        # Map from component name → inclusion threshold (tuned or None)
        self._thresholds = per_component_thresholds or {}

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, oracle_dir: Path) -> "ClassifierSelector":
        """Load the active classifier from oracle_dir.

        Prefers ``classifier_current.pkl`` (written by the model registry when a
        new version is promoted).  Falls back to the legacy ``classifier.pkl``
        for backwards compatibility with deployments that have not yet run the
        updated training pipeline.

        Also loads ``classifier_thresholds.json`` when present (written by
        HyperparameterSearch) to apply per-component inclusion thresholds at
        inference time.

        Raises FileNotFoundError if no classifier pkl exists.
        """
        current_path = oracle_dir / "classifier_current.pkl"
        legacy_path = oracle_dir / "classifier.pkl"

        if current_path.exists():
            classifier_path = current_path
        elif legacy_path.exists():
            classifier_path = legacy_path
            logger.warning(
                "classifier_current.pkl not found; loading legacy classifier.pkl. "
                "Re-run train-classifier to create a versioned model."
            )
        else:
            raise FileNotFoundError(
                f"No classifier found in {oracle_dir}. "
                "Run: python -m thalamus.oracle_builder train-classifier --oracle-dir <dir>"
            )

        classifier = ComponentInclusionClassifier.load(classifier_path)
        per_component_thresholds = _load_per_component_thresholds(oracle_dir)

        if per_component_thresholds:
            logger.info(
                "Loaded per-component thresholds from %s (%d components)",
                _THRESHOLDS_FILE,
                len(per_component_thresholds),
            )
        logger.info(
            "Loaded classifier from %s (%d components)",
            classifier_path,
            classifier.n_components,
        )
        return cls(classifier, per_component_thresholds=per_component_thresholds)

    # ── inference ─────────────────────────────────────────────────────────────

    def select(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """Predict component inclusion from a query embedding.

        Per-component thresholds (from HyperparameterSearch) override the
        ``threshold`` argument when available.  If no per-component threshold
        exists for a component, the global ``threshold`` argument is used.

        Parameters
        ----------
        query_embedding : np.ndarray
            Pre-computed embedding vector, shape (d_embed,).
        threshold : float
            Global inclusion threshold (default 0.5).  Overridden per-component
            by tuned thresholds when ``classifier_thresholds.json`` was loaded.

        Returns
        -------
        dict with keys:
            skills        list[str]        — skill names above threshold
            memory        list[str]        — memory section names above threshold
            tools         list[str]        — tool names above threshold
            probabilities dict[str, float] — per-component sigmoid scores
            confidence    float            — mean certainty (mean max(p, 1-p))
            source        str              — always "classifier"
        """
        proba = self._classifier.predict_proba(query_embedding)
        confidence = _mean_confidence(proba)

        # Build inclusion list respecting per-component thresholds
        included: list[str] = []
        probabilities: dict[str, float] = {}
        for name, p in zip(self._classifier.component_names, proba):
            p_float = float(p)
            probabilities[name] = p_float
            t = self._thresholds.get(name, threshold)
            if p_float >= t:
                included.append(name)

        skills: list[str] = []
        memory: list[str] = []
        tools: list[str] = []
        for name in included:
            if "skill" in name.lower():
                skills.append(name)
            elif "mem" in name.lower() or "::" in name:
                memory.append(name)
            else:
                tools.append(name)

        return {
            "skills": skills,
            "memory": memory,
            "tools": tools,
            "probabilities": probabilities,
            "confidence": round(confidence, 4),
            "source": "classifier",
        }

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def n_components(self) -> int:
        return self._classifier.n_components

    @property
    def component_names(self) -> list[str]:
        return self._classifier.component_names


def _mean_confidence(proba: np.ndarray) -> float:
    """Return mean max(p, 1-p) — average certainty per component decision."""
    if len(proba) == 0:
        return 0.0
    return float(np.mean(np.maximum(proba, 1 - proba)))
