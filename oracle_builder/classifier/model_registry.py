# oracle_builder/classifier/model_registry.py
# Track classifier versions: training metadata, evaluation metrics, and active model pointer.
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .classifier_evaluator import EvalResult

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "classifier_registry.json"
_CURRENT_LINK = "classifier_current.pkl"
_PROMOTION_MIN_IMPROVEMENT = 0.01  # macro_f1 must improve by at least this to promote


@dataclass
class RegistryEntry:
    """Metadata for one trained classifier version."""
    filename: str            # e.g. "classifier_2025-01-14_093000.pkl"
    trained_at: str          # ISO timestamp
    n_train_turns: int
    n_val_turns: int
    macro_f1: float
    macro_auc: float
    threshold: float
    is_current: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """Versioned store for classifier.pkl artifacts.

    Each training run writes a dated ``classifier_YYYY-MM-DD_HHMMSS.pkl`` file.
    The registry tracks all versions and points ``classifier_current.pkl`` at
    whichever version is active.

    A new version is only made current (promoted) if its macro F1 on the
    held-out validation set exceeds the current model's stored F1 by at least
    ``min_improvement`` (default: 0.01).

    Usage::

        registry = ModelRegistry(oracle_dir)
        registry.register(
            model_path=new_pkl,
            eval_result=eval_result,
            n_train_turns=800,
        )
        # register() calls promote() internally when the gate passes.
    """

    def __init__(
        self,
        oracle_dir: Path,
        min_improvement: float = _PROMOTION_MIN_IMPROVEMENT,
    ):
        self._oracle_dir = oracle_dir
        self._registry_path = oracle_dir / _REGISTRY_FILE
        self._current_path = oracle_dir / _CURRENT_LINK
        self._min_improvement = min_improvement

    # ── public API ────────────────────────────────────────────────────────────

    def register(
        self,
        model_path: Path,
        eval_result: EvalResult,
        n_train_turns: int,
        *,
        force_promote: bool = False,
    ) -> RegistryEntry:
        """Record a new classifier version and optionally promote it.

        Parameters
        ----------
        model_path:
            Path to the newly trained ``*.pkl`` file.
        eval_result:
            Evaluation result from ClassifierEvaluator.
        n_train_turns:
            Number of training turns used.
        force_promote:
            If True, promote regardless of the improvement gate.

        Returns
        -------
        The newly created RegistryEntry.
        """
        entries = self._load()
        current = self._find_current(entries)

        entry = RegistryEntry(
            filename=model_path.name,
            trained_at=_now_iso(),
            n_train_turns=n_train_turns,
            n_val_turns=eval_result.n_val_turns,
            macro_f1=eval_result.macro_f1,
            macro_auc=eval_result.macro_auc,
            threshold=eval_result.threshold,
            is_current=False,
        )

        should_promote = force_promote or self._passes_gate(entry, current)

        if should_promote:
            # Mark old current as not current
            for e in entries:
                e.is_current = False
            entry.is_current = True
            self._promote_file(model_path)
            if current:
                logger.info(
                    "Promoted %s (macro_f1=%.4f) over %s (macro_f1=%.4f)",
                    entry.filename,
                    entry.macro_f1,
                    current.filename,
                    current.macro_f1,
                )
            else:
                logger.info("Promoted %s as first registered model.", entry.filename)
        else:
            logger.info(
                "Model %s (macro_f1=%.4f) did NOT improve over current %s "
                "(macro_f1=%.4f, required delta=%.4f). Not promoted.",
                entry.filename,
                entry.macro_f1,
                current.filename if current else "(none)",
                current.macro_f1 if current else 0.0,
                self._min_improvement,
            )

        entries.append(entry)
        self._save(entries)
        return entry

    def get_current(self) -> RegistryEntry | None:
        """Return the currently active model's registry entry, or None."""
        entries = self._load()
        return self._find_current(entries)

    def list_versions(self) -> list[RegistryEntry]:
        """Return all registered versions, newest first."""
        entries = self._load()
        return sorted(entries, key=lambda e: e.trained_at, reverse=True)

    def current_model_path(self) -> Path | None:
        """Return the path to ``classifier_current.pkl``, or None if it doesn't exist."""
        if self._current_path.exists():
            return self._current_path
        return None

    # ── private helpers ───────────────────────────────────────────────────────

    def _passes_gate(
        self,
        candidate: RegistryEntry,
        current: RegistryEntry | None,
    ) -> bool:
        if current is None:
            return True  # Always promote first model
        return candidate.macro_f1 >= current.macro_f1 + self._min_improvement

    def _promote_file(self, model_path: Path) -> None:
        """Copy model_path to classifier_current.pkl."""
        shutil.copy2(model_path, self._current_path)

    def _load(self) -> list[RegistryEntry]:
        if not self._registry_path.exists():
            return []
        try:
            raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return [RegistryEntry(**item) for item in raw]
        except Exception as exc:
            logger.warning("Could not load classifier registry: %s", exc)
            return []

    def _save(self, entries: list[RegistryEntry]) -> None:
        self._oracle_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _find_current(entries: list[RegistryEntry]) -> RegistryEntry | None:
        for e in entries:
            if e.is_current:
                return e
        return None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
