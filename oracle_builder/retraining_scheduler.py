# oracle_builder/retraining_scheduler.py
# Determine whether the classifier or oracle needs to be rebuilt.
#
# Logic:
#   should_retrain_classifier → True if ANY of:
#     1. drift_status.json reports drift_detected = True
#     2. The classifier has not been trained yet (classifier_current.pkl absent)
#     3. The number of new turns since the last training exceeds new_turns_threshold
#
#   should_rebuild_oracle → True if ANY of:
#     1. staleness_status.json reports stale = True  (new/removed/updated matrices)
#     2. context_configs.json does not exist
#     3. drift_detected AND n_recent_turns > drift_rebuild_min_turns
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .staleness_checker import load_staleness_status
from ..shared.distribution_monitor import load_drift_status
from ..shared.turn_logger import TurnLogger

logger = logging.getLogger(__name__)

_DEFAULT_NEW_TURNS_THRESHOLD = 200   # retrain classifier after this many new turns
_DEFAULT_DRIFT_REBUILD_MIN_TURNS = 50  # min recent turns required to trigger drift rebuild


@dataclass
class RebuildRecommendation:
    """Output of the retraining scheduler."""
    retrain_classifier: bool
    retrain_reasons: list[str]
    rebuild_oracle: bool
    rebuild_reasons: list[str]


class RetrainingScheduler:
    """Decide whether to retrain the classifier or rebuild the oracle.

    Usage::

        scheduler = RetrainingScheduler(oracle_dir, log_dir)
        rec = scheduler.check()
        if rec.retrain_classifier:
            # run train-classifier
        if rec.rebuild_oracle:
            # run evolve
    """

    def __init__(
        self,
        oracle_dir: Path,
        log_dir: Path | None = None,
        new_turns_threshold: int = _DEFAULT_NEW_TURNS_THRESHOLD,
        drift_rebuild_min_turns: int = _DEFAULT_DRIFT_REBUILD_MIN_TURNS,
    ):
        self._oracle_dir = oracle_dir
        self._log_dir = log_dir or (oracle_dir / "online_logs")
        self._new_turns_threshold = new_turns_threshold
        self._drift_rebuild_min_turns = drift_rebuild_min_turns

    def check(self) -> RebuildRecommendation:
        """Run all checks and return a RebuildRecommendation."""
        retrain_reasons: list[str] = []
        rebuild_reasons: list[str] = []

        # ── Load cached statuses ──────────────────────────────────────────────
        drift = load_drift_status(self._oracle_dir)
        staleness = load_staleness_status(self._oracle_dir)

        # ── Classifier checks ─────────────────────────────────────────────────
        classifier_path = self._oracle_dir / "classifier_current.pkl"
        if not classifier_path.exists():
            retrain_reasons.append(
                "No trained classifier found (classifier_current.pkl missing)."
            )

        if drift and drift.get("drift_detected"):
            retrain_reasons.append(
                f"Query distribution drift detected "
                f"(JS={drift.get('js_divergence', '?'):.4f}, "
                f"threshold={drift.get('js_threshold', '?')})."
            )

        # Count turns logged since the last classifier was trained
        n_new = self._count_new_turns(classifier_path if classifier_path.exists() else None)
        if n_new >= self._new_turns_threshold:
            retrain_reasons.append(
                f"{n_new} new turns logged since last training "
                f"(threshold={self._new_turns_threshold})."
            )

        # ── Oracle checks ─────────────────────────────────────────────────────
        oracle_path = self._oracle_dir / "context_configs.json"
        if not oracle_path.exists():
            rebuild_reasons.append(
                "context_configs.json does not exist — oracle has never been built."
            )

        if staleness and staleness.get("stale"):
            added = staleness.get("added_components", [])
            removed = staleness.get("removed_components", [])
            updated = staleness.get("updated_components", [])
            detail_parts = []
            if added:
                detail_parts.append(f"{len(added)} new component(s)")
            if removed:
                detail_parts.append(f"{len(removed)} removed component(s)")
            if updated:
                detail_parts.append(f"{len(updated)} updated matrix file(s)")
            rebuild_reasons.append(
                "Scoring matrices have changed: " + ", ".join(detail_parts) + "."
            )

        if drift and drift.get("drift_detected"):
            n_recent = drift.get("n_recent_turns", 0)
            if n_recent >= self._drift_rebuild_min_turns:
                rebuild_reasons.append(
                    f"Distribution drift with sufficient recent data "
                    f"({n_recent} recent turns ≥ {self._drift_rebuild_min_turns}) — "
                    "oracle cluster structure may be outdated."
                )

        rec = RebuildRecommendation(
            retrain_classifier=bool(retrain_reasons),
            retrain_reasons=retrain_reasons,
            rebuild_oracle=bool(rebuild_reasons),
            rebuild_reasons=rebuild_reasons,
        )

        _log_recommendation(rec)
        return rec

    # ── helpers ───────────────────────────────────────────────────────────────

    def _count_new_turns(self, classifier_path: Path | None) -> int:
        """Count turns logged after the classifier was last trained.

        If classifier_path is None (no classifier trained yet), counts all turns.
        """
        try:
            turn_logger = TurnLogger(self._log_dir)
            all_turns = turn_logger.load_turns(max_weeks=8)
        except Exception as exc:
            logger.warning("Could not load turns: %s", exc)
            return 0

        if classifier_path is None or not classifier_path.exists():
            return len(all_turns)

        clf_mtime = classifier_path.stat().st_mtime

        new_turns = 0
        for turn in all_turns:
            ts = turn.get("timestamp", "")
            try:
                from datetime import datetime, timezone
                turn_ts = datetime.fromisoformat(
                    ts.replace("Z", "+00:00")
                ).timestamp()
                if turn_ts > clf_mtime:
                    new_turns += 1
            except Exception:
                continue

        return new_turns


def _log_recommendation(rec: RebuildRecommendation) -> None:
    if rec.retrain_classifier:
        logger.info(
            "RECOMMEND: retrain classifier — %s",
            "; ".join(rec.retrain_reasons),
        )
    else:
        logger.info("Classifier is up-to-date.")

    if rec.rebuild_oracle:
        logger.info(
            "RECOMMEND: rebuild oracle — %s",
            "; ".join(rec.rebuild_reasons),
        )
    else:
        logger.info("Oracle is up-to-date.")
