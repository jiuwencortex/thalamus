# shared/distribution_monitor.py
# Monitor query distribution drift between a recent window and a longer baseline.
# Uses Jensen-Shannon divergence over quantised embedding histograms to detect
# distribution shift without storing raw text (only embeddings).
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .turn_logger import TurnLogger

logger = logging.getLogger(__name__)

_JS_DRIFT_THRESHOLD = 0.15   # JS divergence above this → flag drift
_N_BINS = 32                  # histogram bins per embedding dimension (after PCA proj.)
_DRIFT_FILE = "drift_status.json"


class DistributionMonitor:
    """Detect query distribution drift from logged turn embeddings.

    The monitor compares two time windows:
    - **Recent** window  : last ``recent_weeks`` weeks (default 1)
    - **Baseline** window: last ``baseline_weeks`` weeks (default 4)

    For each window it builds a flat histogram over the first principal
    direction of the embedding space (determined from the baseline) and
    computes the Jensen-Shannon divergence between the two histograms.

    A JS divergence above ``js_threshold`` (default 0.15) indicates that
    the query distribution has shifted significantly and the oracle may need
    to be rebuilt.

    The result is written to ``drift_status.json`` in ``oracle_dir``.

    Usage::

        monitor = DistributionMonitor(log_dir, oracle_dir)
        result = monitor.check()
        monitor.save(result, oracle_dir)
    """

    def __init__(
        self,
        log_dir: Path,
        oracle_dir: Path,
        recent_weeks: int = 1,
        baseline_weeks: int = 4,
        n_bins: int = _N_BINS,
        js_threshold: float = _JS_DRIFT_THRESHOLD,
    ):
        self._log_dir = log_dir
        self._oracle_dir = oracle_dir
        self._recent_weeks = recent_weeks
        self._baseline_weeks = baseline_weeks
        self._n_bins = n_bins
        self._js_threshold = js_threshold

    # ── public API ────────────────────────────────────────────────────────────

    def check(self) -> DriftStatus:
        """Load turn embeddings, compute JS divergence, return DriftStatus."""
        turn_logger = TurnLogger(self._log_dir)
        all_turns = turn_logger.load_turns(max_weeks=self._baseline_weeks)

        if not all_turns:
            logger.warning("No turns found; cannot assess distribution drift.")
            return DriftStatus(
                js_divergence=0.0,
                drift_detected=False,
                n_recent=0,
                n_baseline=0,
                threshold=self._js_threshold,
                message="No logged turns found.",
            )

        now = datetime.now(tz=timezone.utc)
        recent_cutoff = now - timedelta(weeks=self._recent_weeks)

        recent_turns = [t for t in all_turns if _turn_timestamp(t) >= recent_cutoff]
        baseline_turns = all_turns   # baseline = all loaded turns

        n_recent = len(recent_turns)
        n_baseline = len(baseline_turns)

        if n_recent < 5:
            logger.info(
                "Only %d recent turns (< 5); skipping drift check.", n_recent
            )
            return DriftStatus(
                js_divergence=0.0,
                drift_detected=False,
                n_recent=n_recent,
                n_baseline=n_baseline,
                threshold=self._js_threshold,
                message=f"Too few recent turns ({n_recent}) to assess drift.",
            )

        recent_emb = _extract_embeddings(recent_turns)
        baseline_emb = _extract_embeddings(baseline_turns)

        # Project onto first principal component of baseline for a 1-D histogram
        direction = _first_pc(baseline_emb)
        recent_proj = recent_emb @ direction
        baseline_proj = baseline_emb @ direction

        js = _js_divergence_1d(recent_proj, baseline_proj, n_bins=self._n_bins)
        drift_detected = bool(js > self._js_threshold)

        if drift_detected:
            msg = (
                f"Drift detected: JS={js:.4f} > threshold={self._js_threshold:.4f}. "
                f"Recent={n_recent} turns vs baseline={n_baseline} turns. "
                "Consider rebuilding the oracle."
            )
            logger.warning(msg)
        else:
            msg = (
                f"No drift detected: JS={js:.4f} ≤ threshold={self._js_threshold:.4f}. "
                f"Recent={n_recent} turns vs baseline={n_baseline} turns."
            )
            logger.info(msg)

        return DriftStatus(
            js_divergence=round(float(js), 6),
            drift_detected=drift_detected,
            n_recent=n_recent,
            n_baseline=n_baseline,
            threshold=self._js_threshold,
            message=msg,
        )

    def save(self, status: DriftStatus, oracle_dir: Path) -> Path:
        """Write drift status to drift_status.json."""
        data = {
            "checked_at": _now_iso(),
            "js_divergence": status.js_divergence,
            "drift_detected": status.drift_detected,
            "n_recent_turns": status.n_recent,
            "n_baseline_turns": status.n_baseline,
            "js_threshold": status.threshold,
            "message": status.message,
        }
        path = oracle_dir / _DRIFT_FILE
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Drift status saved to %s", path)
        return path


class DriftStatus:
    """Result of a drift check."""

    def __init__(
        self,
        *,
        js_divergence: float,
        drift_detected: bool,
        n_recent: int,
        n_baseline: int,
        threshold: float,
        message: str,
    ):
        self.js_divergence = js_divergence
        self.drift_detected = drift_detected
        self.n_recent = n_recent
        self.n_baseline = n_baseline
        self.threshold = threshold
        self.message = message


def load_drift_status(oracle_dir: Path) -> dict | None:
    """Load the last saved drift_status.json from oracle_dir.

    Returns None if the file does not exist.
    """
    path = oracle_dir / _DRIFT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load drift_status.json: %s", exc)
        return None


# ── internal helpers ──────────────────────────────────────────────────────────

def _extract_embeddings(turns: list[dict]) -> np.ndarray:
    """Return (N, D) float32 array of query embeddings from turns."""
    vecs = [t["query_embedding"] for t in turns if t.get("query_embedding") is not None]
    return np.array(vecs, dtype=np.float32)


def _first_pc(X: np.ndarray) -> np.ndarray:
    """Return the first principal component of X as a unit vector.

    Uses the power iteration method — no scipy/sklearn dependency.
    """
    if X.shape[0] < 2:
        # Can't do PCA; return first basis vector
        d = X.shape[1] if X.ndim == 2 else 1
        v = np.zeros(d, dtype=np.float32)
        v[0] = 1.0
        return v

    X_c = X - X.mean(axis=0)
    # Power iteration: repeatedly apply X^T X to a random vector
    rng = np.random.default_rng(seed=42)
    v = rng.standard_normal(X_c.shape[1]).astype(np.float32)
    for _ in range(30):
        v = X_c.T @ (X_c @ v)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            break
        v /= norm
    return v


def _js_divergence_1d(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    n_bins: int,
) -> float:
    """Compute JS divergence between two 1-D sample distributions.

    Both arrays are projected onto a shared bin grid determined by the union
    of their ranges, then smoothed with a small epsilon before computing
    KL divergence in both directions.
    """
    lo = float(min(p_samples.min(), q_samples.min()))
    hi = float(max(p_samples.max(), q_samples.max()))
    if hi == lo:
        return 0.0

    edges = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_samples, bins=edges, density=False)
    q_hist, _ = np.histogram(q_samples, bins=edges, density=False)

    # Normalise to probability distributions
    eps = 1e-10
    p = p_hist.astype(np.float64) + eps
    q = q_hist.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # JS is in [0, log(2)]; normalise to [0, 1]
    return float(min(1.0, js / math.log(2)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(P || Q) — both arrays must be positive."""
    return float(np.sum(p * np.log(p / q)))


def _turn_timestamp(turn: dict) -> datetime:
    ts = turn.get("timestamp", "")
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
