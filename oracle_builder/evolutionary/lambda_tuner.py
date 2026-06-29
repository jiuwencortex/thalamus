# oracle_builder/evolutionary/lambda_tuner.py
# Tune the token-penalty weight λ per cluster from logged turn outcomes.
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...shared.outcome_scorer import compute_outcome_quality
from ...shared.turn_logger import TurnLogger
from ...shared.query_clusterer import QueryClusterer

logger = logging.getLogger(__name__)

_LAMBDA_GRID = [0.01, 0.05, 0.1, 0.2, 0.5]
_LAMBDA_FILE = "per_cluster_lambda.json"
_MIN_TURNS_PER_CLUSTER = 10   # minimum turns in a cluster to tune its λ


@dataclass
class ClusterLambdaEntry:
    """Per-cluster λ tuning result."""
    cluster_id: int
    n_turns: int
    best_lambda: float
    token_outcome_correlation: float   # Spearman rank correlation


@dataclass
class LambdaTuneResult:
    """Result of the per-cluster λ tuning."""
    per_cluster_lambda: dict[int, float]   # cluster_id → best λ
    default_lambda: float                  # used for clusters with too few data
    entries: list[ClusterLambdaEntry]


class LambdaTuner:
    """Tune the token-penalty weight λ per cluster from logged interaction data.

    The intuition:
    - λ controls how much the fitness function penalises token usage.
    - If, for a given cluster, turns with many tokens tend to produce *better*
      outcomes, then λ should be small (don't over-penalise using more context).
    - If turns with fewer tokens produce better outcomes, λ should be large.

    Concretely, for each cluster:
    1. Gather logged turns assigned to that cluster (using the saved clusterer).
    2. Compute Spearman rank correlation between context_tokens and outcome_quality.
    3. Map correlation to λ:
        correlation > +0.2   → λ = 0.01  (more tokens → better; penalise less)
        correlation in [-0.2, +0.2] → λ = 0.1 (no clear trend; use default)
        correlation < -0.2   → λ = 0.2   (fewer tokens → better; penalise more)

    Clusters with fewer than ``min_turns`` turns use the default λ unchanged.

    The result is written to ``per_cluster_lambda.json``.
    The oracle builder reads this file (when present) and injects per-cluster λ
    values into the fitness function instead of the global default.

    Usage::

        tuner = LambdaTuner(log_dir, oracle_dir, max_weeks=8)
        result = tuner.tune(default_lambda=0.1)
        tuner.save(result, oracle_dir)
    """

    def __init__(
        self,
        log_dir: Path,
        oracle_dir: Path,
        max_weeks: int = 8,
        min_turns: int = _MIN_TURNS_PER_CLUSTER,
    ):
        self._log_dir = log_dir
        self._oracle_dir = oracle_dir
        self._max_weeks = max_weeks
        self._min_turns = min_turns

    def tune(self, default_lambda: float = 0.1) -> LambdaTuneResult:
        """Run the per-cluster λ tuning.

        Parameters
        ----------
        default_lambda:
            Fallback λ for clusters with insufficient data (default: 0.1).

        Returns
        -------
        LambdaTuneResult with per-cluster λ values.
        """
        # ── Load clusterer ────────────────────────────────────────────────────
        pkl_path = self._oracle_dir / "context_configs.pkl"
        if not pkl_path.exists():
            logger.warning(
                "context_configs.pkl not found in %s; cannot assign clusters to turns. "
                "Run 'oracle_builder evolve' first.",
                self._oracle_dir,
            )
            return LambdaTuneResult(
                per_cluster_lambda={},
                default_lambda=default_lambda,
                entries=[],
            )

        clusterer = QueryClusterer.load(pkl_path)
        n_clusters = clusterer.n_clusters

        # ── Load turns ────────────────────────────────────────────────────────
        turn_logger = TurnLogger(self._log_dir)
        turns = turn_logger.load_turns(self._max_weeks)

        if not turns:
            logger.warning("No logged turns found; cannot tune λ.")
            return LambdaTuneResult(
                per_cluster_lambda={},
                default_lambda=default_lambda,
                entries=[],
            )

        # ── Assign each turn to a cluster via its stored embedding ────────────
        # Each turn has a query_embedding (the vector used at selection time).
        cluster_turns: dict[int, list[dict]] = {k: [] for k in range(n_clusters)}

        for turn in turns:
            embedding = turn.get("query_embedding")
            if embedding is None:
                continue
            try:
                cluster_id = int(clusterer._kmeans.predict(
                    np.array([embedding], dtype=np.float32)
                )[0])
                cluster_turns[cluster_id].append(turn)
            except Exception:
                continue

        # ── Tune λ per cluster ────────────────────────────────────────────────
        per_cluster_lambda: dict[int, float] = {}
        entries: list[ClusterLambdaEntry] = []

        for cluster_id in range(n_clusters):
            cluster_t = cluster_turns[cluster_id]
            n_turns = len(cluster_t)

            if n_turns < self._min_turns:
                logger.debug(
                    "Cluster %d: only %d turns (need %d); using default λ=%.3f.",
                    cluster_id, n_turns, self._min_turns, default_lambda,
                )
                entries.append(ClusterLambdaEntry(
                    cluster_id=cluster_id,
                    n_turns=n_turns,
                    best_lambda=default_lambda,
                    token_outcome_correlation=0.0,
                ))
                continue

            # Collect (context_tokens, outcome_quality) pairs
            token_counts: list[float] = []
            qualities: list[float] = []
            for turn in cluster_t:
                # token count: sum tokens across selected components if available,
                # otherwise proxy from config list size
                config = turn.get("context_config", {})
                n_components = (
                    len(config.get("skills", []))
                    + len(config.get("memory_sections", []))
                    + len(config.get("tools", []))
                )
                token_counts.append(float(n_components))
                qualities.append(compute_outcome_quality(turn))

            correlation = _spearman_correlation(
                np.array(token_counts), np.array(qualities)
            )

            # Map correlation to λ
            if correlation > 0.2:
                best_lambda = 0.01   # more context → better outcomes → penalise less
            elif correlation < -0.2:
                best_lambda = 0.2    # fewer components → better → penalise more
            else:
                best_lambda = default_lambda   # no clear trend

            per_cluster_lambda[cluster_id] = best_lambda
            entries.append(ClusterLambdaEntry(
                cluster_id=cluster_id,
                n_turns=n_turns,
                best_lambda=best_lambda,
                token_outcome_correlation=round(float(correlation), 4),
            ))
            logger.debug(
                "Cluster %d: n=%d  corr=%.3f  λ=%.3f",
                cluster_id, n_turns, correlation, best_lambda,
            )

        logger.info(
            "λ tuning complete: %d clusters tuned, %d used default.",
            sum(1 for e in entries if e.n_turns >= self._min_turns),
            sum(1 for e in entries if e.n_turns < self._min_turns),
        )

        return LambdaTuneResult(
            per_cluster_lambda=per_cluster_lambda,
            default_lambda=default_lambda,
            entries=entries,
        )

    def save(self, result: LambdaTuneResult, oracle_dir: Path) -> Path:
        """Write tuning result to per_cluster_lambda.json."""
        data = {
            "default_lambda": result.default_lambda,
            "per_cluster_lambda": {str(k): v for k, v in result.per_cluster_lambda.items()},
            "entries": [asdict(e) for e in result.entries],
        }
        path = oracle_dir / _LAMBDA_FILE
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Per-cluster λ values saved to %s", path)
        return path


def load_per_cluster_lambda(oracle_dir: Path) -> dict[int, float] | None:
    """Load per-cluster λ values from oracle_dir/per_cluster_lambda.json.

    Returns None if the file does not exist (caller uses global default λ).
    """
    path = oracle_dir / _LAMBDA_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("per_cluster_lambda", {})
        return {int(k): float(v) for k, v in raw.items()}
    except Exception as exc:
        logger.warning("Could not load per_cluster_lambda.json: %s", exc)
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without scipy dependency."""
    if len(x) < 3:
        return 0.0
    # Rank both arrays
    rx = _rank(x)
    ry = _rank(y)
    # Pearson correlation of ranks = Spearman
    x_mean = rx.mean()
    y_mean = ry.mean()
    num = float(((rx - x_mean) * (ry - y_mean)).sum())
    denom = float(np.sqrt(((rx - x_mean) ** 2).sum() * ((ry - y_mean) ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return num / denom


def _rank(arr: np.ndarray) -> np.ndarray:
    """Return ranks of elements (average ranks for ties)."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return ranks
