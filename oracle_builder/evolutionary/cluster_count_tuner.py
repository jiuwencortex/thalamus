# oracle_builder/evolutionary/cluster_count_tuner.py
# Select the optimal K (number of K-means clusters) from the data.
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

_K_GRID = [5, 10, 15, 20, 30, 50]


@dataclass
class KTuneEntry:
    """Metrics for one candidate K."""
    k: int
    inertia: float
    silhouette: float   # -1 to 1; higher is better separated clusters


@dataclass
class KTuneResult:
    """Result of the cluster count tuning search."""
    best_k: int
    entries: list[KTuneEntry]
    method: str   # "elbow" | "silhouette"


class ClusterCountTuner:
    """Select the best K for K-means clustering of query texts.

    Two signals are combined to find the elbow / silhouette optimum:

    1. **Inertia elbow** — fit K-means for each K, compute within-cluster
       sum of squares (inertia).  The optimal K is where the marginal
       reduction in inertia drops the most (first-derivative minimum).

    2. **Silhouette score** — average silhouette coefficient across all
       samples.  Higher means more compact and well-separated clusters.

    The final K is chosen by a simple rule:
    - If the elbow K and the silhouette-max K agree (or are adjacent), use
      the silhouette-max K (slightly more data-driven signal).
    - Otherwise default to the silhouette-max K.

    Usage::

        tuner = ClusterCountTuner()
        texts = [...]   # all example_input texts from scoring matrices
        result = tuner.tune(texts)
        print(result.best_k)   # use this as --n-clusters
    """

    def __init__(
        self,
        k_grid: list[int] | None = None,
        max_features: int = 2000,
        random_state: int = 42,
    ):
        self._k_grid = sorted(k_grid or _K_GRID)
        self._max_features = max_features
        self._random_state = random_state

    def tune(self, texts: list[str]) -> KTuneResult:
        """Find the best K for the given set of query texts.

        Parameters
        ----------
        texts:
            All example_input query strings from scoring matrices.

        Returns
        -------
        KTuneResult with best_k selected by combined elbow + silhouette.
        """
        if len(texts) < 2:
            logger.warning("Too few texts to tune K; returning K=1.")
            return KTuneResult(best_k=1, entries=[], method="default")

        # Filter K values that are feasible (K < n_texts)
        feasible_ks = [k for k in self._k_grid if k < len(texts)]
        if not feasible_ks:
            best_k = max(1, len(texts) - 1)
            logger.warning("All K values exceed text count (%d); using K=%d.", len(texts), best_k)
            return KTuneResult(best_k=best_k, entries=[], method="default")

        # Embed texts with TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts).toarray()

        entries: list[KTuneEntry] = []
        for k in feasible_ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=self._random_state)
            labels = km.fit_predict(X)

            sil = 0.0
            if k > 1:
                try:
                    sil = float(silhouette_score(X, labels, sample_size=min(len(texts), 1000)))
                except Exception:
                    sil = 0.0

            entries.append(KTuneEntry(k=k, inertia=float(km.inertia_), silhouette=sil))
            logger.debug("K=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)

        # ── Elbow: find K where second-derivative of inertia is maximised ────
        inertias = np.array([e.inertia for e in entries])
        ks = np.array([e.k for e in entries])

        elbow_k = feasible_ks[0]
        if len(inertias) >= 3:
            # Normalise inertia to [0, 1]
            inertia_norm = (inertias - inertias.min()) / max(inertias.max() - inertias.min(), 1e-9)
            # First differences of normalised inertia
            first_diff = np.diff(inertia_norm)
            # Second differences (curvature) — minimum = sharpest elbow
            second_diff = np.diff(first_diff)
            elbow_idx = int(np.argmin(second_diff)) + 1  # +1 for offset from diffs
            elbow_k = int(ks[elbow_idx])
        elif len(inertias) == 2:
            elbow_k = int(ks[0])

        # ── Silhouette: K with highest silhouette score ───────────────────────
        sil_scores = [e.silhouette for e in entries]
        sil_k = int(entries[int(np.argmax(sil_scores))].k)

        logger.info("Elbow K=%d   Silhouette-max K=%d", elbow_k, sil_k)

        # Combine: prefer silhouette-max unless it's very far from the elbow
        if abs(sil_k - elbow_k) <= 5:
            best_k = sil_k
            method = "silhouette"
        else:
            # Large disagreement — pick the geometric mean (round to nearest feasible K)
            geo_mean = int(round((elbow_k * sil_k) ** 0.5))
            best_k = min(feasible_ks, key=lambda k: abs(k - geo_mean))
            method = "elbow_silhouette_compromise"

        logger.info("Selected K=%d (method=%s)", best_k, method)
        return KTuneResult(best_k=best_k, entries=entries, method=method)
