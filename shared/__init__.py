"""Shared infrastructure: interaction logs, outcome scoring, classifier model, query clustering.

Modules
-------
turn_logger.py          TurnLogger — JSONL turn recording with off-policy exploration support
outcome_scorer.py       OutcomeScorer / compute_outcome_quality — scalar quality from signals
                        score_from_llm_judge — semantic quality via an external LLM judge
query_clusterer.py      QueryClusterer — TF-IDF or sentence-transformer + K-means clustering
classifier_model.py     ComponentInclusionClassifier — logistic regression per component
context_orderer.py      bookend_order — lost-in-the-middle mitigation ordering utility
distribution_monitor.py DistributionMonitor — JS divergence drift detection over query embeddings
                        load_drift_status — load saved drift_status.json

Used by both oracle_builder and context_selectors.  Neither package imports the other;
both import from this shared layer.

Usage:
    from jiuwenswarm.thalamus.shared import TurnLogger, OutcomeScorer, score_from_llm_judge
    from jiuwenswarm.thalamus.shared import DistributionMonitor, load_drift_status
"""

from .outcome_scorer import OutcomeScorer, compute_outcome_quality, score_from_llm_judge
from .turn_logger import TurnLogger
from .distribution_monitor import DistributionMonitor, load_drift_status

__all__ = [
    "TurnLogger",
    "OutcomeScorer",
    "compute_outcome_quality",
    "score_from_llm_judge",
    "DistributionMonitor",
    "load_drift_status",
]
