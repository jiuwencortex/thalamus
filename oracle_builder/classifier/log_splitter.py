# oracle_builder/classifier/log_splitter.py
# Split logged turns into training and validation sets by timestamp.
from __future__ import annotations

from pathlib import Path

from ...shared.turn_logger import TurnLogger


class LogSplitter:
    """Split logged turns into a chronological train / validation partition.

    Turns are sorted by ``timestamp`` field.  The most-recent
    ``validation_fraction`` of turns form the held-out validation set;
    the remainder are used for training.

    Chronological splitting (rather than random) ensures the validation set
    always reflects the latest agent behaviour — the same condition under which
    a newly trained model will be deployed.

    Parameters
    ----------
    log_dir:
        Directory containing weekly JSONL turn files.
    max_weeks:
        How many weekly files to scan (default: 8).
    validation_fraction:
        Fraction of turns to reserve for validation (default: 0.20 → 20 %).
    """

    def __init__(
        self,
        log_dir: Path,
        max_weeks: int = 8,
        validation_fraction: float = 0.20,
    ):
        if not (0.0 < validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must be in (0, 1); got {validation_fraction}"
            )
        self._log_dir = log_dir
        self._max_weeks = max_weeks
        self._validation_fraction = validation_fraction

    def split(self) -> tuple[list[dict], list[dict]]:
        """Load and split all turns.

        Returns
        -------
        (train_turns, val_turns)
            Both lists contain raw turn dicts.  val_turns are the most recent
            turns by timestamp; train_turns are everything before them.
        """
        turn_logger = TurnLogger(self._log_dir)
        all_turns = turn_logger.load_turns(self._max_weeks)

        if not all_turns:
            return [], []

        # Sort by timestamp ascending so the chronological split is deterministic.
        all_turns.sort(key=lambda t: t.get("timestamp", ""))

        n_total = len(all_turns)
        n_val = max(1, int(round(n_total * self._validation_fraction)))
        n_train = n_total - n_val

        train_turns = all_turns[:n_train]
        val_turns = all_turns[n_train:]

        return train_turns, val_turns

    @property
    def validation_fraction(self) -> float:
        return self._validation_fraction
