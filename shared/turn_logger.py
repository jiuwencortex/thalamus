# online/turn_logger.py
# Log one JSONL record per agent turn: (embedding, context_config, outcome).
# Privacy: stores query embeddings (vectors), not raw query text.
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np


class TurnLogger:
    """Append one JSONL record per agent turn to a weekly-rotated log file.

    Log format (one JSON object per line):
    {
        "turn_id": "uuid",
        "timestamp": "2025-10-14T09:12:00Z",
        "query_embedding": [...],          # vector — not raw text (privacy)
        "context_config": {
            "skills": [...],
            "memory_sections": [...],
            "tools": [...]
        },
        "outcome": {
            "explicit_rating": null | "positive" | "negative",
            "implicit_signals": {
                "follow_up_correction": bool,
                "task_completed": bool,
                "conversation_length": int
            },
            "component_usage": {
                "skills_used": [...],
                "tools_called": [...]
            }
        }
    }

    Usage::

        logger = TurnLogger(log_dir)

        # After agent selects context (basic):
        turn_id = logger.log_turn(
            query_embedding=embedding_vector,
            context_config={"skills": [...], "memory_sections": [...], "tools": [...]},
        )

        # With off-policy exploration (randomly adds extra components 10% of the time):
        all_names = {"skills": ["sk-a", "sk-b"], "memory": ["mem-a"], "tools": ["tool-a"]}
        turn_id = logger.log_turn(
            query_embedding=embedding_vector,
            context_config=selected_config,
            exploration_rate=0.1,
            all_component_names=all_names,
        )

        # After conversation ends, update the outcome:
        logger.update_outcome(
            turn_id,
            task_completed=True,
            follow_up_correction=False,
            conversation_length=3,
            explicit_rating=None,
            skills_used=["devops-toolkit"],
            tools_called=["bash_exec"],
        )
    """

    _EMBEDDING_KEY = "query_embedding"

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ── writing ───────────────────────────────────────────────────────────────

    def log_turn(
        self,
        query_embedding: np.ndarray,
        context_config: dict,
        *,
        exploration_rate: float = 0.0,
        all_component_names: dict[str, list[str]] | None = None,
    ) -> str:
        """Append a turn record; return its turn_id for later outcome update.

        Parameters
        ----------
        query_embedding:
            TF-IDF or sentence-transformer vector for the user query.
        context_config:
            Dict with keys "skills", "memory_sections", "tools" (lists of names).
        exploration_rate:
            Probability (0–1) that each un-selected component is randomly added to the
            context for this turn.  When > 0 and all_component_names is provided, the
            classifier sees counterfactual inclusions which corrects the off-policy bias
            that arises from training only on components the selector already chose.
            Explored turns are flagged in the log so the trainer can down-weight or
            handle them appropriately.  Default 0.0 (no exploration).
        all_component_names:
            Full pool of available component names by type:
            {"skills": [...], "memory": [...], "tools": [...]}.
            Required when exploration_rate > 0; ignored otherwise.
        """
        selected_skills = list(context_config.get("skills", []))
        selected_memory = list(context_config.get("memory_sections", context_config.get("memory", [])))
        selected_tools = list(context_config.get("tools", []))

        explored_additions: dict[str, list[str]] = {"skills": [], "memory": [], "tools": []}
        is_explored = False

        if exploration_rate > 0.0 and all_component_names:
            for ctype, selected in [
                ("skills", selected_skills),
                ("memory", selected_memory),
                ("tools", selected_tools),
            ]:
                pool = all_component_names.get(ctype, [])
                selected_set = set(selected)
                extras = [
                    name for name in pool
                    if name not in selected_set and random.random() < exploration_rate
                ]
                if extras:
                    explored_additions[ctype] = extras
                    selected.extend(extras)
                    is_explored = True

        turn_id = str(uuid.uuid4())
        record: dict = {
            "turn_id": turn_id,
            "timestamp": _now_iso(),
            self._EMBEDDING_KEY: query_embedding.tolist(),
            "context_config": {
                "skills":          selected_skills,
                "memory_sections": selected_memory,
                "tools":           selected_tools,
            },
            "outcome": {
                "explicit_rating": None,
                "implicit_signals": {
                    "follow_up_correction": False,
                    "task_completed": False,
                    "conversation_length": 1,
                },
                "component_usage": {
                    "skills_used": [],
                    "tools_called": [],
                },
            },
        }

        if is_explored:
            record["exploration"] = {
                "explored": True,
                "exploration_rate": exploration_rate,
                "explored_additions": explored_additions,
            }

        self._append(record)
        return turn_id

    def update_outcome(
        self,
        turn_id: str,
        *,
        task_completed: bool = False,
        follow_up_correction: bool = False,
        conversation_length: int = 1,
        explicit_rating: Literal["positive", "negative"] | None = None,
        llm_judge_score: float | None = None,
        skills_used: list[str] | None = None,
        tools_called: list[str] | None = None,
    ) -> bool:
        """Rewrite the turn record identified by turn_id with outcome data.

        Scans the current week's log file for the record.  Returns True if
        the record was found and updated, False if not found.

        Parameters
        ----------
        llm_judge_score:
            Optional semantic quality score in [0, 1] produced by an external
            LLM judge (e.g. via ``outcome_scorer.score_from_llm_judge``).
            When provided this is stored in the record and used by
            ``compute_outcome_quality`` as a higher-priority signal than the
            implicit formula (but lower-priority than ``explicit_rating``).
        """
        log_path = self._current_log_path()
        if not log_path.exists():
            return False

        lines = log_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines: list[str] = []

        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue

            if rec.get("turn_id") == turn_id:
                outcome: dict = {
                    "explicit_rating": explicit_rating,
                    "implicit_signals": {
                        "follow_up_correction": follow_up_correction,
                        "task_completed": task_completed,
                        "conversation_length": conversation_length,
                    },
                    "component_usage": {
                        "skills_used": list(skills_used or []),
                        "tools_called": list(tools_called or []),
                    },
                }
                if llm_judge_score is not None:
                    outcome["llm_judge_score"] = float(
                        max(0.0, min(1.0, llm_judge_score))
                    )
                rec["outcome"] = outcome
                updated = True
            new_lines.append(json.dumps(rec, ensure_ascii=False))

        if updated:
            log_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        return updated

    # ── reading ───────────────────────────────────────────────────────────────

    def load_turns(self, max_weeks: int = 8) -> list[dict]:
        """Load turn records from up to max_weeks of log files.

        Returns records as raw dicts (embedding is a plain Python list).
        """
        turns: list[dict] = []
        for path in sorted(self._log_dir.glob("turns_*.jsonl"), reverse=True)[:max_weeks]:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return turns

    def count_turns(self, max_weeks: int = 8) -> int:
        """Return the total number of logged turns (fast scan)."""
        total = 0
        for path in sorted(self._log_dir.glob("turns_*.jsonl"), reverse=True)[:max_weeks]:
            total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return total

    # ── helpers ───────────────────────────────────────────────────────────────

    def _current_log_path(self) -> Path:
        """Return the log file path for the current ISO week."""
        now = datetime.now(tz=timezone.utc)
        week_tag = now.strftime("%Y-W%W")
        return self._log_dir / f"turns_{week_tag}.jsonl"

    def _append(self, record: dict) -> None:
        path = self._current_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
