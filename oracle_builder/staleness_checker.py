# oracle_builder/staleness_checker.py
# Detect whether context_configs.json is stale relative to the current set
# of component scoring matrices.
#
# Staleness signals:
#   • New matrices exist that are not represented in the oracle
#   • Matrices that were in the oracle have been removed
#   • One or more matrices have a newer mtime than context_configs.json
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STALENESS_FILE = "staleness_status.json"

_PATTERN_MAP = [
    ("scoring_matrix_skill_*.json", "skill"),
    ("scoring_matrix_mem_*.json",   "memory_section"),
    ("scoring_matrix_tool_*.json",  "tool"),
]


@dataclass
class StalenessStatus:
    """Result of a staleness check."""
    stale: bool
    oracle_exists: bool
    oracle_mtime: str                      # ISO timestamp or ""
    n_oracle_components: int
    n_current_matrices: int
    added_components: list[str] = field(default_factory=list)    # in matrices but not oracle
    removed_components: list[str] = field(default_factory=list)  # in oracle but not matrices
    updated_components: list[str] = field(default_factory=list)  # matrix newer than oracle
    message: str = ""


class StalenessChecker:
    """Check whether the oracle is stale relative to current scoring matrices.

    Usage::

        checker = StalenessChecker(oracle_dir)
        status = checker.check()
        checker.save(status, oracle_dir)
    """

    def __init__(self, oracle_dir: Path):
        self._oracle_dir = oracle_dir

    def check(self) -> StalenessStatus:
        """Run the staleness check and return a StalenessStatus."""
        oracle_path = self._oracle_dir / "context_configs.json"

        if not oracle_path.exists():
            return StalenessStatus(
                stale=True,
                oracle_exists=False,
                oracle_mtime="",
                n_oracle_components=0,
                n_current_matrices=0,
                message="context_configs.json does not exist — oracle has never been built.",
            )

        oracle_mtime = oracle_path.stat().st_mtime
        oracle_mtime_iso = _mtime_iso(oracle_mtime)

        # ── Names in current oracle ───────────────────────────────────────────
        oracle_names = _load_oracle_component_names(oracle_path)

        # ── Names in current scoring matrices ─────────────────────────────────
        matrix_files: dict[str, Path] = {}   # component_name → file path
        for pat, _ in _PATTERN_MAP:
            for p in sorted(self._oracle_dir.glob(pat)):
                name = _component_name_from_matrix(p)
                if name:
                    matrix_files[name] = p

        current_names = set(matrix_files)

        # ── Diff ──────────────────────────────────────────────────────────────
        added = sorted(current_names - oracle_names)
        removed = sorted(oracle_names - current_names)

        # Components whose matrix file is newer than the oracle JSON
        updated = sorted(
            name
            for name, path in matrix_files.items()
            if name in oracle_names and path.stat().st_mtime > oracle_mtime
        )

        stale = bool(added or removed or updated)

        parts = []
        if added:
            parts.append(f"{len(added)} new component(s): {', '.join(added[:5])}" +
                         (" …" if len(added) > 5 else ""))
        if removed:
            parts.append(f"{len(removed)} removed component(s): {', '.join(removed[:5])}" +
                         (" …" if len(removed) > 5 else ""))
        if updated:
            parts.append(f"{len(updated)} updated matrix file(s): {', '.join(updated[:5])}" +
                         (" …" if len(updated) > 5 else ""))

        if stale:
            message = "Oracle is stale — " + "; ".join(parts) + ". Rebuild recommended."
            logger.warning(message)
        else:
            message = (
                f"Oracle is up-to-date: {len(oracle_names)} component(s), "
                f"{len(current_names)} matrix file(s)."
            )
            logger.info(message)

        return StalenessStatus(
            stale=stale,
            oracle_exists=True,
            oracle_mtime=oracle_mtime_iso,
            n_oracle_components=len(oracle_names),
            n_current_matrices=len(current_names),
            added_components=added,
            removed_components=removed,
            updated_components=updated,
            message=message,
        )

    def save(self, status: StalenessStatus, oracle_dir: Path) -> Path:
        """Write staleness_status.json to oracle_dir."""
        data = asdict(status)
        data["checked_at"] = _now_iso()
        path = oracle_dir / _STALENESS_FILE
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Staleness status saved to %s", path)
        return path


def load_staleness_status(oracle_dir: Path) -> dict | None:
    """Load the last saved staleness_status.json from oracle_dir.

    Returns None if the file does not exist.
    """
    path = oracle_dir / _STALENESS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load staleness_status.json: %s", exc)
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_oracle_component_names(oracle_path: Path) -> set[str]:
    """Extract the set of component names referenced in context_configs.json."""
    try:
        data = json.loads(oracle_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse context_configs.json: %s", exc)
        return set()

    names: set[str] = set()
    # context_configs.json stores a list of cluster entries; each entry has
    # per-budget configs with lists of skill/memory/tool names.
    clusters = data.get("clusters", [])
    if not clusters:
        # Flat format fallback: top-level list
        if isinstance(data, list):
            clusters = data

    for cluster in clusters:
        for budget_config in cluster.get("configs", {}).values() if isinstance(cluster, dict) else []:
            names.update(budget_config.get("skills", []))
            names.update(budget_config.get("memory_sections", []))
            names.update(budget_config.get("tools", []))

    return names


def _component_name_from_matrix(path: Path) -> str | None:
    """Extract the component_name field from a scoring matrix JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("component_name") or data.get("skill_name")
        if name:
            return str(name)
        # Fallback: derive from filename
        stem = path.stem
        for prefix in ("scoring_matrix_skill_", "scoring_matrix_mem_", "scoring_matrix_tool_"):
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return None


def _mtime_iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
