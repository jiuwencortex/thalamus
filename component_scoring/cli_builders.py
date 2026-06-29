# component_scoring/cli.py
# Unified entry point for building component scoring matrices.
# Supports: python -m jiuwenswarm.tools.component_scoring build --type skills|memory|tools
from __future__ import annotations

import argparse
import re
import sys

from openjiuwen.core.foundation.llm import Model

from .shared.dry_run import run_dry_run


def _metric_kwargs(args: argparse.Namespace) -> dict:
    """Extract metric configuration from CLI args for passing to composer __init__."""
    metrics_raw = getattr(args, "metrics", None)
    metrics = None
    if metrics_raw:
        metrics = [m.strip() for m in metrics_raw.split(",") if m.strip()]
    return {
        "metrics": metrics,
        "judge_model": getattr(args, "judge_model", "gpt-4o-mini"),
        "judge_api_key": getattr(args, "judge_api_key", None),
        "judge_api_base": getattr(args, "judge_api_base", "https://api.openai.com/v1"),
        "eval_combination_size": getattr(args, "eval_combination_size", 1),
    }


async def build_skills(args: argparse.Namespace, model: Model, model_name: str) -> None:
    from .skills.composer import SkillMatrixComposer
    from .skills.scanner import ExistingSkillsScanner

    if not args.skills_dir:
        print("ERROR: --skills-dir is required for --type skills", file=sys.stderr)
        sys.exit(1)
    if not args.skills_dir.exists():
        print(f"ERROR: --skills-dir does not exist: {args.skills_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        scanner = ExistingSkillsScanner(args.skills_dir)
        skills = scanner.scan_and_filter(args.only)
        print(f"Skills directory: {args.skills_dir}")
        run_dry_run(skills, args.matrix_dir, args.force,
                    state_file="matrix_state_skills.json", label="skill")
        return

    composer = SkillMatrixComposer(
        skills_dir=args.skills_dir,
        matrix_dir=args.matrix_dir,
        model=model,
        model_name=model_name,
        n_examples=args.n_examples,
        parallel=args.parallel,
        cross_eval=args.cross_eval,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        **_metric_kwargs(args),
    )
    await composer.build(force=args.force, only=args.only)

    if args.prune:
        scanner = ExistingSkillsScanner(args.skills_dir)
        current_safe = {re.sub(r"[^\w-]", "_", s.name) for s in scanner.scan()}
        for f in args.matrix_dir.glob("scoring_matrix_skill_*.json"):
            stem = f.stem[len("scoring_matrix_skill_"):]
            if stem not in current_safe:
                f.unlink()
                print(f"Pruned: {f.name}")


async def build_memory(args: argparse.Namespace, model: Model, model_name: str) -> None:
    from .memory.composer import MemoryMatrixComposer
    from .memory.scanner import MemorySectionScanner

    if not args.project_dir:
        print("ERROR: --project-dir is required for --type memory", file=sys.stderr)
        sys.exit(1)
    if not args.project_dir.exists():
        print(f"ERROR: --project-dir does not exist: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        scanner = MemorySectionScanner(args.project_dir)
        sections = scanner.scan_and_filter(args.only)
        print(f"Project directory: {args.project_dir}")
        run_dry_run(sections, args.matrix_dir, args.force,
                    state_file="matrix_state_memory.json", label="memory section")
        return

    composer = MemoryMatrixComposer(
        project_dir=args.project_dir,
        matrix_dir=args.matrix_dir,
        model=model,
        model_name=model_name,
        n_examples=args.n_examples,
        parallel=args.parallel,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        **_metric_kwargs(args),
    )
    await composer.build(force=args.force, only=args.only)


async def build_tools(args: argparse.Namespace, model: Model, model_name: str) -> None:
    from .tools.composer import ToolMatrixComposer
    from .tools.scanner import ToolCodeScanner

    if not args.tools_dir:
        print("ERROR: --tools-dir is required for --type tools", file=sys.stderr)
        sys.exit(1)
    tool_dirs = [d for d in args.tools_dir if d.exists()]
    missing = [str(d) for d in args.tools_dir if not d.exists()]
    if missing:
        print(f"ERROR: --tools-dir does not exist: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        scanner = ToolCodeScanner(tool_dirs)
        tools = scanner.scan_and_filter(args.only)
        print(f"Tools directories: {', '.join(str(d) for d in tool_dirs)}")
        run_dry_run(tools, args.matrix_dir, args.force,
                    state_file="matrix_state_tools.json", label="tool")
        return

    composer = ToolMatrixComposer(
        tool_dirs=tool_dirs,
        matrix_dir=args.matrix_dir,
        model=model,
        model_name=model_name,
        n_examples=args.n_examples,
        parallel=args.parallel,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        **_metric_kwargs(args),
    )
    await composer.build(force=args.force, only=args.only)
