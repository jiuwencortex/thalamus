# component_scoring/cli.py
# Unified entry point for building component scoring matrices.
# Supports: python -m jiuwenswarm.tools.component_scoring build --type skills|memory|tools
from __future__ import annotations

import argparse
import sys

from pathlib import Path


def make_parser():
    p = argparse.ArgumentParser(
        prog="python -m jiuwenswarm.tools.component_scoring",
        description=(
            "Build recommendation-matrix files for skills, memory sections, and tools.\n\n"
            "Stage 1: LLM reads each component and invents (query, expected_answer) pairs.\n"
            "Stage 2: LLM evaluates each component against those pairs; outputs are scored.\n"
            "Result: scoring_matrix_<type>_<name>.json files ready for routing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    build = sub.add_parser("build", help="Build or update the recommendation matrix")

    build.add_argument(
        "--type",
        dest="build_type",
        choices=["skills", "memory", "tools", "enrich", "all"],
        default="all",
        help="Which component type to build/run (default: all). 'enrich' blends real interaction logs into existing matrix files without LLM calls.",
    )
    build.add_argument(
        "--skills-dir",
        type=Path,
        help="Directory with skill subdirectories (each must have SKILL.md). Required for --type skills|all",
    )
    build.add_argument(
        "--project-dir",
        type=Path,
        help="Directory containing project.md and user.md. Required for --type memory|all",
    )
    build.add_argument(
        "--tools-dir",
        action="append",
        type=Path,
        help=(
            "Directory containing tool Python source files (repeatable). "
            "Required for --type tools|all. Example paths: "
            "<agent-core>/openjiuwen/harness/tools, "
            "<jiuwenswarm>/jiuwenswarm/agents/harness/common/tools"
        ),
    )
    build.add_argument("--matrix-dir", required=True, type=Path,
                       help="Output directory for scoring_matrix_*.json and matrix_state_*.json")
    build.add_argument("--model", default="",
                       help="LLM model name, e.g. gpt-4o-mini (required unless --type enrich)")
    build.add_argument("--provider", default="openai",
                       help="LLM provider name (default: openai)")
    build.add_argument("--api-key", default="",
                       help="API key for the LLM provider (required unless --dry-run)")
    build.add_argument("--api-base", default="https://api.openai.com/v1",
                       help="API base URL (default: https://api.openai.com/v1)")
    build.add_argument("--timeout", type=float, default=3600.0,
                       help="Per-request timeout in seconds (default: 3600)")
    build.add_argument("--temperature", type=float, default=0.2,
                       help="LLM temperature during component execution (default: 0.2)")
    build.add_argument("--max-tokens", type=int, default=57000,
                       help="Max tokens per LLM call during execution (default: 57000)")
    build.add_argument("--n-examples", type=int, default=20,
                       help="Number of (query, answer) pairs to generate per component (default: 20)")
    build.add_argument("--parallel", type=int, default=5,
                       help="Max concurrent LLM calls (default: 5)")
    build.add_argument("--force", action="store_true",
                       help="Ignore fingerprint cache and rebuild all components")
    build.add_argument("--only", action="append", metavar="NAME",
                       help="Rebuild only this component (repeatable; omit to rebuild all changed)")
    build.add_argument("--dry-run", action="store_true",
                       help="Show what would be rebuilt without making any LLM calls")
    build.add_argument("--prune", action="store_true",
                       help="Delete matrix files for components no longer present")
    build.add_argument("--cross-eval", action="store_true",
                       help="Evaluate every component against ALL generated queries (higher cost)")
    build.add_argument("--verbose", action="store_true",
                       help="Enable debug logging")
    build.add_argument("--log-dir", type=Path,
                       help="Directory containing JSONL turn logs for --type enrich (default: <matrix-dir>/online_logs)")
    build.add_argument("--n-needed", type=int, default=100,
                       help="Real samples per component at which synthetic scores are fully replaced (default: 100, used with --type enrich)")
    build.add_argument("--max-weeks", type=int, default=8,
                       help="Number of weekly log files to scan (default: 8, used with --type enrich)")
    build.add_argument(
        "--judge-model", dest="judge_model", default="gpt-4o-mini",
        help=(
            "LLM model used as a quality judge.  When 'llm_judge' is included in --metrics, "
            "this model rates semantic correctness 1–10.  Also used during --type enrich "
            "(default: gpt-4o-mini)."
        ),
    )
    build.add_argument(
        "--judge-api-key", dest="judge_api_key", default=None,
        help="API key for the LLM judge (falls back to OPENAI_API_KEY env var).",
    )
    build.add_argument(
        "--judge-api-base", dest="judge_api_base", default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL for the LLM judge (default: https://api.openai.com/v1).",
    )
    build.add_argument(
        "--metrics", dest="metrics", default=None,
        help=(
            "Comma-separated list of scoring metrics to compute.  "
            "Default: f1,bigram_f1,bag_of_words,length_ratio.  "
            "Available: f1, bigram_f1, bag_of_words, length_ratio, bert_score, llm_judge.  "
            "Example: --metrics f1,bert_score,llm_judge"
        ),
    )
    build.add_argument(
        "--eval-combination-size", dest="eval_combination_size", type=int, default=1,
        help=(
            "When > 1, evaluate each component as part of random N-component combinations "
            "using leave-one-out delta scoring.  This detects components that are only useful "
            "in combination with others.  Default: 1 (isolated evaluation, fast)."
        ),
    )

    args = p.parse_args()

    if args.command is None:
        p.print_help()
        sys.exit(0)

    return args
