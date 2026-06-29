# recommendation_matrix/tools/composer.py
# ToolMatrixComposer: thin orchestrator for the tool scoring pipeline.
from __future__ import annotations

import json
import re
from pathlib import Path

from openjiuwen.core.foundation.llm import Model

from .scanner import ToolCodeScanner
from .template import TOOL_PROMPT_TEMPLATE
from ..shared.all_items_evaluator import AllItemsEvaluator
from ..shared.changed_items_determiner import ChangedItemsDeterminer
from ..shared.fingerprint import ComponentRecord
from ..shared.queries_generator import QueriesGenerator
from ..shared.state_saver import StateSaver
from ..shared.summary_printer import SummaryPrinter

_STATE_FILE = "matrix_state_tools.json"
_FILE_PREFIX = "tool_"
_COMPONENT_TYPE = "tool"


class ToolMatrixComposer:
    """Compose the tool relevance matrix.

    Stage 1: For each tool, generate N (query, expected_answer) pairs.
    Stage 2: For each (tool, query) pair, run the LLM and score outputs.
    Result: scoring_matrix_tool_<name>.json files ready for tool routing.
    """

    def __init__(
        self,
        tool_dirs: list[Path],
        matrix_dir: Path,
        model: Model,
        model_name: str,
        n_examples: int = 15,
        parallel: int = 5,
        timeout: float = 3600.0,
        temperature: float = 0.2,
        max_tokens: int = 57000,
        metrics: list[str] | None = None,
        judge_model: str = "gpt-4o-mini",
        judge_api_key: str | None = None,
        judge_api_base: str = "https://api.openai.com/v1",
        eval_combination_size: int = 1,
    ):
        self._scanner = ToolCodeScanner(tool_dirs)
        self._matrix_dir = matrix_dir
        self._determiner = ChangedItemsDeterminer(matrix_dir, state_file=_STATE_FILE)
        self._generator = QueriesGenerator(
            model, model_name, n_examples, parallel,
            prompt_template=TOOL_PROMPT_TEMPLATE,
        )
        self._evaluator = AllItemsEvaluator(
            model, model_name, matrix_dir, parallel,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            file_prefix=_FILE_PREFIX,
            component_type=_COMPONENT_TYPE,
            metrics=metrics,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            judge_api_base=judge_api_base,
            eval_combination_size=eval_combination_size,
        )
        self._saver = StateSaver(matrix_dir, state_file=_STATE_FILE, component_type=_COMPONENT_TYPE)
        self._printer = SummaryPrinter()

    async def build(self, force: bool = False, only: list[str] | None = None) -> None:
        tools = self._scanner.scan_and_filter(only)
        if not tools:
            print("No tools found (check that --tools-dir paths are correct).")
            return

        changed, skipped = self._determiner.determine(tools, force)
        if not changed:
            print("Nothing to rebuild.")
            if skipped:
                print(f"Skipped: {', '.join(skipped)}")
            return

        gen_results = await self._generator.generate_for_items(changed)
        states, llm_calls = await self._evaluator.evaluate_all(gen_results)
        self._saver.save(tools, skipped, states)
        self._printer.print(changed, skipped, llm_calls)
