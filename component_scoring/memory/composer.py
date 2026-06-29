# recommendation_matrix/memory/composer.py
# MemoryMatrixComposer: thin orchestrator for the memory section scoring pipeline.
from __future__ import annotations

from pathlib import Path

from openjiuwen.core.foundation.llm import Model

from ..shared.all_items_evaluator import AllItemsEvaluator
from ..shared.changed_items_determiner import ChangedItemsDeterminer
from ..shared.queries_generator import QueriesGenerator
from ..shared.state_saver import StateSaver
from ..shared.summary_printer import SummaryPrinter
from .template import MEMORY_PROMPT_TEMPLATE
from .scanner import MemorySectionScanner

_STATE_FILE = "matrix_state_memory.json"
_FILE_PREFIX = "mem_"
_COMPONENT_TYPE = "memory_section"


class MemoryMatrixComposer:
    """Compose the memory section relevance matrix.

    Stage 1: For each ## section in project.md / user.md, generate N (query, expected_answer) pairs.
    Stage 2: For each (section, query) pair, run the LLM and score outputs.
    Result: scoring_matrix_mem_<name>.json files ready for memory routing.
    """

    def __init__(
        self,
        project_dir: Path,
        matrix_dir: Path,
        model: Model,
        model_name: str,
        n_examples: int = 20,
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
        self._scanner = MemorySectionScanner(project_dir)
        self._determiner = ChangedItemsDeterminer(matrix_dir, state_file=_STATE_FILE)
        self._generator = QueriesGenerator(
            model, model_name, n_examples, parallel,
            prompt_template=MEMORY_PROMPT_TEMPLATE,
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
        sections = self._scanner.scan_and_filter(only)
        if not sections:
            print("No memory sections found (check that project.md or user.md exist in project-dir).")
            return

        changed, skipped = self._determiner.determine(sections, force)
        if not changed:
            print("Nothing to rebuild.")
            if skipped:
                print(f"Skipped: {', '.join(skipped)}")
            return

        gen_results = await self._generator.generate_for_items(changed)
        states, llm_calls = await self._evaluator.evaluate_all(gen_results)
        self._saver.save(sections, skipped, states)
        self._printer.print(changed, skipped, llm_calls)
