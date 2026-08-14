"""Focused tests for templating, chaining, and deterministic mocks."""

from __future__ import annotations

import unittest
from pathlib import Path

from geneline.quality.answer import AnswerJudge
from geneline.runners import MockRunner, ScriptedRunner, StepResult, run_pipeline
from geneline.utils.io import read_json
from geneline.utils.prompt import render_prompt
from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MESSAGE = (ROOT / "examples" / "message.txt").read_text(encoding="utf-8").strip()
ANSWER_PATH = ROOT / "examples" / "quality.answer.json"
EXAMPLE_GENOME = Genome.from_dict(read_json(ROOT / "examples" / "genome.mock.json"))


def _step(prompt: str, model: str = "mock-fast") -> PipelineStep:
    return PipelineStep(
        prompt=prompt,
        model=ModelSpec(name=model, cost_per_token=0.000002),
        hyperparameters=Hyperparameters(temperature=0.3, top_p=0.9),
    )


class PromptTests(unittest.TestCase):
    def test_renders_single_placeholder(self) -> None:
        self.assertEqual(
            render_prompt("Average the shoes: {{input}}", "round please"),
            "Average the shoes: round please",
        )

    def test_rejects_missing_or_extra_placeholder(self) -> None:
        with self.assertRaises(ValueError):
            render_prompt("no placeholder", "x")
        with self.assertRaises(ValueError):
            render_prompt("{{input}} and {{input}}", "x")


class PipelineTests(unittest.TestCase):
    def test_scripted_runner_chains_prior_output_into_next_prompt(self) -> None:
        genome = Genome(
            id="test",
            steps=[
                _step("Average shoes: {{input}}"),
                _step("Add 10% sales tax: {{input}}"),
            ],
        )
        runner = ScriptedRunner(
            queue=[
                StepResult(message="70.00", latency_ms=10, cost=0.001, total_tokens=20),
                StepResult(message="77.00", latency_ms=12, cost=0.002, total_tokens=22),
            ]
        )
        judge = AnswerJudge.from_path(ANSWER_PATH)
        response = run_pipeline(
            runner, genome, EXAMPLE_MESSAGE, quality_judge=judge
        )
        self.assertEqual(response.message, "77.00")
        self.assertEqual(response.quality, 1.0)
        self.assertEqual(response.step_messages, ["70.00", "77.00"])
        self.assertEqual(response.latency_ms, 22.0)
        self.assertEqual(response.cost, 0.003)
        self.assertEqual(response.total_tokens, 42)
        self.assertEqual(runner.queue, [])

    def test_mock_runner_is_deterministic_for_same_request(self) -> None:
        runner = MockRunner(seed=7)
        model = ModelSpec(name="mock-balanced", cost_per_token=0.00001)
        hp = Hyperparameters(temperature=0.4, top_p=0.9)
        prompt = EXAMPLE_GENOME.steps[0].prompt.replace("{{input}}", EXAMPLE_MESSAGE)
        a = runner.run_step(model=model, hyperparameters=hp, rendered_prompt=prompt)
        b = runner.run_step(model=model, hyperparameters=hp, rendered_prompt=prompt)
        self.assertEqual(a, b)

    def test_full_pipeline_reproducible_with_mock(self) -> None:
        runner = MockRunner(seed=42)
        judge = AnswerJudge.from_path(ANSWER_PATH)
        first = run_pipeline(
            runner, EXAMPLE_GENOME, EXAMPLE_MESSAGE, quality_judge=judge
        )
        second = run_pipeline(
            MockRunner(seed=42),
            EXAMPLE_GENOME,
            EXAMPLE_MESSAGE,
            quality_judge=judge,
        )
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
