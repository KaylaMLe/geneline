"""Focused tests for templating, chaining, and deterministic mocks."""

from __future__ import annotations

import unittest
from pathlib import Path

from geneline.runners import MockRunner, ScriptedRunner, StepResult, run_pipeline
from geneline.utils.prompt import render_prompt
from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep

EXAMPLE_MESSAGE = (
    Path(__file__).resolve().parents[1] / "examples" / "message.txt"
).read_text(encoding="utf-8").strip()


def _step(prompt: str, model: str = "mock-fast") -> PipelineStep:
    return PipelineStep(
        prompt=prompt,
        model=ModelSpec(name=model, cost_per_token=0.000002),
        hyperparameters=Hyperparameters(temperature=0.3, top_p=0.9),
    )


class PromptTests(unittest.TestCase):
    def test_renders_single_placeholder(self) -> None:
        self.assertEqual(
            render_prompt("Summarize this text: {{input}}", "hello"),
            "Summarize this text: hello",
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
                _step("Summarize this text: {{input}}"),
                _step("Extract the main claim from this text: {{input}}"),
            ],
        )
        # Canned step outputs stand in for model replies on the two-sentence example input.
        runner = ScriptedRunner(
            queue=[
                StepResult(
                    message=(
                        "GAs evolve candidates by selection, crossover, and mutation; "
                        "fitness also weighs latency and cost."
                    ),
                    latency_ms=10,
                    cost=0.001,
                    total_tokens=20,
                ),
                StepResult(
                    message=(
                        "Genetic algorithms evolve candidates until a fitness goal "
                        "that balances quality with resource cost."
                    ),
                    latency_ms=12,
                    cost=0.002,
                    total_tokens=22,
                ),
            ]
        )
        response = run_pipeline(runner, genome, EXAMPLE_MESSAGE)
        self.assertIn("fitness", response.message.lower())
        self.assertEqual(response.latency_ms, 22.0)
        self.assertEqual(response.cost, 0.003)
        self.assertEqual(response.total_tokens, 42)
        self.assertEqual(runner.queue, [])

    def test_mock_runner_is_deterministic_for_same_request(self) -> None:
        runner = MockRunner(seed=7)
        model = ModelSpec(name="mock-balanced", cost_per_token=0.00001)
        hp = Hyperparameters(temperature=0.4, top_p=0.9)
        prompt = f"Summarize this text: {EXAMPLE_MESSAGE}"
        a = runner.run_step(model=model, hyperparameters=hp, rendered_prompt=prompt)
        b = runner.run_step(model=model, hyperparameters=hp, rendered_prompt=prompt)
        self.assertEqual(a, b)

    def test_full_pipeline_reproducible_with_mock(self) -> None:
        genome = Genome(
            id="seed",
            steps=[
                _step("Summarize this text: {{input}}", "mock-balanced"),
                _step("Extract the main claim from this text: {{input}}", "mock-fast"),
            ],
        )
        runner = MockRunner(seed=42)
        first = run_pipeline(runner, genome, EXAMPLE_MESSAGE)
        second = run_pipeline(runner, genome, EXAMPLE_MESSAGE)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
