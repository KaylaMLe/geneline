"""Tests for parallel generation evaluation."""

from __future__ import annotations

import unittest
from pathlib import Path

from geneline.runners import MockRunner
from geneline.tuner import Tuner, TunerConfig
from geneline.utils.io import read_json
from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MESSAGE = (ROOT / "examples" / "message.txt").read_text(encoding="utf-8").strip()
ANSWER_PATH = str(ROOT / "examples" / "quality.answer.json")
SEED_PROMPT = Genome.from_dict(read_json(ROOT / "examples" / "genome.mock.json")).steps[0].prompt
TAX_PROMPT = Genome.from_dict(read_json(ROOT / "examples" / "genome.mock.json")).steps[1].prompt
MODELS = [
    ModelSpec(
        name="mock-fast",
        cost_per_input_token=0.000002,
        cost_per_output_token=0.000002,
    ),
    ModelSpec(
        name="mock-balanced",
        cost_per_input_token=0.00001,
        cost_per_output_token=0.00001,
    ),
]


def _genome(genome_id: str, temp: float = 0.4) -> Genome:
    return Genome(
        id=genome_id,
        steps=[
            PipelineStep(
                prompt=SEED_PROMPT,
                model=ModelSpec(
                    name="mock-balanced",
                    cost_per_input_token=0.00001,
                    cost_per_output_token=0.00001,
                ),
                hyperparameters=Hyperparameters(temperature=temp, top_p=0.9),
            ),
            PipelineStep(
                prompt=TAX_PROMPT,
                model=ModelSpec(
                    name="mock-fast",
                    cost_per_input_token=0.000002,
                    cost_per_output_token=0.000002,
                ),
                hyperparameters=Hyperparameters(temperature=0.3, top_p=0.9),
            ),
        ],
    )


class ParallelEvalTests(unittest.TestCase):
    def test_evaluate_generation_runs_in_parallel_and_preserves_count(self) -> None:
        config = TunerConfig(
            population_size=4,
            max_generations=1,
            goal_score=0.99,
            max_parallel_genomes=4,
            runner="mock",
            seed=7,
            models=MODELS,
            quality={"judge": "answer", "answer_path": ANSWER_PATH},
        )
        tuner = Tuner(config, runner=MockRunner(seed=7))
        population = [_genome(f"g-{i}", temp=0.3 + 0.05 * i) for i in range(4)]
        results = tuner.evaluate_generation(
            population,
            EXAMPLE_MESSAGE,
            generation=0,
            max_generations=1,
        )
        self.assertEqual(len(results), 4)
        self.assertEqual([item.genome.id for item in results], [f"g-{i}" for i in range(4)])
        self.assertTrue(all(isinstance(item.score, float) for item in results))

    def test_parallel_run_reaches_goal_or_completes(self) -> None:
        config = TunerConfig(
            population_size=4,
            max_generations=2,
            goal_score=0.2,
            max_parallel_genomes=3,
            runner="mock",
            seed=3,
            models=MODELS,
            min_score_to_breed=0.0,
            quality={"judge": "answer", "answer_path": ANSWER_PATH},
        )
        tuner = Tuner(config, runner=MockRunner(seed=3))
        result = tuner.run(_genome("seed"), EXAMPLE_MESSAGE)
        self.assertGreaterEqual(result.generations_run, 1)
        self.assertIsNotNone(result.best)

    def test_stops_after_configured_patience_without_improvement(self) -> None:
        fixed_model = [MODELS[0]]
        config = TunerConfig(
            population_size=3,
            max_generations=10,
            goal_score=2.0,
            max_parallel_genomes=3,
            runner="mock",
            seed=9,
            models=fixed_model,
            patience=1,
            quality={"judge": "answer", "answer_path": ANSWER_PATH},
        )
        seed = _genome("seed")
        for step in seed.steps:
            step.model = fixed_model[0]

        result = Tuner(config, runner=MockRunner(seed=9)).run(seed, EXAMPLE_MESSAGE)

        self.assertEqual(result.stopped_reason, "no_improvement")
        self.assertEqual(result.generations_run, 2)

    def test_model_mutation_can_improve_mock_step_quality(self) -> None:
        models = [
            MODELS[0],
            ModelSpec(
                name="mock-quality",
                cost_per_input_token=0.00005,
                cost_per_output_token=0.00005,
            ),
        ]
        config = TunerConfig(
            population_size=6,
            max_generations=1,
            goal_score=2.0,
            mutation_rate=1.0,
            elite_count=0,
            quality_weight=1.0,
            latency_weight=0.0,
            cost_weight=0.0,
            runner="mock",
            seed=42,
            models=models,
            quality={"judge": "answer", "answer_path": ANSWER_PATH},
        )
        seed = _genome("seed")
        for step in seed.steps:
            step.model = models[0]

        result = Tuner(config, runner=MockRunner(seed=42)).run(seed, EXAMPLE_MESSAGE)

        self.assertGreater(result.best.response.quality, 0.641)
        self.assertIn(
            "mock-quality",
            [step.model.name for step in result.best.genome.steps],
        )


if __name__ == "__main__":
    unittest.main()
