"""Tests for parallel generation evaluation."""

from __future__ import annotations

import unittest

from geneline.runners import MockRunner
from geneline.tuner import Tuner, TunerConfig
from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep


def _genome(genome_id: str, temp: float = 0.4) -> Genome:
    return Genome(
        id=genome_id,
        steps=[
            PipelineStep(
                prompt="Summarize this text: {{input}}",
                model=ModelSpec(name="mock-balanced", cost_per_token=0.00001),
                hyperparameters=Hyperparameters(temperature=temp, top_p=0.9),
            ),
            PipelineStep(
                prompt="Extract the main claim from this text: {{input}}",
                model=ModelSpec(name="mock-fast", cost_per_token=0.000002),
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
        )
        tuner = Tuner(config, runner=MockRunner(seed=7))
        population = [_genome(f"g-{i}", temp=0.3 + 0.05 * i) for i in range(4)]
        results = tuner.evaluate_generation(
            population,
            "Genetic algorithms evolve candidates.",
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
            min_score_to_breed=0.0,
        )
        tuner = Tuner(config, runner=MockRunner(seed=3))
        result = tuner.run(_genome("seed"), "Genetic algorithms evolve candidates.")
        self.assertGreaterEqual(result.generations_run, 1)
        self.assertIsNotNone(result.best)


if __name__ == "__main__":
    unittest.main()
