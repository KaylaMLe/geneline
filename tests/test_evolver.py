"""Tests for evolver mutation behavior."""

from __future__ import annotations

import unittest

from geneline.evolver import Evolver
from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep, ScoredGenome
from geneline.utils.types import Response


def _genome(temp: float = 0.4, top_p: float = 0.9) -> Genome:
    return Genome(
        id="g",
        steps=[
            PipelineStep(
                prompt="step1: {{input}}",
                model=ModelSpec(name="mock-fast", cost_per_token=0.000002),
                hyperparameters=Hyperparameters(temperature=temp, top_p=top_p),
            ),
            PipelineStep(
                prompt="step2: {{input}}",
                model=ModelSpec(name="mock-fast", cost_per_token=0.000002),
                hyperparameters=Hyperparameters(temperature=temp, top_p=top_p),
            ),
        ],
    )


class EvolverMutationTests(unittest.TestCase):
    def test_next_generation_always_applies_per_gene_mutation_when_rate_is_one(self) -> None:
        evolver = Evolver(
            population_size=4,
            mutation_rate=1.0,
            mutation_scale=0.2,
            elite_count=0,
            rng=__import__("random").Random(0),
        )
        parent = _genome(temp=0.5, top_p=0.5)
        scored = [
            ScoredGenome(
                genome=parent.clone(),
                score=1.0,
                response=Response(
                    message="77", latency_ms=1, cost=0.0, total_tokens=1, quality=1.0
                ),
                components={},
            )
            for _ in range(2)
        ]
        next_pop = evolver.next_generation(scored)
        self.assertEqual(len(next_pop), 4)
        # With rate=1 every gene mutates; averaged parents share 0.5/0.5 so children move.
        moved = 0
        for child in next_pop:
            for step in child.steps:
                if step.hyperparameters.temperature != 0.5 or step.hyperparameters.top_p != 0.5:
                    moved += 1
        self.assertGreater(moved, 0)


if __name__ == "__main__":
    unittest.main()
