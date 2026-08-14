"""Tests for model-gene evolution."""

from __future__ import annotations

import random
import unittest

from geneline.evolver import Evolver
from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep


MODELS = [
    ModelSpec(
        name="mock-fast",
        cost_per_input_token=0.000002,
        cost_per_output_token=0.000002,
    ),
    ModelSpec(
        name="mock-quality",
        cost_per_input_token=0.00005,
        cost_per_output_token=0.00005,
    ),
]


def _genome(model: ModelSpec = MODELS[0], temp: float = 0.4, top_p: float = 0.9) -> Genome:
    return Genome(
        id="g",
        steps=[
            PipelineStep(
                prompt="step1: {{input}}",
                model=model,
                hyperparameters=Hyperparameters(temperature=temp, top_p=top_p),
            ),
            PipelineStep(
                prompt="step2: {{input}}",
                model=model,
                hyperparameters=Hyperparameters(temperature=temp, top_p=top_p),
            ),
        ],
    )


class EvolverModelGeneTests(unittest.TestCase):
    def test_mutation_changes_models_but_preserves_hypers_and_prompts(self) -> None:
        evolver = Evolver(
            population_size=4,
            models=MODELS,
            mutation_rate=1.0,
            elite_count=0,
            rng=random.Random(0),
        )
        parent = _genome(temp=0.5, top_p=0.5)
        child = evolver.mutate(parent)

        for original, mutated in zip(parent.steps, child.steps, strict=True):
            self.assertEqual(mutated.prompt, original.prompt)
            self.assertEqual(mutated.hyperparameters, original.hyperparameters)
            self.assertIn(mutated.model.name, {model.name for model in MODELS})

    def test_crossover_inherits_fitter_models(self) -> None:
        evolver = Evolver(population_size=2, models=MODELS, rng=random.Random(0))
        fitter = _genome(MODELS[1], temp=0.3, top_p=0.8)
        weaker = _genome(MODELS[0], temp=0.7, top_p=0.5)
        child = evolver.crossover(fitter, weaker)

        self.assertEqual(
            [step.model.name for step in child.steps],
            ["mock-quality", "mock-quality"],
        )
        self.assertEqual(
            [step.hyperparameters for step in child.steps],
            [step.hyperparameters for step in fitter.steps],
        )


if __name__ == "__main__":
    unittest.main()
