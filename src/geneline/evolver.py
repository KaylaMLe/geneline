"""Evolve the next generation from scored genomes.

Topology, prompts, and models stay fixed. Only temperature / top_p mutate.
Selection uses score-weighted parents with a minimum-score exclusion floor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep, ScoredGenome


@dataclass
class Evolver:
    population_size: int
    mutation_rate: float = 0.35
    mutation_scale: float = 0.15
    min_score_to_breed: float = 0.2
    elite_count: int = 1
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.rng = self.rng or random.Random()
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")

    def seed_population(self, seed: Genome) -> list[Genome]:
        """Wrap the user genome as generation 0, then mutate hypers to fill the pool."""
        population = [seed.clone(new_id=False)]
        while len(population) < self.population_size:
            population.append(self.mutate(seed.clone()))
        return population

    def next_generation(self, results: list[ScoredGenome]) -> list[Genome]:
        if not results:
            raise ValueError("cannot evolve from empty results")

        ranked = sorted(results, key=lambda item: item.score, reverse=True)
        breedable = [item for item in ranked if item.score >= self.min_score_to_breed]
        if not breedable:
            breedable = ranked[: max(1, len(ranked) // 2)]

        next_pop: list[Genome] = []
        for elite in ranked[: min(self.elite_count, len(ranked))]:
            next_pop.append(elite.genome.clone(new_id=True))

        while len(next_pop) < self.population_size:
            parent_a = self._weighted_pick(breedable)
            parent_b = self._weighted_pick(breedable)
            child = self.crossover(parent_a.genome, parent_b.genome)
            child = self.mutate(child)
            next_pop.append(child)

        return next_pop

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """Average hypers per step; copy prompt + model from A (fixed topology)."""
        if len(a.steps) != len(b.steps):
            raise ValueError(
                f"crossover requires equal step counts, got {len(a.steps)} and {len(b.steps)}"
            )
        steps: list[PipelineStep] = []
        for step_a, step_b in zip(a.steps, b.steps, strict=True):
            temperature = (step_a.hyperparameters.temperature + step_b.hyperparameters.temperature) / 2
            top_p = (step_a.hyperparameters.top_p + step_b.hyperparameters.top_p) / 2
            steps.append(
                PipelineStep(
                    prompt=step_a.prompt,
                    model=ModelSpec(
                        name=step_a.model.name,
                        cost_per_token=step_a.model.cost_per_token,
                    ),
                    hyperparameters=Hyperparameters(
                        temperature=temperature, top_p=top_p
                    ).clamped(),
                )
            )
        return Genome(steps=steps)

    def mutate(self, genome: Genome) -> Genome:
        """Nudge temperature / top_p only — never prompt or model."""
        child = genome.clone()
        for step in child.steps:
            if self.rng.random() < self.mutation_rate:
                step.hyperparameters.temperature += self.rng.uniform(
                    -self.mutation_scale, self.mutation_scale
                )
            if self.rng.random() < self.mutation_rate:
                step.hyperparameters.top_p += self.rng.uniform(
                    -self.mutation_scale, self.mutation_scale
                )
            step.hyperparameters = step.hyperparameters.clamped()
        return child

    def _weighted_pick(self, pool: list[ScoredGenome]) -> ScoredGenome:
        floor = min(item.score for item in pool)
        weights = [item.score - floor + 1e-6 for item in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]
