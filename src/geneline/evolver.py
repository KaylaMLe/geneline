"""Evolve the next generation from scored genomes.

Strategies (from the architecture sketch):
- exclude genomes below a minimum score
- select parents with score-weighted probabilities
- crossover hyperparameters / model choice
- mutate temperature and top_p
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from geneline.types import Genome, Hyperparameters, ModelSpec, PipelineStep, ScoredGenome


@dataclass
class Evolver:
    population_size: int
    models: list[ModelSpec]
    mutation_rate: float = 0.35
    mutation_scale: float = 0.15
    min_score_to_breed: float = 0.2
    elite_count: int = 1
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.rng = self.rng or random.Random()
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if not self.models:
            raise ValueError("at least one model must be configured")

    def seed_population(self, seed: Genome) -> list[Genome]:
        """Wrap the user genome as generation 0, then mutate to fill the pool."""
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
            if self.rng.random() < self.mutation_rate:
                child = self.mutate(child)
            next_pop.append(child)

        return next_pop

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """Per-step: keep immutable prompt from A; blend hypers; pick model by coin flip."""
        steps: list[PipelineStep] = []
        for step_a, step_b in zip(a.steps, b.steps, strict=False):
            # Weighted average of hyperparameters (score-agnostic blend; selection already weighted).
            temperature = (step_a.hyperparameters.temperature + step_b.hyperparameters.temperature) / 2
            top_p = (step_a.hyperparameters.top_p + step_b.hyperparameters.top_p) / 2
            model = step_a.model if self.rng.random() < 0.5 else step_b.model
            steps.append(
                PipelineStep(
                    prompt=step_a.prompt,  # immutable for now
                    model=ModelSpec(name=model.name, cost_per_token=model.cost_per_token),
                    hyperparameters=Hyperparameters(temperature=temperature, top_p=top_p).clamped(),
                )
            )
        # If genomes differ in length, keep remaining steps from the longer parent.
        longer = a if len(a.steps) >= len(b.steps) else b
        for step in longer.steps[len(steps) :]:
            steps.append(
                PipelineStep(
                    prompt=step.prompt,
                    model=ModelSpec(name=step.model.name, cost_per_token=step.model.cost_per_token),
                    hyperparameters=step.hyperparameters.clamped(),
                )
            )
        return Genome(steps=steps)

    def mutate(self, genome: Genome) -> Genome:
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
            if self.rng.random() < self.mutation_rate * 0.5:
                pick = self.rng.choice(self.models)
                step.model = ModelSpec(name=pick.name, cost_per_token=pick.cost_per_token)
        return child

    def _weighted_pick(self, pool: list[ScoredGenome]) -> ScoredGenome:
        # Shift scores so the worst breedable parent still has positive weight.
        floor = min(item.score for item in pool)
        weights = [item.score - floor + 1e-6 for item in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]
