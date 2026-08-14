"""Evolve the next generation from scored genomes.

Phase ``models``: topology, prompts, and hyperparameters stay fixed; only
per-step model choice mutates (from a config allow-list).

Phase ``hypers``: models and prompts stay fixed; temperature / top_p jitter.

Selection is score-weighted; crossover inherits genes from the fitter parent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from geneline.utils.types import Genome, Hyperparameters, ModelSpec, PipelineStep, ScoredGenome

GeneMode = Literal["models", "hypers"]


@dataclass
class Evolver:
    population_size: int
    models: list[ModelSpec] = field(default_factory=list)
    mutation_rate: float = 0.35
    min_score_to_breed: float = 0.2
    elite_count: int = 1
    gene: GeneMode = "models"
    temperature_sigma: float = 0.15
    top_p_sigma: float = 0.08
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.rng = self.rng or random.Random()
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.gene not in ("models", "hypers"):
            raise ValueError(f"unsupported gene mode: {self.gene!r}")
        if self.gene == "models" and not self.models:
            raise ValueError("evolver requires a non-empty models allow-list")

    def seed_population(self, seed: Genome) -> list[Genome]:
        """Keep the user genome, then fill the pool by mutating active genes."""
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
            if parent_a.score >= parent_b.score:
                fitter, weaker = parent_a, parent_b
            else:
                fitter, weaker = parent_b, parent_a
            child = self.crossover(fitter.genome, weaker.genome)
            child = self.mutate(child)
            next_pop.append(child)

        return next_pop

    def crossover(self, fitter: Genome, weaker: Genome) -> Genome:
        """Inherit active genes from the fitter parent; copy the rest."""
        del weaker  # fitness pressure is selection + fitter inheritance
        if not fitter.steps:
            raise ValueError("crossover requires at least one step")
        steps: list[PipelineStep] = []
        for step in fitter.steps:
            steps.append(
                PipelineStep(
                    prompt=step.prompt,
                    model=ModelSpec(
                        name=step.model.name,
                        cost_per_input_token=step.model.cost_per_input_token,
                        cost_per_output_token=step.model.cost_per_output_token,
                    ),
                    hyperparameters=step.hyperparameters.clamped(),
                )
            )
        return Genome(steps=steps)

    def mutate(self, genome: Genome) -> Genome:
        """Mutate only the active gene mode; leave other fields untouched."""
        child = genome.clone()
        for step in child.steps:
            if self.rng.random() >= self.mutation_rate:
                continue
            if self.gene == "models":
                pick = self.rng.choice(self.models)
                step.model = ModelSpec(
                    name=pick.name,
                    cost_per_input_token=pick.cost_per_input_token,
                    cost_per_output_token=pick.cost_per_output_token,
                )
            else:
                hp = step.hyperparameters
                step.hyperparameters = Hyperparameters(
                    temperature=hp.temperature
                    + self.rng.gauss(0.0, self.temperature_sigma),
                    top_p=hp.top_p + self.rng.gauss(0.0, self.top_p_sigma),
                ).clamped()
        return child

    def _weighted_pick(self, pool: list[ScoredGenome]) -> ScoredGenome:
        floor = min(item.score for item in pool)
        weights = [item.score - floor + 1e-6 for item in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]
