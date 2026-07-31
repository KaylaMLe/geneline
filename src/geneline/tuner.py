"""Main genetic tuning loop: evaluate → score → evolve → repeat."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geneline.evolver import Evolver
from geneline.io import read_json, read_text, write_json
from geneline.runner import MockRunner
from geneline.scorer import Scorer
from geneline.types import Genome, ModelSpec, ScoredGenome


@dataclass
class TunerConfig:
    population_size: int = 6
    max_generations: int = 5
    goal_score: float = 0.92
    min_score_to_breed: float = 0.2
    mutation_rate: float = 0.35
    mutation_scale: float = 0.15
    elite_count: int = 1
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    cost_weight: float = 0.25
    latency_ref_ms: float = 800.0
    cost_ref: float = 0.002
    seed: int = 42
    models: list[ModelSpec] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunerConfig:
        weights = data.get("weights", {})
        models_raw = data.get("models") or [
            {"name": "mock-fast", "cost_per_token": 0.000002},
            {"name": "mock-balanced", "cost_per_token": 0.00001},
            {"name": "mock-quality", "cost_per_token": 0.00005},
        ]
        return cls(
            population_size=int(data.get("population_size", 6)),
            max_generations=int(data.get("max_generations", 5)),
            goal_score=float(data.get("goal_score", 0.92)),
            min_score_to_breed=float(data.get("min_score_to_breed", 0.2)),
            mutation_rate=float(data.get("mutation_rate", 0.35)),
            mutation_scale=float(data.get("mutation_scale", 0.15)),
            elite_count=int(data.get("elite_count", 1)),
            quality_weight=float(weights.get("quality", data.get("quality_weight", 1.0))),
            latency_weight=float(weights.get("latency", data.get("latency_weight", 0.25))),
            cost_weight=float(weights.get("cost", data.get("cost_weight", 0.25))),
            latency_ref_ms=float(data.get("latency_ref_ms", 800.0)),
            cost_ref=float(data.get("cost_ref", 0.002)),
            seed=int(data.get("seed", 42)),
            models=[ModelSpec.from_dict(item) for item in models_raw],
        )


@dataclass
class TunerResult:
    best: ScoredGenome
    generations_run: int
    stopped_reason: str
    history: list[dict[str, Any]]


class Tuner:
    def __init__(self, config: TunerConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.runner = MockRunner(rng=random.Random(config.seed + 1))
        self.scorer = Scorer(
            quality_weight=config.quality_weight,
            latency_weight=config.latency_weight,
            cost_weight=config.cost_weight,
            latency_ref_ms=config.latency_ref_ms,
            cost_ref=config.cost_ref,
        )
        self.evolver = Evolver(
            population_size=config.population_size,
            models=config.models or [],
            mutation_rate=config.mutation_rate,
            mutation_scale=config.mutation_scale,
            min_score_to_breed=config.min_score_to_breed,
            elite_count=config.elite_count,
            rng=self.rng,
        )

    def evaluate_generation(
        self, population: list[Genome], message: str
    ) -> list[ScoredGenome]:
        scored: list[ScoredGenome] = []
        for genome in population:
            response = self.runner.run(genome, message)
            total, components = self.scorer.score(response)
            scored.append(
                ScoredGenome(
                    genome=genome,
                    score=round(total, 4),
                    response=response,
                    components=components,
                )
            )
        return scored

    def run(self, seed_genome: Genome, message: str) -> TunerResult:
        population = self.evolver.seed_population(seed_genome)
        best: ScoredGenome | None = None
        history: list[dict[str, Any]] = []
        stopped_reason = "max_generations_reached"

        for generation in range(self.config.max_generations):
            results = self.evaluate_generation(population, message)
            results.sort(key=lambda item: item.score, reverse=True)
            generation_best = results[0]
            if best is None or generation_best.score > best.score:
                best = generation_best

            history.append(
                {
                    "generation": generation,
                    "best_score": generation_best.score,
                    "best_genome_id": generation_best.genome.id,
                    "mean_score": round(
                        sum(item.score for item in results) / len(results), 4
                    ),
                    "results": [item.to_dict() for item in results],
                }
            )

            if best.score >= self.config.goal_score:
                stopped_reason = "goal_score_achieved"
                break

            if generation < self.config.max_generations - 1:
                population = self.evolver.next_generation(results)

        assert best is not None
        return TunerResult(
            best=best,
            generations_run=len(history),
            stopped_reason=stopped_reason,
            history=history,
        )


def run_from_paths(
    genome_path: str | Path,
    message_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
) -> TunerResult:
    """Read JSON/text inputs, run the tuner, write JSON outputs under out_dir."""
    out = Path(out_dir)
    seed = Genome.from_dict(read_json(genome_path))
    message = read_text(message_path)
    config = TunerConfig.from_dict(read_json(config_path))

    result = Tuner(config).run(seed, message)

    write_json(out / "best_genome.json", result.best.genome.to_dict())
    write_json(
        out / "best_result.json",
        {
            "score": result.best.score,
            "components": result.best.components,
            "response": result.best.response.to_dict(),
            "genome": result.best.genome.to_dict(),
            "stopped_reason": result.stopped_reason,
            "generations_run": result.generations_run,
        },
    )
    write_json(out / "history.json", result.history)

    # Also emit per-generation result files for easy inspection.
    for entry in result.history:
        gen = entry["generation"]
        write_json(out / f"generation_{gen}_results.json", entry["results"])

    return result
