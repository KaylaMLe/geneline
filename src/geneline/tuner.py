"""Main genetic tuning loop: evaluate → score → evolve → repeat."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geneline import progress
from geneline.evolver import Evolver
from geneline.quality import QualityJudge, build_quality_judge
from geneline.runners import Runner, RunnerName, build_runner, run_pipeline
from geneline.scorer import Scorer
from geneline.utils.io import read_json, read_text, write_json
from geneline.utils.types import Genome, ModelSpec, ScoredGenome


@dataclass
class TunerConfig:
    population_size: int = 6
    max_generations: int = 5
    goal_score: float = 0.92
    min_score_to_breed: float = 0.2
    mutation_rate: float = 0.35
    elite_count: int = 1
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    cost_weight: float = 0.25
    latency_ref_ms: float = 800.0
    cost_ref: float = 0.002
    seed: int = 42
    models: list[ModelSpec] | None = None
    runner: RunnerName = "openrouter"
    max_parallel_genomes: int = 8
    patience: int = 5
    quality: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunerConfig:
        weights = data.get("weights", {})
        models_raw = data.get("models") or [
            {
                "name": "openai/gpt-5.6-luna",
                "cost_per_input_token": 0.0000001,
                "cost_per_output_token": 0.0000006,
            },
        ]
        runner = str(data.get("runner", "openrouter")).lower()
        if runner not in ("mock", "openrouter"):
            raise ValueError(f"unsupported runner: {runner!r}")
        quality = data.get("quality")
        if quality is not None and not isinstance(quality, dict):
            raise ValueError("config.quality must be an object")
        return cls(
            population_size=int(data.get("population_size", 6)),
            max_generations=int(data.get("max_generations", 5)),
            goal_score=float(data.get("goal_score", 0.92)),
            min_score_to_breed=float(data.get("min_score_to_breed", 0.2)),
            mutation_rate=float(data.get("mutation_rate", 0.35)),
            elite_count=int(data.get("elite_count", 1)),
            quality_weight=float(weights.get("quality", data.get("quality_weight", 1.0))),
            latency_weight=float(weights.get("latency", data.get("latency_weight", 0.25))),
            cost_weight=float(weights.get("cost", data.get("cost_weight", 0.25))),
            latency_ref_ms=float(data.get("latency_ref_ms", 800.0)),
            cost_ref=float(data.get("cost_ref", 0.002)),
            seed=int(data.get("seed", 42)),
            models=[ModelSpec.from_dict(item) for item in models_raw],
            runner=runner,  # type: ignore[arg-type]
            max_parallel_genomes=max(1, int(data.get("max_parallel_genomes", 8))),
            patience=max(1, int(data.get("patience", 5))),
            quality=quality,
        )


@dataclass
class TunerResult:
    best: ScoredGenome
    generations_run: int
    stopped_reason: str
    history: list[dict[str, Any]]


class Tuner:
    def __init__(
        self,
        config: TunerConfig,
        runner: Runner | None = None,
        quality_judge: QualityJudge | None = None,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.runner: Runner = (
            runner if runner is not None else build_runner(config.runner, seed=config.seed)
        )
        self.quality_judge = (
            quality_judge
            if quality_judge is not None
            else build_quality_judge(config.quality)
        )
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
            min_score_to_breed=config.min_score_to_breed,
            elite_count=config.elite_count,
            rng=self.rng,
        )

    def _evaluate_one(
        self,
        index: int,
        total: int,
        genome: Genome,
        message: str,
    ) -> tuple[int, ScoredGenome]:
        label = f"[{index}/{total} {genome.id}]"
        progress.log(
            f"  genome {index}/{total} id={genome.id} "
            f"({len(genome.steps)} steps) — starting"
        )
        timer = progress.Timer()
        try:
            response = run_pipeline(
                self.runner,
                genome,
                message,
                label=label,
                quality_judge=self.quality_judge,
            )
        except Exception as exc:
            raise RuntimeError(
                f"genome evaluation failed for id={genome.id} (index {index}/{total}): {exc}"
            ) from exc
        total_score, components = self.scorer.score(response)
        scored = ScoredGenome(
            genome=genome,
            score=round(total_score, 4),
            response=response,
            components=components,
        )
        progress.log(
            f"  genome {index}/{total} id={genome.id} done in "
            f"{timer.elapsed_s():.1f}s  score={scored.score}  "
            f"quality={response.quality}  "
            f"latency_ms={response.latency_ms}  cost={response.cost}"
        )
        return index, scored

    def evaluate_generation(
        self,
        population: list[Genome],
        message: str,
        *,
        generation: int,
        max_generations: int,
    ) -> list[ScoredGenome]:
        total = len(population)
        workers = min(self.config.max_parallel_genomes, total)
        progress.log(
            f"generation {generation + 1}/{max_generations}: "
            f"evaluating {total} genome(s) (parallelism={workers})"
        )

        results_by_index: dict[int, ScoredGenome] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._evaluate_one, index, total, genome, message): index
                for index, genome in enumerate(population, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                index, scored = future.result()
                results_by_index[index] = scored
                completed += 1
                progress.log(
                    f"  completed {completed}/{total} "
                    f"(latest id={scored.genome.id} score={scored.score})"
                )

        return [results_by_index[i] for i in range(1, total + 1)]

    def run(self, seed_genome: Genome, message: str) -> TunerResult:
        population = self.evolver.seed_population(seed_genome)
        best: ScoredGenome | None = None
        history: list[dict[str, Any]] = []
        stopped_reason = "max_generations_reached"
        gens_without_improvement = 0
        steps_per_genome = max(1, len(seed_genome.steps))
        max_api_calls = (
            self.config.population_size
            * self.config.max_generations
            * steps_per_genome
        )
        progress.log(
            f"tuner start  runner={self.config.runner}  "
            f"population={self.config.population_size}  "
            f"max_generations={self.config.max_generations}  "
            f"max_parallel_genomes={self.config.max_parallel_genomes}  "
            f"goal_score={self.config.goal_score}  "
            f"patience={self.config.patience}  "
            f"steps/genome={steps_per_genome}  "
            f"up to ~{max_api_calls} model calls"
        )
        run_timer = progress.Timer()

        for generation in range(self.config.max_generations):
            results = self.evaluate_generation(
                population,
                message,
                generation=generation,
                max_generations=self.config.max_generations,
            )
            results.sort(key=lambda item: item.score, reverse=True)
            generation_best = results[0]
            mean_score = round(sum(item.score for item in results) / len(results), 4)
            if best is None or generation_best.score > best.score:
                best = generation_best
                gens_without_improvement = 0
                progress.log(
                    f"generation {generation + 1}: new global best "
                    f"score={best.score} id={best.genome.id}  mean={mean_score}"
                )
            else:
                gens_without_improvement += 1
                progress.log(
                    f"generation {generation + 1}: "
                    f"best_this_gen={generation_best.score}  "
                    f"global_best={best.score}  mean={mean_score}"
                )

            history.append(
                {
                    "generation": generation,
                    "best_score": generation_best.score,
                    "best_genome_id": generation_best.genome.id,
                    "mean_score": mean_score,
                    "results": [item.to_dict() for item in results],
                }
            )

            if best.score >= self.config.goal_score:
                stopped_reason = "goal_score_achieved"
                progress.log(
                    f"goal score {self.config.goal_score} reached "
                    f"(best={best.score}); stopping"
                )
                break

            if gens_without_improvement >= self.config.patience:
                stopped_reason = "no_improvement"
                progress.log(
                    f"no global-best improvement for {self.config.patience} "
                    f"generation(s); stopping"
                )
                break

            if generation < self.config.max_generations - 1:
                progress.log(
                    f"generation {generation + 1}: evolving next population..."
                )
                population = self.evolver.next_generation(results)

        assert best is not None
        progress.log(
            f"tuner done in {run_timer.elapsed_s():.1f}s  "
            f"reason={stopped_reason}  generations={len(history)}  "
            f"best_score={best.score}"
        )
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
    *,
    runner: RunnerName | None = None,
) -> TunerResult:
    """Read JSON/text inputs, run the tuner, write JSON outputs under out_dir."""
    out = Path(out_dir)
    seed = Genome.from_dict(read_json(genome_path))
    message = read_text(message_path)
    config = TunerConfig.from_dict(read_json(config_path))
    if runner is not None:
        config.runner = runner

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
            "runner": config.runner,
        },
    )
    write_json(out / "history.json", result.history)

    for entry in result.history:
        gen = entry["generation"]
        write_json(out / f"generation_{gen}_results.json", entry["results"])

    return result
