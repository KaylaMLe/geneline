"""Execute pipeline genomes via a pluggable Runner.

MockRunner is hash-deterministic per rendered request so unit/E2E tests stay stable.
ScriptedRunner serves canned step results for focused chaining tests.
Swap in an OpenRouter (or other) Runner later behind the same protocol.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Protocol

from geneline.prompt import render_prompt
from geneline.types import Genome, Hyperparameters, ModelSpec, Response


# Ideal final claim used only by the mock quality metric.
REFERENCE_CLAIM = (
    "Genetic algorithms evolve candidates via selection, crossover, and mutation "
    "until a fitness goal is reached."
)


@dataclass(frozen=True)
class StepResult:
    """Raw metrics from a single model call (no quality — scored on final output)."""

    message: str
    latency_ms: float
    cost: float
    total_tokens: int


class Runner(Protocol):
    def run_step(
        self,
        *,
        model: ModelSpec,
        hyperparameters: Hyperparameters,
        rendered_prompt: str,
    ) -> StepResult:
        """Execute one data-processing step against an already-rendered prompt."""


@dataclass
class MockRunner:
    """Deterministic mock: same (model, hypers, rendered_prompt) ⇒ same StepResult."""

    seed: int = 0

    def run_step(
        self,
        *,
        model: ModelSpec,
        hyperparameters: Hyperparameters,
        rendered_prompt: str,
    ) -> StepResult:
        hp = hyperparameters.clamped()
        digest = hashlib.sha256(
            f"{self.seed}|{model.name}|{hp.temperature:.6f}|{hp.top_p:.6f}|{rendered_prompt}".encode()
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        prompt_tokens = max(1, len(rendered_prompt.split()))
        completion_tokens = self._completion_tokens(model, hp.temperature, rng)
        total_tokens = prompt_tokens + completion_tokens
        cost = total_tokens * model.cost_per_token

        base_latency = {
            "mock-fast": 180.0,
            "mock-balanced": 420.0,
            "mock-quality": 780.0,
        }.get(model.name, 500.0)
        # Derived from the request hash — stable, not wall-clock jitter.
        jitter = (rng.random() * 120.0) - 40.0 + (hp.temperature * 35.0)
        latency_ms = max(50.0, base_latency + jitter + (completion_tokens * 1.5))

        text = self._generate_text(model, hp.temperature, hp.top_p, rendered_prompt, rng)

        return StepResult(
            message=text,
            latency_ms=round(latency_ms, 2),
            cost=round(cost, 8),
            total_tokens=total_tokens,
        )

    def _completion_tokens(
        self, model: ModelSpec, temperature: float, rng: random.Random
    ) -> int:
        base = {"mock-fast": 28, "mock-balanced": 40, "mock-quality": 55}.get(model.name, 36)
        return max(12, int(base + temperature * 12 + rng.randint(-4, 6)))

    def _generate_text(
        self,
        model: ModelSpec,
        temperature: float,
        top_p: float,
        rendered_prompt: str,
        rng: random.Random,
    ) -> str:
        fidelity = _fidelity(model, temperature, top_p)
        # Prefer extracting/summarizing toward the reference claim for this toy task.
        words = REFERENCE_CLAIM.split()
        keep = max(4, int(len(words) * fidelity))
        kept = words[:keep]

        if fidelity < 0.75:
            noise = ["roughly", "perhaps", "sort of", "maybe", "kinda"]
            kept.insert(rng.randint(0, len(kept)), rng.choice(noise))
        if fidelity < 0.55:
            kept.append("etc.")

        # Slightly bias wording using a stable crumb from the rendered prompt.
        if "Extract" in rendered_prompt and fidelity >= 0.7:
            return " ".join(kept)
        if "Summarize" in rendered_prompt and fidelity >= 0.65:
            return " ".join(kept)
        return " ".join(kept)


@dataclass
class ScriptedRunner:
    """Return canned StepResults in order (or by rendered_prompt key). For tests."""

    queue: list[StepResult] | None = None
    by_prompt: dict[str, StepResult] | None = None

    def run_step(
        self,
        *,
        model: ModelSpec,
        hyperparameters: Hyperparameters,
        rendered_prompt: str,
    ) -> StepResult:
        if self.by_prompt is not None and rendered_prompt in self.by_prompt:
            return self.by_prompt[rendered_prompt]
        if self.queue:
            return self.queue.pop(0)
        raise LookupError(
            f"ScriptedRunner has no canned response for prompt: {rendered_prompt!r}"
        )


def run_pipeline(runner: Runner, genome: Genome, message: str) -> Response:
    """Render {{input}} → run_step → chain outputs; aggregate metrics; score final text."""
    if not genome.steps:
        raise ValueError("genome must contain at least one step")

    incoming = message
    total_latency = 0.0
    total_cost = 0.0
    total_tokens = 0
    final_message = ""
    last_step = genome.steps[-1]

    for step in genome.steps:
        rendered = render_prompt(step.prompt, incoming)
        result = runner.run_step(
            model=step.model,
            hyperparameters=step.hyperparameters.clamped(),
            rendered_prompt=rendered,
        )
        incoming = result.message
        final_message = result.message
        total_latency += result.latency_ms
        total_cost += result.cost
        total_tokens += result.total_tokens

    quality = _score_quality(
        final_message,
        last_step.model,
        last_step.hyperparameters.clamped().temperature,
        last_step.hyperparameters.clamped().top_p,
    )

    return Response(
        message=final_message,
        latency_ms=round(total_latency, 2),
        cost=round(total_cost, 8),
        total_tokens=total_tokens,
        quality=round(quality, 4),
    )


def _fidelity(model: ModelSpec, temperature: float, top_p: float) -> float:
    model_boost = {
        "mock-fast": 0.62,
        "mock-balanced": 0.78,
        "mock-quality": 0.9,
    }.get(model.name, 0.7)
    temp_term = math.exp(-((temperature - 0.3) ** 2) / (2 * 0.25**2))
    top_p_term = math.exp(-((top_p - 0.92) ** 2) / (2 * 0.12**2))
    return max(0.05, min(1.0, model_boost * (0.55 + 0.25 * temp_term + 0.2 * top_p_term)))


def _score_quality(
    text: str,
    model: ModelSpec,
    temperature: float,
    top_p: float,
) -> float:
    ref_tokens = set(REFERENCE_CLAIM.lower().split())
    out_tokens = set(text.lower().split())
    overlap = len(ref_tokens & out_tokens) / max(1, len(ref_tokens))
    fidelity = _fidelity(model, temperature, top_p)
    return max(0.0, min(1.0, 0.55 * overlap + 0.45 * fidelity))
