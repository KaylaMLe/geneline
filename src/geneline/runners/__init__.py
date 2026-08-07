"""Pluggable pipeline runners: protocol, orchestration, and factory."""

from __future__ import annotations

from typing import Literal

from geneline import progress
from geneline.runners.mock import REFERENCE_CLAIM, MockRunner, ScriptedRunner, mock_fidelity
from geneline.runners.openrouter import OpenRouterError, OpenRouterRunner, load_api_key
from geneline.runners.protocol import Runner, StepResult
from geneline.utils.prompt import render_prompt
from geneline.utils.types import Genome, ModelSpec, Response

RunnerName = Literal["mock", "openrouter"]

__all__ = [
    "MockRunner",
    "OpenRouterError",
    "OpenRouterRunner",
    "Runner",
    "RunnerName",
    "ScriptedRunner",
    "StepResult",
    "build_runner",
    "load_api_key",
    "run_pipeline",
]


def build_runner(name: RunnerName, *, seed: int = 0) -> Runner:
    if name == "mock":
        return MockRunner(seed=seed)
    if name == "openrouter":
        return OpenRouterRunner.from_env()
    raise ValueError(f"unsupported runner: {name!r}")


def run_pipeline(
    runner: Runner,
    genome: Genome,
    message: str,
    *,
    label: str | None = None,
) -> Response:
    """Render {{input}} → run_step → chain outputs; aggregate metrics; score final text."""
    if not genome.steps:
        raise ValueError("genome must contain at least one step")

    prefix = f"{label} " if label else ""
    incoming = message
    total_latency = 0.0
    total_cost = 0.0
    total_tokens = 0
    final_message = ""
    last_step = genome.steps[-1]

    for step_index, step in enumerate(genome.steps, start=1):
        progress.log(
            f"    {prefix}step {step_index}/{len(genome.steps)} model={step.model.name}"
        )
        rendered = render_prompt(step.prompt, incoming)
        result = runner.run_step(
            model=step.model,
            hyperparameters=step.hyperparameters.clamped(),
            rendered_prompt=rendered,
            label=label,
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


def _score_quality(
    text: str,
    model: ModelSpec,
    temperature: float,
    top_p: float,
) -> float:
    ref_tokens = set(REFERENCE_CLAIM.lower().split())
    out_tokens = set(text.lower().split())
    overlap = len(ref_tokens & out_tokens) / max(1, len(ref_tokens))
    if not model.name.startswith("mock-"):
        return max(0.0, min(1.0, overlap))
    fidelity = mock_fidelity(model, temperature, top_p)
    return max(0.0, min(1.0, 0.55 * overlap + 0.45 * fidelity))
