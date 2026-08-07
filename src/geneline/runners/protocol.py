"""Shared Runner protocol and step result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from geneline.utils.types import Hyperparameters, ModelSpec


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
        label: str | None = None,
    ) -> StepResult:
        """Execute one data-processing step against an already-rendered prompt."""
