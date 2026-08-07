"""Deterministic and scripted runners for offline tests."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from geneline.runners.protocol import StepResult
from geneline.utils.types import Hyperparameters, ModelSpec

# Ideal final claim used by the mock text generator / mock quality blend.
REFERENCE_CLAIM = (
    "Genetic algorithms evolve candidates via selection, crossover, and mutation "
    "until a fitness goal is reached."
)


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
        label: str | None = None,
    ) -> StepResult:
        del label  # accepted for Runner protocol / parallel logging
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
        fidelity = mock_fidelity(model, temperature, top_p)
        words = REFERENCE_CLAIM.split()
        keep = max(4, int(len(words) * fidelity))
        kept = words[:keep]

        if fidelity < 0.75:
            noise = ["roughly", "perhaps", "sort of", "maybe", "kinda"]
            kept.insert(rng.randint(0, len(kept)), rng.choice(noise))
        if fidelity < 0.55:
            kept.append("etc.")

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
        label: str | None = None,
    ) -> StepResult:
        del label
        if self.by_prompt is not None and rendered_prompt in self.by_prompt:
            return self.by_prompt[rendered_prompt]
        if self.queue:
            return self.queue.pop(0)
        raise LookupError(
            f"ScriptedRunner has no canned response for prompt: {rendered_prompt!r}"
        )


def mock_fidelity(model: ModelSpec, temperature: float, top_p: float) -> float:
    model_boost = {
        "mock-fast": 0.62,
        "mock-balanced": 0.78,
        "mock-quality": 0.9,
    }.get(model.name, 0.7)
    temp_term = math.exp(-((temperature - 0.3) ** 2) / (2 * 0.25**2))
    top_p_term = math.exp(-((top_p - 0.92) ** 2) / (2 * 0.12**2))
    return max(0.05, min(1.0, model_boost * (0.55 + 0.25 * temp_term + 0.2 * top_p_term)))
