"""Execute a pipeline genome against an input message.

Prototype uses a deterministic mock model so you can evolve without API keys.
Swap `MockRunner` for a real provider runner later.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from geneline.types import Genome, ModelSpec, Response


# Ideal summary used only by the mock quality metric.
REFERENCE_SUMMARY = (
    "Genetic algorithms evolve candidates via selection, crossover, and mutation "
    "until a fitness goal is reached."
)


@dataclass
class MockRunner:
    """Simulates latency, cost, and quality from model + hyperparameters."""

    rng: random.Random

    def run(self, genome: Genome, message: str) -> Response:
        if not genome.steps:
            raise ValueError("genome must contain at least one step")

        # Multi-step pipelines: last step is the "answer" producer for now.
        step = genome.steps[-1]
        hp = step.hyperparameters.clamped()
        model = step.model

        prompt_tokens = max(1, len(message.split()) + len(step.prompt.split()))
        completion_tokens = self._completion_tokens(model, hp.temperature)
        total_tokens = prompt_tokens + completion_tokens
        cost = total_tokens * model.cost_per_token

        base_latency = {
            "mock-fast": 180.0,
            "mock-balanced": 420.0,
            "mock-quality": 780.0,
        }.get(model.name, 500.0)

        # Higher temperature / lower top_p → slightly more variable latency.
        jitter = self.rng.uniform(-40.0, 80.0) + (hp.temperature * 35.0)
        latency_ms = max(50.0, base_latency + jitter + (completion_tokens * 1.5))

        text = self._generate_text(model, hp.temperature, hp.top_p, message)
        quality = self._score_quality(text, model, hp.temperature, hp.top_p)

        return Response(
            message=text,
            latency_ms=round(latency_ms, 2),
            cost=round(cost, 8),
            total_tokens=total_tokens,
            quality=round(quality, 4),
        )

    def _completion_tokens(self, model: ModelSpec, temperature: float) -> int:
        base = {"mock-fast": 28, "mock-balanced": 40, "mock-quality": 55}.get(model.name, 36)
        return max(12, int(base + temperature * 12 + self.rng.randint(-4, 6)))

    def _generate_text(
        self,
        model: ModelSpec,
        temperature: float,
        top_p: float,
        message: str,
    ) -> str:
        """Produce a pseudo-response whose fidelity depends on model + params."""
        digest = hashlib.sha256(
            f"{model.name}|{temperature:.3f}|{top_p:.3f}|{message}".encode()
        ).hexdigest()
        self.rng.seed(int(digest[:12], 16) ^ self.rng.randint(0, 2**16))

        # Prefer mid-low temperature and high top_p for this summarization task.
        fidelity = self._fidelity(model, temperature, top_p)
        words = REFERENCE_SUMMARY.split()
        keep = max(4, int(len(words) * fidelity))
        kept = words[:keep]

        if fidelity < 0.75:
            noise = ["roughly", "perhaps", "sort of", "maybe", "kinda"]
            kept.insert(self.rng.randint(0, len(kept)), self.rng.choice(noise))
        if fidelity < 0.55:
            kept.append("etc.")

        return " ".join(kept)

    def _score_quality(
        self,
        text: str,
        model: ModelSpec,
        temperature: float,
        top_p: float,
    ) -> float:
        ref_tokens = set(REFERENCE_SUMMARY.lower().split())
        out_tokens = set(text.lower().split())
        overlap = len(ref_tokens & out_tokens) / max(1, len(ref_tokens))
        fidelity = self._fidelity(model, temperature, top_p)
        # Blend lexical overlap with the parametric fidelity prior.
        return max(0.0, min(1.0, 0.55 * overlap + 0.45 * fidelity))

    def _fidelity(self, model: ModelSpec, temperature: float, top_p: float) -> float:
        model_boost = {
            "mock-fast": 0.62,
            "mock-balanced": 0.78,
            "mock-quality": 0.9,
        }.get(model.name, 0.7)
        # Peak around temp≈0.3 and top_p≈0.92 for this toy task.
        temp_term = math.exp(-((temperature - 0.3) ** 2) / (2 * 0.25**2))
        top_p_term = math.exp(-((top_p - 0.92) ** 2) / (2 * 0.12**2))
        return max(0.05, min(1.0, model_boost * (0.55 + 0.25 * temp_term + 0.2 * top_p_term)))
