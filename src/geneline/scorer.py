"""Multi-objective response scoring: quality − latency penalty − cost penalty."""

from __future__ import annotations

from dataclasses import dataclass

from geneline.utils.types import Response


@dataclass
class Scorer:
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    cost_weight: float = 0.25
    latency_ref_ms: float = 800.0
    cost_ref: float = 0.002

    def score(self, response: Response) -> tuple[float, dict[str, float]]:
        quality = max(0.0, min(1.0, response.quality))
        latency_penalty = min(1.0, response.latency_ms / max(self.latency_ref_ms, 1e-9))
        cost_penalty = min(1.0, response.cost / max(self.cost_ref, 1e-12))

        total = (
            self.quality_weight * quality
            - self.latency_weight * latency_penalty
            - self.cost_weight * cost_penalty
        )
        components = {
            "quality": round(quality, 4),
            "latency_penalty": round(latency_penalty, 4),
            "cost_penalty": round(cost_penalty, 4),
            "weighted_total": round(total, 4),
        }
        return total, components
