"""JSON-serializable genome / response types for the geneline tuner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ModelSpec:
    name: str
    cost_per_token: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        return cls(name=data["name"], cost_per_token=float(data["cost_per_token"]))


@dataclass
class Hyperparameters:
    temperature: float
    top_p: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hyperparameters:
        return cls(temperature=float(data["temperature"]), top_p=float(data["top_p"]))

    def clamped(self) -> Hyperparameters:
        return Hyperparameters(
            temperature=min(2.0, max(0.0, self.temperature)),
            top_p=min(1.0, max(0.01, self.top_p)),
        )


@dataclass
class PipelineStep:
    """One pipeline stage. Prompt is immutable for now."""

    prompt: str
    model: ModelSpec
    hyperparameters: Hyperparameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStep:
        return cls(
            prompt=data["prompt"],
            model=ModelSpec.from_dict(data["model"]),
            hyperparameters=Hyperparameters.from_dict(data["hyperparameters"]),
        )


@dataclass
class Genome:
    steps: list[PipelineStep]
    id: str = field(default_factory=lambda: f"genome-{uuid4().hex[:8]}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        return cls(
            id=data.get("id", f"genome-{uuid4().hex[:8]}"),
            steps=[PipelineStep.from_dict(step) for step in data["steps"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def clone(self, *, new_id: bool = True) -> Genome:
        cloned = Genome.from_dict(deepcopy(self.to_dict()))
        if new_id:
            cloned.id = f"genome-{uuid4().hex[:8]}"
        return cloned


@dataclass
class Response:
    """Observed run metrics for a genome evaluation."""

    message: str
    latency_ms: float
    cost: float
    total_tokens: int
    quality: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Response:
        return cls(
            message=data["message"],
            latency_ms=float(data["latency_ms"]),
            cost=float(data["cost"]),
            total_tokens=int(data["total_tokens"]),
            quality=float(data["quality"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredGenome:
    genome: Genome
    score: float
    response: Response
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "score": self.score,
            "response": self.response.to_dict(),
            "components": self.components,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoredGenome:
        return cls(
            genome=Genome.from_dict(data["genome"]),
            score=float(data["score"]),
            response=Response.from_dict(data["response"]),
            components={k: float(v) for k, v in data.get("components", {}).items()},
        )
