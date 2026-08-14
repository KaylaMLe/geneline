"""JSON-serializable genome / response types for the geneline tuner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ModelSpec:
    name: str
    cost_per_input_token: float
    cost_per_output_token: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        name = str(data["name"])
        if "cost_per_input_token" in data and "cost_per_output_token" in data:
            return cls(
                name=name,
                cost_per_input_token=float(data["cost_per_input_token"]),
                cost_per_output_token=float(data["cost_per_output_token"]),
            )
        if "cost_per_token" in data:
            blended = float(data["cost_per_token"])
            return cls(
                name=name,
                cost_per_input_token=blended,
                cost_per_output_token=blended,
            )
        raise ValueError(
            "model requires cost_per_input_token and cost_per_output_token "
            "(or legacy cost_per_token)"
        )

    @property
    def cost_per_token(self) -> float:
        """Blended rate for legacy callers; prefer split fields for costing."""
        return (self.cost_per_input_token + self.cost_per_output_token) / 2

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.cost_per_input_token
            + completion_tokens * self.cost_per_output_token
        )


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
    """One data-processing stage.

    ``prompt`` is a fixed task template with exactly one ``{{input}}``
    placeholder (not a role/system persona). Model choice evolves first;
    hypers may be fine-tuned in a second phase with models frozen.
    """

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
    step_messages: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Response:
        return cls(
            message=data["message"],
            latency_ms=float(data["latency_ms"]),
            cost=float(data["cost"]),
            total_tokens=int(data["total_tokens"]),
            quality=float(data["quality"]),
            step_messages=[str(m) for m in data.get("step_messages") or []],
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
