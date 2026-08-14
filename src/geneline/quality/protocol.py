"""Quality judge protocol and context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from geneline.utils.types import Genome, ModelSpec


@dataclass(frozen=True)
class QualityContext:
    """Extra context available to judges (input text, genome, last-step model)."""

    input_message: str
    genome: Genome | None = None
    model: ModelSpec | None = None
    temperature: float | None = None
    top_p: float | None = None
    step_messages: tuple[str, ...] = ()


class QualityJudge(Protocol):
    def score(self, text: str, *, context: QualityContext) -> float:
        """Return quality in [0, 1]."""
