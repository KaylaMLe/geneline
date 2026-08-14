"""LLM-as-judge seam for non-mock runs (not implemented yet)."""

from __future__ import annotations

from dataclasses import dataclass

from geneline.quality.protocol import QualityContext


@dataclass
class LlmJudge:
    """Placeholder for a future OpenRouter rubric judge.

    Planned behavior:
    - Call a judge model at temperature 0 with a fixed rubric prompt
    - Parse a numeric score in [0, 1] (or 1–5 scaled)
    - Reuse the OpenRouter runner / chat-completions path
    """

    def score(self, text: str, *, context: QualityContext) -> float:
        del text, context
        raise NotImplementedError(
            "LlmJudge is not implemented yet. "
            "Use quality.judge='answer' (numeric gold) or 'constraints' / 'overlap' for now."
        )
