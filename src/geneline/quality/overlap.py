"""Legacy lexical-overlap quality judge (plus mock fidelity blend)."""

from __future__ import annotations

from dataclasses import dataclass

from geneline.quality.protocol import QualityContext
from geneline.runners.mock import REFERENCE_CLAIM, mock_fidelity


@dataclass
class OverlapJudge:
    """Bag-of-words overlap with REFERENCE_CLAIM; blends mock_fidelity for mock-* models."""

    def score(self, text: str, *, context: QualityContext) -> float:
        ref_tokens = set(REFERENCE_CLAIM.lower().split())
        out_tokens = set(text.lower().split())
        overlap = len(ref_tokens & out_tokens) / max(1, len(ref_tokens))
        model_name = context.model.name if context.model else ""
        if not model_name.startswith("mock-"):
            return max(0.0, min(1.0, overlap))
        temperature = context.temperature if context.temperature is not None else 0.3
        top_p = context.top_p if context.top_p is not None else 0.9
        assert context.model is not None
        fidelity = mock_fidelity(context.model, temperature, top_p)
        return max(0.0, min(1.0, 0.55 * overlap + 0.45 * fidelity))
