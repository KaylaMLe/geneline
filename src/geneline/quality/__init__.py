"""Pluggable quality judges for pipeline final outputs."""

from __future__ import annotations

from typing import Any

from geneline.quality.protocol import QualityContext, QualityJudge

__all__ = [
    "QualityContext",
    "QualityJudge",
    "build_quality_judge",
]


def build_quality_judge(config: dict[str, Any] | None = None) -> QualityJudge:
    """Build a judge from a config ``quality`` object (or defaults)."""
    from geneline.quality.answer import AnswerJudge
    from geneline.quality.constraints import ConstraintsJudge
    from geneline.quality.llm import LlmJudge
    from geneline.quality.overlap import OverlapJudge

    cfg = config or {}
    name = str(cfg.get("judge", "answer")).lower()
    if name == "answer":
        if "expected" in cfg:
            return AnswerJudge.from_dict(cfg)
        path = cfg.get("answer_path", "examples/quality.answer.json")
        return AnswerJudge.from_path(str(path))
    if name == "constraints":
        path = cfg.get("constraints_path", "examples/quality.constraints.json")
        return ConstraintsJudge.from_path(str(path))
    if name == "overlap":
        return OverlapJudge()
    if name == "llm":
        return LlmJudge()
    raise ValueError(f"unsupported quality judge: {name!r}")
