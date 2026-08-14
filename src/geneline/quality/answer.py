"""Ground-truth numeric answer quality judge."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from geneline.quality.protocol import QualityContext
from geneline.utils.io import read_json

MatchMode = Literal["exact", "distance"]

# Bare number: optional leading sign, digits, optional decimal part.
_BARE_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_NUMBER_IN_TEXT = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?")


def _parse_number(text: str) -> float | None:
    """Extract a plausible number; strips $ and commas for value comparison only."""
    cleaned = text.strip().replace(",", "")
    # Prefer a $-prefixed amount if present, else first number-like token.
    dollar = re.search(r"\$\s*([+-]?(?:\d+(?:\.\d+)?))", cleaned)
    if dollar:
        try:
            return float(dollar.group(1))
        except ValueError:
            pass
    match = _NUMBER_IN_TEXT.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _is_bare_number(text: str) -> bool:
    return bool(_BARE_NUMBER.match(text.strip()))


@dataclass(frozen=True)
class StepAnswerSpec:
    """Per-step gold for optional step-by-step scoring."""

    expected: float
    match: MatchMode = "exact"
    tolerance: float = 0.01
    distance_scale: float = 30.0
    require_bare_number: bool | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        defaults: AnswerJudge | None = None,
    ) -> StepAnswerSpec:
        if "expected" not in data:
            raise ValueError("step answer spec requires 'expected'")
        match = str(data.get("match", defaults.match if defaults else "exact")).lower()
        if match not in ("exact", "distance"):
            raise ValueError(f"unsupported match mode: {match!r}")
        return cls(
            expected=float(data["expected"]),
            match=match,  # type: ignore[arg-type]
            tolerance=float(
                data.get("tolerance", defaults.tolerance if defaults else 0.01)
            ),
            distance_scale=float(
                data.get(
                    "distance_scale",
                    defaults.distance_scale if defaults else 30.0,
                )
            ),
            require_bare_number=(
                bool(data["require_bare_number"])
                if "require_bare_number" in data
                else None
            ),
        )


@dataclass
class AnswerJudge:
    """Score pipeline text against expected numeric answer(s).

    Defaults preserve final-only exact match. Opt into ``match: "distance"``
    and/or a ``steps`` list via config JSON.
    """

    expected: float
    tolerance: float = 0.01
    require_bare_number: bool = True
    match: MatchMode = "exact"
    distance_scale: float = 30.0
    steps: list[StepAnswerSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerJudge:
        steps_raw = data.get("steps") or []
        if "expected" not in data and not steps_raw:
            raise ValueError("answer judge requires 'expected' (or non-empty steps)")
        match = str(data.get("match", "exact")).lower()
        if match not in ("exact", "distance"):
            raise ValueError(f"unsupported match mode: {match!r}")
        # expected may be omitted only when steps alone are used; still need a
        # final expected — use last step's expected if top-level missing.
        if "expected" in data:
            expected = float(data["expected"])
        else:
            expected = float(steps_raw[-1]["expected"])
        judge = cls(
            expected=expected,
            tolerance=float(data.get("tolerance", 0.01)),
            require_bare_number=bool(data.get("require_bare_number", True)),
            match=match,  # type: ignore[arg-type]
            distance_scale=float(data.get("distance_scale", 30.0)),
        )
        judge.steps = [
            StepAnswerSpec.from_dict(item, defaults=judge)
            for item in steps_raw
            if item is not None
        ]
        return judge

    @classmethod
    def from_path(cls, path: str | Path) -> AnswerJudge:
        return cls.from_dict(read_json(path))

    def score(self, text: str, *, context: QualityContext) -> float:
        final_score = self._score_text(
            text,
            expected=self.expected,
            match=self.match,
            tolerance=self.tolerance,
            distance_scale=self.distance_scale,
            require_bare_number=self.require_bare_number,
        )
        if not self.steps:
            return final_score

        step_scores: list[float] = []
        for index, spec in enumerate(self.steps):
            if index >= len(context.step_messages):
                break
            require_bare = (
                self.require_bare_number
                if spec.require_bare_number is None
                else spec.require_bare_number
            )
            step_scores.append(
                self._score_text(
                    context.step_messages[index],
                    expected=spec.expected,
                    match=spec.match,
                    tolerance=spec.tolerance,
                    distance_scale=spec.distance_scale,
                    require_bare_number=require_bare,
                )
            )
        combined = step_scores + [final_score]
        return sum(combined) / len(combined)

    def _score_text(
        self,
        text: str,
        *,
        expected: float,
        match: MatchMode,
        tolerance: float,
        distance_scale: float,
        require_bare_number: bool,
    ) -> float:
        value = _parse_number(text)
        if value is None:
            return 0.0
        err = abs(value - expected)
        if match == "exact":
            value_score = 1.0 if err <= tolerance else 0.0
        else:
            # Full credit inside tolerance; linear falloff over distance_scale.
            scale = max(distance_scale, 1e-9)
            value_score = max(0.0, 1.0 - max(0.0, err - tolerance) / scale)
        if require_bare_number and not _is_bare_number(text):
            return 0.5 * value_score
        return value_score
