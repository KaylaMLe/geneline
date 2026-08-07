"""Ground-truth numeric answer quality judge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geneline.quality.protocol import QualityContext
from geneline.utils.io import read_json

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


@dataclass
class AnswerJudge:
    """Score final text against an expected numeric answer."""

    expected: float
    tolerance: float = 0.01
    require_bare_number: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerJudge:
        if "expected" not in data:
            raise ValueError("answer judge requires 'expected'")
        return cls(
            expected=float(data["expected"]),
            tolerance=float(data.get("tolerance", 0.01)),
            require_bare_number=bool(data.get("require_bare_number", True)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> AnswerJudge:
        return cls.from_dict(read_json(path))

    def score(self, text: str, *, context: QualityContext) -> float:
        del context
        value = _parse_number(text)
        if value is None:
            return 0.0
        if abs(value - self.expected) > self.tolerance:
            return 0.0
        if self.require_bare_number and not _is_bare_number(text):
            return 0.5
        return 1.0
