"""Constraint / rubric quality judge (offline, deterministic)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geneline.quality.protocol import QualityContext
from geneline.utils.io import read_json


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+", text.strip()) if p.strip()]
    return max(1, len(parts)) if text.strip() else 0


@dataclass
class ConstraintsJudge:
    """Score text against required term groups, forbidden terms, and length/structure."""

    required_term_groups: list[list[str]]
    forbidden_terms: list[str]
    min_chars: int = 20
    max_chars: int = 400
    max_sentences: int = 2
    required_weight: float = 0.7
    structure_weight: float = 0.2
    forbidden_weight: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintsJudge:
        groups = data.get("required_terms") or []
        normalized_groups: list[list[str]] = []
        for group in groups:
            if isinstance(group, str):
                normalized_groups.append([group])
            else:
                normalized_groups.append([str(term) for term in group])
        return cls(
            required_term_groups=normalized_groups,
            forbidden_terms=[str(t) for t in data.get("forbidden_terms") or []],
            min_chars=int(data.get("min_chars", 20)),
            max_chars=int(data.get("max_chars", 400)),
            max_sentences=int(data.get("max_sentences", 2)),
            required_weight=float(data.get("required_weight", 0.7)),
            structure_weight=float(data.get("structure_weight", 0.2)),
            forbidden_weight=float(data.get("forbidden_weight", 0.1)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> ConstraintsJudge:
        return cls.from_dict(read_json(path))

    def score(self, text: str, *, context: QualityContext) -> float:
        del context  # available for future rubric features
        normalized = _normalize(text)
        required_score = self._required_score(normalized)
        structure_score = self._structure_score(text)
        forbidden_score = self._forbidden_score(normalized)
        total = (
            self.required_weight * required_score
            + self.structure_weight * structure_score
            + self.forbidden_weight * forbidden_score
        )
        return max(0.0, min(1.0, total))

    def _required_score(self, normalized: str) -> float:
        if not self.required_term_groups:
            return 1.0
        hits = 0
        for group in self.required_term_groups:
            if any(_normalize(term) in normalized for term in group):
                hits += 1
        return hits / len(self.required_term_groups)

    def _structure_score(self, text: str) -> float:
        length = len(text.strip())
        sentences = _sentence_count(text)
        checks = [
            length >= self.min_chars,
            length <= self.max_chars,
            sentences <= self.max_sentences,
            sentences >= 1,
        ]
        return sum(1 for ok in checks if ok) / len(checks)

    def _forbidden_score(self, normalized: str) -> float:
        if not self.forbidden_terms:
            return 1.0
        # Full score if none appear; else fraction of forbidden terms absent.
        absent = sum(1 for term in self.forbidden_terms if _normalize(term) not in normalized)
        return absent / len(self.forbidden_terms)
