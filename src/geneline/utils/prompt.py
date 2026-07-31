"""Prompt templating for pipeline steps."""

from __future__ import annotations

INPUT_PLACEHOLDER = "{{input}}"


def render_prompt(template: str, input_text: str) -> str:
    """Replace exactly one ``{{input}}`` placeholder with ``input_text``."""
    count = template.count(INPUT_PLACEHOLDER)
    if count != 1:
        raise ValueError(
            f"prompt must contain exactly one {INPUT_PLACEHOLDER!r} placeholder, found {count}"
        )
    return template.replace(INPUT_PLACEHOLDER, input_text)
