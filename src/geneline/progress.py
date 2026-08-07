"""Simple flushed progress lines so long runs don't look hung."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone


def log(message: str) -> None:
    """Print a timestamped progress line to stderr (flushed immediately)."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


class Timer:
    def __init__(self) -> None:
        self._started = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started
