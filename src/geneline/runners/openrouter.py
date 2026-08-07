"""OpenRouter chat-completions Runner (OpenAI-compatible HTTP API)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any

from geneline import progress
from geneline.runners.protocol import StepResult
from geneline.utils.types import Hyperparameters, ModelSpec

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ENV_API_KEY = "OPENROUTER_API_KEY"

load_dotenv()


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails."""


def load_api_key(explicit: str | None = None) -> str:
    key = (explicit or os.environ.get(ENV_API_KEY) or "").strip()
    if not key:
        raise OpenRouterError(
            f"OpenRouter API key missing. Set {ENV_API_KEY} or pass api_key=..."
        )
    return key


@dataclass
class OpenRouterRunner:
    """Live runner: POST /chat/completions for each pipeline step."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 90.0
    site_url: str = "https://github.com/local/geneline"
    site_title: str = "geneline"
    max_tokens: int = 256

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenRouterRunner:
        return cls(api_key=load_api_key(), **kwargs)

    def run_step(
        self,
        *,
        model: ModelSpec,
        hyperparameters: Hyperparameters,
        rendered_prompt: str,
        label: str | None = None,
    ) -> StepResult:
        hp = hyperparameters.clamped()
        payload = {
            "model": model.name,
            "messages": [{"role": "user", "content": rendered_prompt}],
            "temperature": hp.temperature,
            "top_p": hp.top_p,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        prefix = f"{label} " if label else ""
        started = time.perf_counter()
        progress.log(
            f"    {prefix}openrouter call model={model.name} "
            f"temp={hp.temperature:.3f} top_p={hp.top_p:.3f} "
            f"(waiting up to {self.timeout_s:.0f}s)..."
        )
        data = self._post_chat(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        progress.log(f"    {prefix}openrouter call finished in {latency_ms:.0f}ms")

        message = _extract_message(data)
        usage = data.get("usage") or {}
        total_tokens = int(usage.get("total_tokens") or 0)
        if total_tokens <= 0:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = prompt_tokens + completion_tokens

        if "cost" in usage and usage["cost"] is not None:
            cost = float(usage["cost"])
        else:
            cost = total_tokens * model.cost_per_token

        return StepResult(
            message=message,
            latency_ms=round(latency_ms, 2),
            cost=round(cost, 8),
            total_tokens=total_tokens,
        )

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_title,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenRouterError(
                f"OpenRouter HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc.reason}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter returned non-JSON response") from exc

        if isinstance(data, dict) and data.get("error"):
            raise OpenRouterError(f"OpenRouter error: {data['error']}")
        return data


def _extract_message(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError("OpenRouter response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise OpenRouterError("OpenRouter response missing message content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    text = str(content).strip()
    if not text:
        raise OpenRouterError("OpenRouter returned empty message content")
    return text
