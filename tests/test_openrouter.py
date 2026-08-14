"""Unit tests for OpenRouterRunner parsing (no live network)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from geneline.runners.openrouter import OpenRouterError, OpenRouterRunner, load_api_key
from geneline.utils.types import Hyperparameters, ModelSpec


class OpenRouterParseTests(unittest.TestCase):
    def test_load_api_key_requires_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(OpenRouterError):
                load_api_key()

    def test_run_step_parses_usage_and_cost(self) -> None:
        payload = {
            "choices": [{"message": {"content": "Main claim about genetic algorithms."}}],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 12,
                "total_tokens": 52,
                "cost": 0.000042,
            },
        }
        raw = json.dumps(payload).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = raw
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        runner = OpenRouterRunner(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_response) as mocked:
            result = runner.run_step(
                model=ModelSpec(
                    name="openai/gpt-4o-mini",
                    cost_per_input_token=0.00000015,
                    cost_per_output_token=0.00000015,
                ),
                hyperparameters=Hyperparameters(temperature=0.3, top_p=0.9),
                rendered_prompt="Extract the main claim from this text: hello",
            )

        self.assertEqual(result.message, "Main claim about genetic algorithms.")
        self.assertEqual(result.total_tokens, 52)
        self.assertEqual(result.cost, 0.000042)
        self.assertGreater(result.latency_ms, 0)
        mocked.assert_called_once()

    def test_run_step_falls_back_to_split_token_rates(self) -> None:
        payload = {
            "choices": [{"message": {"content": "77.00"}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        }
        raw = json.dumps(payload).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = raw
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        runner = OpenRouterRunner(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = runner.run_step(
                model=ModelSpec(
                    name="test",
                    cost_per_input_token=0.000001,
                    cost_per_output_token=0.000005,
                ),
                hyperparameters=Hyperparameters(temperature=0.3, top_p=0.9),
                rendered_prompt="hello",
            )

        self.assertEqual(result.total_tokens, 52)
        self.assertEqual(result.cost, 0.0001)


if __name__ == "__main__":
    unittest.main()
