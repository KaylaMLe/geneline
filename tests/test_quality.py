"""Offline unit tests for pluggable quality judges."""

from __future__ import annotations

import unittest
from pathlib import Path

from geneline.quality import build_quality_judge
from geneline.quality.answer import AnswerJudge
from geneline.quality.llm import LlmJudge
from geneline.quality.overlap import OverlapJudge
from geneline.quality.protocol import QualityContext
from geneline.runners import MockRunner, run_pipeline
from geneline.runners.mock import REFERENCE_CLAIM
from geneline.utils.io import read_json
from geneline.utils.types import Genome, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
ANSWER_PATH = ROOT / "examples" / "quality.answer.json"
EXAMPLE_MESSAGE = (ROOT / "examples" / "message.txt").read_text(encoding="utf-8").strip()
EXAMPLE_GENOME = Genome.from_dict(read_json(ROOT / "examples" / "genome.mock.json"))


def _ctx(*, model_name: str = "openai/gpt-4o-mini") -> QualityContext:
    return QualityContext(
        input_message=EXAMPLE_MESSAGE,
        model=ModelSpec(name=model_name, cost_per_token=0.00000015),
        temperature=0.3,
        top_p=0.9,
    )


class AnswerJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.judge = AnswerJudge.from_path(ANSWER_PATH)

    def test_exact_bare_number_scores_perfect(self) -> None:
        self.assertEqual(self.judge.score("77.00", context=_ctx()), 1.0)
        self.assertEqual(self.judge.score("77", context=_ctx()), 1.0)

    def test_dollar_sign_partial_credit(self) -> None:
        self.assertEqual(self.judge.score("$77.00", context=_ctx()), 0.5)

    def test_prose_wrapped_partial_credit(self) -> None:
        self.assertEqual(self.judge.score("The answer is 77", context=_ctx()), 0.5)

    def test_wrong_number_is_zero(self) -> None:
        self.assertEqual(self.judge.score("70.00", context=_ctx()), 0.0)
        self.assertEqual(self.judge.score("nope", context=_ctx()), 0.0)


class OverlapJudgeTests(unittest.TestCase):
    def test_matches_legacy_overlap_math(self) -> None:
        text = (
            "Genetic algorithms evolve candidates via selection, crossover, and mutation "
            "until a fitness goal is reached."
        )
        ref_tokens = set(REFERENCE_CLAIM.lower().split())
        out_tokens = set(text.lower().split())
        expected = len(ref_tokens & out_tokens) / max(1, len(ref_tokens))
        got = OverlapJudge().score(text, context=_ctx(model_name="openai/gpt-4o-mini"))
        self.assertAlmostEqual(got, expected, places=6)


class BuildJudgeTests(unittest.TestCase):
    def test_default_is_answer(self) -> None:
        judge = build_quality_judge(
            {"judge": "answer", "answer_path": str(ANSWER_PATH)}
        )
        self.assertIsInstance(judge, AnswerJudge)

    def test_llm_stub_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            LlmJudge().score("anything", context=_ctx())


class PipelineAnswerSmokeTests(unittest.TestCase):
    def test_mock_pipeline_with_answer_judge(self) -> None:
        judge = AnswerJudge.from_path(ANSWER_PATH)
        response = run_pipeline(
            MockRunner(seed=42),
            EXAMPLE_GENOME,
            EXAMPLE_MESSAGE,
            quality_judge=judge,
        )
        self.assertGreaterEqual(response.quality, 0.0)
        self.assertLessEqual(response.quality, 1.0)
        self.assertTrue(response.message)


if __name__ == "__main__":
    unittest.main()
