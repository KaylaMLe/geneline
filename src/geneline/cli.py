"""CLI for the geneline prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from geneline.tuner import run_from_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geneline",
        description="Genetic-algorithm automatic tuner for LLM pipeline genomes (JSON I/O).",
    )
    parser.add_argument(
        "--genome",
        type=Path,
        default=Path("examples/genome.json"),
        help="Path to input pipeline genome JSON",
    )
    parser.add_argument(
        "--message",
        type=Path,
        default=Path("examples/message.txt"),
        help="Path to input message text",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/config.json"),
        help="Path to tuner config JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/latest"),
        help="Directory for output JSON artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_from_paths(args.genome, args.message, args.config, args.out)
    best = result.best
    print(f"stopped: {result.stopped_reason} after {result.generations_run} generation(s)")
    print(f"best score: {best.score}")
    print(f"best genome: {best.genome.id} ({len(best.genome.steps)} steps)")
    for index, step in enumerate(best.genome.steps, start=1):
        print(
            f"  step {index}: model={step.model.name}  "
            f"temp={step.hyperparameters.temperature:.3f}  "
            f"top_p={step.hyperparameters.top_p:.3f}"
        )
    print(
        f"quality={best.components.get('quality')}  "
        f"latency_ms={best.response.latency_ms}  "
        f"cost={best.response.cost}"
    )
    print(f"wrote JSON outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
