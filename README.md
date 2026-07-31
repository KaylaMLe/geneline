# geneline

Genetic-algorithm automatic tuner for LLM pipeline genomes. Input and output are plain JSON files — no database.

## Flow

1. Load a seed **pipeline genome** (`examples/genome.json`) and an input **message**.
2. Expand into a generation of candidate genomes.
3. Run each genome through a **mock LLM runner** (swap for a real provider later).
4. **Score** on quality − latency penalty − cost penalty.
5. **Evolve** the next generation (weighted selection, hyperparameter averages, exclusions, mutation).
6. Repeat until `max_generations` or `goal_score`.
7. Write `best_genome.json`, per-generation results, and `history.json`.

Prompts are treated as immutable for now. Tunable genes: `temperature`, `top_p`, and model choice.

## Quick start

```bash
cd /home/kayla/dev/Projects/geneline
python3 -m pip install -e .
geneline --genome examples/genome.json --message examples/message.txt --config examples/config.json --out runs/latest
```

Or without installing:

```bash
PYTHONPATH=src python3 -m geneline.cli --out runs/latest
```

## Genome shape

```json
{
  "id": "seed-0",
  "steps": [
    {
      "prompt": "You are a concise technical writer...",
      "model": { "name": "mock-fast", "cost_per_token": 0.000002 },
      "hyperparameters": { "temperature": 0.4, "top_p": 0.9 }
    }
  ]
}
```

## Outputs

| File | Contents |
|------|----------|
| `best_genome.json` | Winning pipeline genome |
| `best_result.json` | Score, response metrics, stop reason |
| `generation_N_results.json` | All `{genome, score, response}` for generation N |
| `history.json` | Compact per-generation summary + full results |

## Layout

```
examples/          # seed genome, message, tuner config
src/geneline/
  types.py         # Genome / Response dataclasses
  runner.py        # Mock model execution
  scorer.py        # Multi-objective fitness
  evolver.py       # Selection, crossover, mutation
  tuner.py         # Main loop + JSON I/O orchestration
  cli.py           # CLI entrypoint
runs/              # Written artifacts (gitignored)
```

## Next hooks

- Replace `MockRunner` with a real provider (OpenAI, Anthropic, etc.).
- Plug in a task-specific quality scorer instead of the mock overlap metric.
- Allow prompt mutation once you want the genome’s text genes to evolve.
