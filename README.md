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

Pipeline topology, task prompts, and model names are immutable. Tunable genes: `temperature` and `top_p` only.

## Quick start

```bash
cd geneline
python3 -m pip install -e .
geneline --genome examples/genome.json --message examples/message.txt --config examples/config.json --out runs/latest
```

Or without installing:

```bash
PYTHONPATH=src python3 -m geneline --out runs/latest
```

## Genome shape

Each step is a data-processing stage. Prompts are task templates with exactly one `{{input}}` placeholder (message.txt for step 1, previous step output thereafter) — not role/system personas.

```json
{
  "id": "seed-0",
  "steps": [
    {
      "prompt": "Summarize this text: {{input}}",
      "model": { "name": "mock-balanced", "cost_per_token": 0.00001 },
      "hyperparameters": { "temperature": 0.4, "top_p": 0.9 }
    },
    {
      "prompt": "Extract the main claim from this text: {{input}}",
      "model": { "name": "mock-fast", "cost_per_token": 0.000002 },
      "hyperparameters": { "temperature": 0.3, "top_p": 0.9 }
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
  prompt.py        # {{input}} template rendering
  runner.py        # Runner protocol, MockRunner, ScriptedRunner, pipeline orchestrator
  scorer.py        # Multi-objective fitness
  evolver.py       # Selection, crossover, mutation (hypers only)
  tuner.py         # Main loop + JSON I/O orchestration
  cli.py           # CLI entrypoint
runs/              # Written artifacts (gitignored)
```

## Next hooks

- Add an OpenRouter (or other) `Runner` behind the same protocol; keep `MockRunner` / `ScriptedRunner` for tests.
- Plug in a task-specific quality scorer instead of the mock overlap metric.
- Allow prompt mutation once you want the genome’s text genes to evolve.
