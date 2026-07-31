# geneline

Genetic-algorithm automatic tuner for LLM pipeline genomes. Input and output are plain JSON files — no database. **OpenRouter is the default runner.**

## Architecture

```mermaid
flowchart LR
  cli[cli] -->|args| tuner[tuner]
  tuner -->|logs| cli
  subgraph utils [utils]
    io[io]
    prompt[prompt]
    types[types]
  end
  subgraph tunerBox [tuner]
    runner[runners]
    scorer[scorer]
    evolver[evolver]
    runner -->|responses| scorer
    scorer -->|scored genomes| evolver
    evolver -->|next generation genomes| runner
  end
  cli --> tunerBox
  tunerBox -.-> utils
```

1. **cli** passes paths/args into **tuner** and prints run logs.
2. **runners** execute each genome in the current generation against the input text → responses (`openrouter` by default; `mock` / `scripted` for offline tests).
3. **scorer** turns those responses into fitness scores.
4. **evolver** breeds the next generation of genomes (hypers only) and the loop repeats.
5. **utils** (`io`, `prompt`, `types`) are shared helpers used across the package.

Pipeline topology, task prompts, and model names are immutable. Tunable genes: `temperature` and `top_p` only.

## Setup

From the repo root, pick one:

**Option A — no install.** Prefix Python commands with `PYTHONPATH=src` so the package imports without installing:

```bash
cd /path/to/geneline
PYTHONPATH=src python -m ...
```

**Option B — install once.** Use a project venv (required on Debian/Ubuntu’s externally managed Python):

```bash
cd /path/to/geneline
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

After that, plain `python -m ...` or the `geneline` console script work with no `PYTHONPATH`. Reactivate later with `source .venv/bin/activate`.

## Commands

### Unit tests

```bash
unittest discover -s tests -v
```

### Mock smoke run

```bash
geneline \
  --genome examples/genome.mock.json \
  --message examples/message.txt \
  --config examples/config.mock.json \
  --runner mock \
  --out runs/mock
```

### OpenRouter run (default)

```bash
export OPENROUTER_API_KEY=sk-or-...

geneline \
  --genome examples/genome.json \
  --message examples/message.txt \
  --config examples/config.json \
  --out runs/latest
```

Latency is wall-clock. Cost prefers OpenRouter `usage.cost`, falling back to `total_tokens * cost_per_token` from the genome.

## Genome shape

Each step is a data-processing stage. Prompts are task templates with exactly one `{{input}}` placeholder (message.txt for step 1, previous step output thereafter) — not role/system personas. `message.txt` is plain input data; instructions live only in the step prompts.

```json
{
  "id": "seed-0",
  "steps": [
    {
      "prompt": "Summarize this text: {{input}}",
      "model": { "name": "openai/gpt-4o-mini", "cost_per_token": 0.00000015 },
      "hyperparameters": { "temperature": 0.4, "top_p": 0.9 }
    },
    {
      "prompt": "Extract the main claim from this text: {{input}}",
      "model": { "name": "openai/gpt-4o-mini", "cost_per_token": 0.00000015 },
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
examples/                # default OpenRouter genome/config; *.mock.json for offline
src/geneline/
  cli.py                 # CLI entrypoint
  tuner.py               # Main loop orchestration
  scorer.py              # Multi-objective fitness
  evolver.py             # Selection, crossover, mutation (hypers only)
  runners/
    __init__.py          # Protocol re-exports, run_pipeline, build_runner
    protocol.py          # Runner + StepResult
    openrouter.py        # Live OpenRouter Runner
    mock.py              # MockRunner + ScriptedRunner (tests)
  utils/
    io.py                # JSON / text file helpers
    prompt.py            # {{input}} template rendering
    types.py             # Genome / Response dataclasses
runs/                    # Written artifacts (gitignored)
```

## Next hooks

- Plug in a task-specific quality scorer instead of the lexical-overlap metric.
- Allow prompt mutation once you want the genome’s text genes to evolve.
- Optionally mutate model choice among a configured OpenRouter allow-list.
