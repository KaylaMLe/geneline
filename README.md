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
2. **runners** execute each genome in the current generation against the input text → responses (`openrouter` by default; `mock` / `scripted` for offline tests). Genomes in a generation run **in parallel** (up to `max_parallel_genomes`); steps inside one genome stay sequential.
3. **scorer** turns those responses into fitness scores.
4. **evolver** breeds the next generation of genomes (hypers only) and the loop repeats.
5. **utils** (`io`, `prompt`, `types`) are shared helpers used across the package.

Pipeline topology, task prompts, and model names are immutable. Tunable genes: `temperature` and `top_p` only. Parallelism uses the stdlib thread pool — no extra packages beyond `uv sync`.

## Setup

From the repo root, prefer **uv** so installs stay in sync with `pyproject.toml` (avoids “I pip-installed a package and forgot to declare it”).

Install uv once: https://docs.astral.sh/uv/getting-started/installation/

```bash
cd /path/to/geneline
uv sync
```

That creates `.venv`, installs geneline in editable mode, and installs declared dependencies. Activate with `source .venv/bin/activate` (or run via `uv run ...` without activating).

**Managing dependencies** — use uv so `pyproject.toml` and the venv update together:

```bash
uv add some-package       # declare + install
uv remove some-package    # undeclare + uninstall
uv sync                   # make the venv match pyproject.toml
```

**Option A — no install.** Prefix with `PYTHONPATH=src` if you skip the venv. Third-party deps still need to be available, so prefer `uv sync` for a full working env:

```bash
cd /path/to/geneline
PYTHONPATH=src python -m ...
```

### Config

Tuner settings live in JSON (default: `examples/config.json`; mock: `examples/config.mock.json`). Important knobs:

| Key | Role |
|-----|------|
| `runner` | `openrouter` (default) or `mock` |
| `population_size` / `max_generations` | Search budget |
| `max_parallel_genomes` | Cap concurrent genome evals within a generation |
| `goal_score` | Early stop when best score reaches this |
| `weights.quality` / `weights.latency` / `weights.cost` | Fitness = quality − latency penalty − cost penalty |
| `latency_ref_ms` / `cost_ref` | Scale for those penalties |
| `quality.judge` | How final pipeline text is scored (see below) |
| `mutation_rate` | Per-gene chance to nudge temperature or top_p (applied to every child) |

**Quality judges** (`quality` object in config):

| `quality.judge` | Behavior |
|-----------------|----------|
| `answer` (default) | Numeric ground-truth check. Path via `quality.answer_path` (example: `examples/quality.answer.json`). Perfect score requires a bare number (no `$`) within tolerance of `expected`. |
| `constraints` | Legacy offline rubric: required term groups, forbidden hedges, length caps |
| `overlap` | Legacy bag-of-words overlap with a fixed reference claim (plus mock fidelity for `mock-*`) |
| `llm` | Reserved for a future LLM-as-judge; raises `NotImplementedError` today |

The default example averages shoe prices from an inventory in `message.txt` (synonyms like sneakers/heels; ignore accessories), then applies 10% tax. Step-1 average is `70`; **final** gold (what `AnswerJudge` scores) is `70 × 1.10 = 77` / `77.00`.

**Latency:** each model call still records wall-clock latency in logs/JSON. Scoring **downweights** it in the OpenRouter example (`weights.latency: 0.05`) so API jitter does not dominate when quality already separates candidates. Set `"latency": 0` to ignore latency in the score entirely.

**Cost:** prefers OpenRouter `usage.cost`, falling back to `total_tokens * cost_per_token` from the genome.

## Commands



### Unit tests

```bash
python -m unittest discover -s tests -v
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

## Genome shape

Each step is a data-processing stage. Prompts are task templates with exactly one `{{input}}` placeholder (`message.txt` for step 1, previous step output thereafter) — not role/system personas. Steady task/format rules live in the step prompts; `message.txt` holds the product inventory data.

```json
{
  "id": "seed-0",
  "steps": [
    {
      "prompt": "From the inventory below, get the average price of all shoes (...). Round to 2 decimal places. Reply with a plain number only — no dollar sign.\\n\\n{{input}}",
      "model": { "name": "openai/gpt-4o-mini", "cost_per_token": 0.00000015 },
      "hyperparameters": { "temperature": 0.4, "top_p": 0.9 }
    },
    {
      "prompt": "Take this average shoe price and add a 10% sales tax. Round to 2 decimal places. Reply with a plain number only — no dollar sign.\\n\\n{{input}}",
      "model": { "name": "openai/gpt-4o-mini", "cost_per_token": 0.00000015 },
      "hyperparameters": { "temperature": 0.3, "top_p": 0.9 }
    }
  ]
}
```



## Outputs


| File                        | Contents                                         |
| --------------------------- | ------------------------------------------------ |
| `best_genome.json`          | Winning pipeline genome                          |
| `best_result.json`          | Score, response metrics, stop reason             |
| `generation_N_results.json` | All `{genome, score, response}` for generation N |
| `history.json`              | Compact per-generation summary + full results    |




## Layout

```
examples/                # default OpenRouter genome/config; *.mock.json for offline
src/geneline/
  cli.py                 # CLI entrypoint
  tuner.py               # Main loop orchestration
  scorer.py              # Multi-objective fitness
  evolver.py             # Selection, crossover, mutation (hypers only)
  quality/               # Pluggable quality judges (answer, constraints, …)
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

- Implement the live LLM-as-judge path for open-ended tasks without a numeric gold answer.
- Allow prompt mutation once you want the genome’s text genes to evolve.
- Optionally mutate model choice among a configured OpenRouter allow-list.

