# Multivac

Multivac turns structural-entropy analysis of Python projects into a time-series you can track commit by commit. Point it at any git repository to compute AST-pattern similarity, persist the history, and (optionally) host a dashboard that visualises entropy over time. The CLI doubles as a GitHub Actions step so you can monitor "vibe" regressions along with your other quality gates.

## Installation

```bash
pip install git+https://github.com/velocitatem/Multivac.git
# or, to isolate the CLI
pipx install git+https://github.com/velocitatem/Multivac.git
```

Multivac requires Python 3.9+ and the `git` executable available on PATH.

## Quickstart

Analyse the working tree exactly like the original one-shot script:

```bash
multivac analyze .
```

Build or update the commit timeline (creates `.multivac/entropy_history.json` by default):

```bash
multivac timeline --repo . --summary
```

Launch the lightweight dashboard (http://127.0.0.1:5800):

```bash
multivac serve --repo . --follow
```

## CLI Overview

- `multivac analyze [DIR]` – inspect the current filesystem snapshot. Pass `--json` for machine-readable output, `--depth` to increase AST depth, or `--min-patterns` to filter out trivial files.
- `multivac timeline` – walk commits in order, recomputing entropy snapshots. Key flags:
  - `--recompute` rebuilds from scratch.
  - `--summary` prints a Markdown digest of the latest results.
  - `--step-summary` appends that digest to `$GITHUB_STEP_SUMMARY` (GitHub Actions only).
  - `--history-dir` changes the output location (defaults to `<repo>/.multivac`).
  - `--follow` keeps polling for new commits (uses simple polling, no external watcher).
- `multivac serve` – runs the Flask dashboard backed by the same history file. Combine with `--follow` for continuous updates.

All commands accept `--depth` / `--min-patterns` to trade off precision vs. runtime. Depth 3 with a 15-pattern threshold works well for most medium projects.

## GitHub Actions

Add Multivac as a quality gate by installing the package and invoking `multivac timeline --summary --step-summary`. A minimal workflow looks like:

```yaml
name: entropy

on:
  push:
    branches: [ main ]
  pull_request:

jobs:
  entropy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install multivac
        run: |
          python -m pip install --upgrade pip
          pip install .
      - name: Compute entropy timeline
        run: |
          multivac timeline --repo . --summary --step-summary
```

The command appends a Markdown table to the workflow summary and stores artefacts under `.multivac/`. Use `--history-dir "$RUNNER_TEMP/multivac"` if you prefer ephemeral storage in CI.

## Local Development

```bash
python -m multivac analyze . --json
python -m multivac timeline --repo . --recompute --summary
python -m multivac serve --repo . --follow --interval 5 --port 5800
```

History files live under `.multivac/` by default. Delete the directory if you want to reset results (`multivac timeline --recompute` recreates it).
