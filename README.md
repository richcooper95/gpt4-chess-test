# Chess Eval

An [Inspect](https://inspect.aisi.org.uk/) evaluation that tests LLMs' chess ability against
[Stockfish](https://stockfishchess.org/) at various skill levels. Models are accessed via
[OpenRouter](https://openrouter.ai/), so any model available there can be evaluated with no
code changes.

## Eval variants

| File | Description |
|---|---|
| `main.py` | Stateless — each turn sends a standalone prompt with the current PGN |
| `main_in_context.py` | In-context — maintains the full conversation history so the model can build on its prior reasoning |
| `main_image.py` | Image — like in-context, but each turn also includes a rendered board image (requires a vision-capable model) |

## What it measures

For each model, the eval plays chess games across a matrix of:

- **Stockfish skill levels** (0-20, configurable)
- **LLM color** (white and black)
- **Repeated games** (via `--epochs`)

And tracks:

| Metric | Description |
|---|---|
| **Win/loss/draw rate** | `accuracy` in Inspect (1.0 = win, 0.5 = draw, 0.0 = loss) |
| **Game length** | Average full moves per game (longer against stronger Stockfish = better play) |
| **Invalid move rate** | Average invalid move attempts per game (lower = better notation understanding) |

## Setup

### Prerequisites

- Python 3.10+
- A [Stockfish](https://stockfishchess.org/) binary (e.g. `brew install stockfish` on macOS)
- An [OpenRouter](https://openrouter.ai/) API key
- For `main_image.py`: no extra system dependencies (uses [fentoboardimage](https://pypi.org/project/fentoboardimage/) + Pillow)

### Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set your API key

```bash
export OPENROUTER_API_KEY=your-openrouter-api-key
```

## Usage

Run an evaluation with `inspect eval`:

```bash
# Stateless (main.py)
inspect eval main.py --model openrouter/openai/gpt-4o --epochs 3

# In-context conversation (main_in_context.py)
inspect eval main_in_context.py --model openrouter/openai/gpt-4o --epochs 3

# With board images (main_image.py) — requires a vision-capable model
inspect eval main_image.py --model openrouter/openai/gpt-4o --epochs 3
```

### Task parameters

Customise Stockfish levels and path with `-T`:

```bash
inspect eval main.py \
  --model openrouter/openai/gpt-4o \
  -T stockfish_levels=[1,5,10,20] \
  -T stockfish_path=/opt/homebrew/bin/stockfish \
  --epochs 3
```

| Parameter | Default | Description |
|---|---|---|
| `stockfish_levels` | `[1, 5, 10, 20]` | List of Stockfish skill levels (0-20) |
| `stockfish_path` | `/opt/homebrew/bin/stockfish` | Path to the Stockfish binary |

### Comparing models

Run the same eval with different `--model` flags:

```bash
inspect eval main.py --model openrouter/openai/gpt-4o --epochs 3
inspect eval main.py --model openrouter/anthropic/claude-sonnet-4-20250514 --epochs 3
inspect eval main.py --model openrouter/google/gemini-2.0-flash-001 --epochs 3
inspect eval main.py --model openrouter/deepseek/deepseek-chat --epochs 3
```

### Viewing results

Inspect includes a web UI for browsing results:

```bash
inspect view
```

This shows per-game scores, metrics, full LLM conversation transcripts, and metadata.
You can filter by sample ID (e.g. `level-10-white`) and compare across model runs.

## How it works

Each eval run creates **`len(stockfish_levels) * 2`** samples (one per level per color).
With `--epochs N`, each sample is played N times, so the total number of games is
`len(stockfish_levels) * 2 * N`.

The game loop:

1. A Stockfish engine is opened and configured to the sample's skill level
2. Moves alternate between Stockfish and the LLM based on the sample's color
3. On each LLM turn, the full PGN is sent as a prompt and the response is parsed as SAN
4. Invalid moves are retried up to 5 times (with feedback about which moves are invalid)
5. The game ends on checkmate, stalemate, draw, or if the LLM fails to produce a valid move

## Background

This project started as a simple script pitting GPT-4 against Stockfish (see git history).
It has since been rewritten as a proper eval harness using Inspect to test multiple
frontier models systematically.
