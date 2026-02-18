#!/usr/bin/env python3
"""Analyze chess eval results from Inspect log files.

Usage:
    python3 analyze.py logs/*.eval
    python3 analyze.py logs/2026-02-17T18-05-08+00-00_chess-eval_*.eval
"""

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def load_eval(path: str) -> dict:
    """Load an eval file (zip archive) and return header + all sample scores."""
    with zipfile.ZipFile(path) as z:
        header = json.loads(z.read("header.json"))
        samples = []
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                sample = json.loads(z.read(name))
                samples.append(sample)
    return {"header": header, "samples": samples}


def extract_reasoning_config(header: dict) -> dict[str, str]:
    """Extract reasoning-related settings from the eval header."""
    gen_config = header["eval"].get("model_generate_config", {})
    model_args = header["eval"].get("model_args", {})
    reasoning = {}
    for key in ("reasoning_effort", "reasoning_tokens"):
        if key in gen_config and gen_config[key] is not None:
            reasoning[key] = str(gen_config[key])
    for key in ("reasoning_effort", "reasoning_tokens", "budget_tokens"):
        if key in model_args and model_args[key] is not None:
            reasoning[key] = str(model_args[key])
    return reasoning


def make_run_label(model: str, reasoning: dict[str, str]) -> str:
    """Build a display label from model name + reasoning config."""
    if not reasoning:
        return model
    parts = [f"{k}={v}" for k, v in sorted(reasoning.items())]
    return f"{model} ({', '.join(parts)})"


def extract_results(eval_data: dict) -> list[dict]:
    """Extract per-game results from an eval file."""
    header = eval_data["header"]
    model = header["eval"]["model"]
    reasoning = extract_reasoning_config(header)
    run_label = make_run_label(model, reasoning)
    rows = []
    for sample in eval_data["samples"]:
        scorer_key = next(iter(sample["scores"]))
        score_data = sample["scores"][scorer_key]
        meta = score_data.get("metadata", {})
        rows.append(
            {
                "model": model,
                "reasoning": reasoning,
                "run_label": run_label,
                "sample_id": sample["id"],
                "epoch": sample["epoch"],
                "stockfish_level": meta.get("stockfish_level", 0),
                "llm_color": meta.get("llm_color", "?"),
                "result": score_data.get("answer", "?"),
                "score": score_data.get("value", 0),
                "total_moves": meta.get("total_moves", 0),
                "invalid_moves": meta.get("invalid_moves", 0),
                "termination": meta.get("termination", "?"),
                "pgn": score_data.get("explanation", ""),
            }
        )
    return rows


def print_summary(all_rows: list[dict]) -> None:
    """Print a summary table of results across all runs."""
    run_labels = sorted(set(r["run_label"] for r in all_rows))

    for run_label in run_labels:
        model_rows = [r for r in all_rows if r["run_label"] == run_label]
        reasoning = model_rows[0]["reasoning"]
        print("=" * 78)
        print(f"Model: {model_rows[0]['model']}")
        if reasoning:
            print(f"Reasoning: {', '.join(f'{k}={v}' for k, v in sorted(reasoning.items()))}")
        print(f"Total games: {len(model_rows)}")
        print("=" * 78)

        # ---- Overall stats ----
        wins = sum(1 for r in model_rows if r["score"] == 1.0)
        draws = sum(1 for r in model_rows if r["score"] == 0.5)
        losses = sum(1 for r in model_rows if r["score"] == 0.0)
        avg_moves = sum(r["total_moves"] for r in model_rows) / len(model_rows)
        avg_invalid = sum(r["invalid_moves"] for r in model_rows) / len(model_rows)

        print(f"\n  Overall: {wins}W / {draws}D / {losses}L "
              f"(win rate {wins/len(model_rows)*100:.0f}%)")
        print(f"  Avg game length: {avg_moves:.1f} moves")
        print(f"  Avg invalid moves: {avg_invalid:.1f}")

        # ---- Game length vs Stockfish level (the key table) ----
        levels = sorted(set(r["stockfish_level"] for r in model_rows))
        colors = ["white", "black"]

        print(f"\n  {'':>8}  ", end="")
        for level in levels:
            print(f"{'Level ' + str(level):^20}", end="")
        print()
        print(f"  {'':>8}  ", end="")
        for level in levels:
            print(f"{'Moves':>7} {'W/D/L':>6} {'Inv':>5}", end="  ")
        print()
        print("  " + "-" * (10 + 20 * len(levels)))

        for color in colors:
            print(f"  {color:>8}  ", end="")
            for level in levels:
                subset = [
                    r for r in model_rows
                    if r["stockfish_level"] == level and r["llm_color"] == color
                ]
                if not subset:
                    print(f"{'--':>7} {'--':>6} {'--':>5}", end="  ")
                    continue
                m = sum(r["total_moves"] for r in subset) / len(subset)
                w = sum(1 for r in subset if r["score"] == 1.0)
                d = sum(1 for r in subset if r["score"] == 0.5)
                lo = sum(1 for r in subset if r["score"] == 0.0)
                iv = sum(r["invalid_moves"] for r in subset) / len(subset)
                print(f"{m:>7.1f} {w}/{d}/{lo:>3} {iv:>5.1f}", end="  ")
            print()

        # ---- Combined (both colors) by level ----
        print(f"  {'both':>8}  ", end="")
        for level in levels:
            subset = [r for r in model_rows if r["stockfish_level"] == level]
            if not subset:
                print(f"{'--':>7} {'--':>6} {'--':>5}", end="  ")
                continue
            m = sum(r["total_moves"] for r in subset) / len(subset)
            w = sum(1 for r in subset if r["score"] == 1.0)
            d = sum(1 for r in subset if r["score"] == 0.5)
            lo = sum(1 for r in subset if r["score"] == 0.0)
            iv = sum(r["invalid_moves"] for r in subset) / len(subset)
            print(f"{m:>7.1f} {w}/{d}/{lo:>3} {iv:>5.1f}", end="  ")
        print()

        # ---- Individual game details ----
        print(f"\n  Game details:")
        print(f"  {'Level':>5} {'Color':>6} {'Ep':>3} {'Moves':>5} "
              f"{'Inv':>4} {'Result':>7} {'Term':>20} PGN (first 60 chars)")
        print("  " + "-" * 120)
        for r in sorted(model_rows, key=lambda x: (x["stockfish_level"], x["llm_color"], x["epoch"])):
            pgn_short = r["pgn"][:60] + ("..." if len(r["pgn"]) > 60 else "")
            print(
                f"  {r['stockfish_level']:>5} {r['llm_color']:>6} {r['epoch']:>3} "
                f"{r['total_moves']:>5} {r['invalid_moves']:>4} {r['result']:>7} "
                f"{r['termination']:>20} {pgn_short}"
            )

        print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <eval_file> [eval_file ...]")
        print("  Accepts one or more .eval files (Inspect log archives).")
        sys.exit(1)

    all_rows: list[dict] = []
    for path in sys.argv[1:]:
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping.", file=sys.stderr)
            continue
        try:
            eval_data = load_eval(path)
            rows = extract_results(eval_data)
            all_rows.extend(rows)
            print(f"Loaded {len(rows)} games from {Path(path).name}")
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)

    if not all_rows:
        print("No results to analyze.")
        sys.exit(1)

    print()
    print_summary(all_rows)


if __name__ == "__main__":
    main()
