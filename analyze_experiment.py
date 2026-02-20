#!/usr/bin/env python3
"""Analyze scaffolding-ablation experiment results.

Walks experiment_logs/*/sf_*/level_*/epoch_*/ and loads every .eval file,
then produces:
  1. A summary table per (model, scaffold_level).
  2. A matplotlib figure with:
       - Panel 1: Mean game length  vs scaffold level (one line per model)
       - Panel 2: Mean invalid moves vs scaffold level
       - Panel 3: Invalid-move category breakdown (stacked bar, first model)
  3. Saves the figure to scaffold_experiment_results.png.

Usage:
    python3 analyze_experiment.py                          # default dir
    python3 analyze_experiment.py --dir experiment_logs    # explicit
    python3 analyze_experiment.py --no-plot                # table only
"""

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

LEVEL_LABELS = {
    0: "Baseline",
    1: "Accum. Invalid",
    2: "Conv. History",
    3: "FEN String",
    4: "Board Image",
    5: "Structured Resp.",
    6: "Legal Moves",
    7: "Piece Rules",
    8: "Path Tracing",
}


def load_eval(path: str) -> dict:
    with zipfile.ZipFile(path) as z:
        header = json.loads(z.read("header.json"))
        samples = []
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                samples.append(json.loads(z.read(name)))
    return {"header": header, "samples": samples}


def extract_results(eval_data: dict) -> list[dict]:
    header = eval_data["header"]
    model = header["eval"]["model"]
    rows = []
    for sample in eval_data["samples"]:
        scorer_key = next(iter(sample["scores"]))
        score_data = sample["scores"][scorer_key]
        meta = score_data.get("metadata", {})
        rows.append({
            "model": model,
            "scaffold_level": meta.get("scaffold_level", 0),
            "stockfish_level": meta.get("stockfish_level", 0),
            "llm_color": meta.get("llm_color", "?"),
            "result": score_data.get("answer", "?"),
            "score": score_data.get("value", 0),
            "total_moves": meta.get("total_moves", 0),
            "invalid_moves": meta.get("invalid_moves", 0),
            "invalid_move_categories": meta.get(
                "invalid_move_categories", {}
            ),
            "termination": meta.get("termination", "?"),
            "pgn": score_data.get("explanation", ""),
        })
    return rows


def discover_evals(base_dir: str) -> list[str]:
    """Walk experiment_logs tree and return paths to all .eval files."""
    paths = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".eval"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def short_model(model: str) -> str:
    """Shorten an OpenRouter model name for display."""
    return model.replace("openrouter/", "").split("/")[-1]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(all_rows: list[dict]) -> None:
    models = sorted(set(r["model"] for r in all_rows))
    levels = sorted(set(r["scaffold_level"] for r in all_rows))

    for model in models:
        model_rows = [r for r in all_rows if r["model"] == model]
        print("=" * 80)
        print(f"Model: {model}")
        print(f"Games: {len(model_rows)}")
        print("=" * 80)

        header = f"  {'Level':>5}  {'Label':<18}  {'Games':>5}  "
        header += f"{'Moves':>6}  {'Inv':>5}  {'W/D/L':>9}  {'Win%':>5}"
        print(header)
        print("  " + "-" * 74)

        for lv in levels:
            subset = [
                r for r in model_rows if r["scaffold_level"] == lv
            ]
            if not subset:
                continue
            n = len(subset)
            avg_moves = sum(r["total_moves"] for r in subset) / n
            avg_inv = sum(r["invalid_moves"] for r in subset) / n
            w = sum(1 for r in subset if r["score"] == 1.0)
            d = sum(1 for r in subset if r["score"] == 0.5)
            lo = sum(1 for r in subset if r["score"] == 0.0)
            win_pct = w / n * 100
            label = LEVEL_LABELS.get(lv, f"Level {lv}")
            print(
                f"  {lv:>5}  {label:<18}  {n:>5}  "
                f"{avg_moves:>6.1f}  {avg_inv:>5.1f}  "
                f"{w}/{d}/{lo:>3}   {win_pct:>5.1f}%"
            )

        # invalid-move category breakdown
        cats: dict[str, int] = defaultdict(int)
        for r in model_rows:
            for cat, count in r["invalid_move_categories"].items():
                cats[cat] += count
        if cats:
            print(f"\n  Invalid-move categories (all levels combined):")
            for cat, count in sorted(
                cats.items(), key=lambda x: -x[1]
            ):
                print(f"    {cat:<25} {count:>5}")

        # per-game details
        print(f"\n  Game details:")
        print(
            f"  {'Lv':>3} {'SF':>3} {'Col':>6} "
            f"{'Moves':>5} {'Inv':>4} {'Result':>7} "
            f"{'Term':>20} PGN (first 60)"
        )
        print("  " + "-" * 100)
        for r in sorted(
            model_rows,
            key=lambda x: (x["scaffold_level"], x["stockfish_level"],
                           x["llm_color"]),
        ):
            pgn_short = r["pgn"][:60] + (
                "..." if len(r["pgn"]) > 60 else ""
            )
            print(
                f"  {r['scaffold_level']:>3} "
                f"{r['stockfish_level']:>3} "
                f"{r['llm_color']:>6} "
                f"{r['total_moves']:>5} {r['invalid_moves']:>4} "
                f"{r['result']:>7} "
                f"{r['termination']:>20} {pgn_short}"
            )
        print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_rows: list[dict], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print(
            "matplotlib not installed — skipping plot. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax_moves, ax_inv = axes

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for idx, model in enumerate(models):
        model_rows = [r for r in all_rows if r["model"] == model]
        xs, ys_moves, ys_inv = [], [], []
        errs_moves, errs_inv = [], []

        for lv in all_levels:
            subset = [
                r for r in model_rows if r["scaffold_level"] == lv
            ]
            if not subset:
                continue
            n = len(subset)
            moves = [r["total_moves"] for r in subset]
            invs = [r["invalid_moves"] for r in subset]
            avg_m = sum(moves) / n
            avg_i = sum(invs) / n

            import math
            se_m = (
                math.sqrt(sum((m - avg_m) ** 2 for m in moves) / n)
                / math.sqrt(n)
            ) if n > 1 else 0
            se_i = (
                math.sqrt(sum((i - avg_i) ** 2 for i in invs) / n)
                / math.sqrt(n)
            ) if n > 1 else 0

            xs.append(lv)
            ys_moves.append(avg_m)
            ys_inv.append(avg_i)
            errs_moves.append(se_m)
            errs_inv.append(se_i)

        label = short_model(model)
        c = colors[idx % len(colors)]

        ax_moves.errorbar(
            xs, ys_moves, yerr=errs_moves,
            marker="o", capsize=4, label=label, color=c,
        )
        ax_inv.errorbar(
            xs, ys_inv, yerr=errs_inv,
            marker="s", capsize=4, label=label, color=c,
        )

    x_labels = [
        f"{lv}: {LEVEL_LABELS.get(lv, '?')}" for lv in all_levels
    ]

    ax_moves.set_ylabel("Mean Game Length (full moves)")
    ax_moves.set_title("Scaffolding Level vs Game Length")
    ax_moves.legend(fontsize=8)
    ax_moves.grid(True, alpha=0.3)

    ax_inv.set_ylabel("Mean Invalid Moves per Game")
    ax_inv.set_title("Scaffolding Level vs Invalid Moves")
    ax_inv.set_xlabel("Scaffold Level")
    ax_inv.legend(fontsize=8)
    ax_inv.grid(True, alpha=0.3)

    ax_inv.set_xticks(all_levels)
    ax_inv.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax_inv.xaxis.set_major_locator(mticker.FixedLocator(all_levels))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Invalid-move category stacked bar (optional third panel)
# ---------------------------------------------------------------------------


def plot_categories(all_rows: list[dict], output_path: str) -> None:
    """Stacked bar of invalid-move categories per scaffold level."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    models = sorted(set(r["model"] for r in all_rows))
    if not models:
        return

    model = models[0]
    model_rows = [r for r in all_rows if r["model"] == model]
    levels = sorted(set(r["scaffold_level"] for r in model_rows))

    all_cats: set[str] = set()
    for r in model_rows:
        all_cats.update(r["invalid_move_categories"].keys())
    if not all_cats:
        return
    cat_names = sorted(all_cats)

    data: dict[str, list[float]] = {c: [] for c in cat_names}
    for lv in levels:
        subset = [r for r in model_rows if r["scaffold_level"] == lv]
        n = len(subset) or 1
        totals: dict[str, int] = defaultdict(int)
        for r in subset:
            for c, v in r["invalid_move_categories"].items():
                totals[c] += v
        for c in cat_names:
            data[c].append(totals[c] / n)

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = [0.0] * len(levels)
    colors = plt.cm.Set3.colors  # type: ignore[attr-defined]

    for i, cat in enumerate(cat_names):
        vals = data[cat]
        ax.bar(
            levels, vals, bottom=bottom,
            label=cat, color=colors[i % len(colors)],
        )
        bottom = [b + v for b, v in zip(bottom, vals)]

    x_labels = [f"{lv}: {LEVEL_LABELS.get(lv, '?')}" for lv in levels]
    ax.set_xticks(levels)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Invalid Moves per Game")
    ax.set_title(
        f"Invalid-Move Categories by Scaffold Level — {short_model(model)}"
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    cat_path = output_path.replace(".png", "_categories.png")
    fig.savefig(cat_path, dpi=150, bbox_inches="tight")
    print(f"Category plot saved to {cat_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze scaffolding experiment results."
    )
    parser.add_argument(
        "--dir", default="experiment_logs",
        help="Base directory containing experiment logs.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating plots (text summary only).",
    )
    parser.add_argument(
        "--output", default="scaffold_experiment_results.png",
        help="Output path for the main plot.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Directory not found: {args.dir}", file=sys.stderr)
        sys.exit(1)

    eval_paths = discover_evals(args.dir)
    if not eval_paths:
        print(f"No .eval files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    for path in eval_paths:
        try:
            data = load_eval(path)
            rows = extract_results(data)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}", file=sys.stderr)

    print(f"Loaded {len(all_rows)} games from {len(eval_paths)} eval files.\n")

    if not all_rows:
        print("No results to analyze.")
        sys.exit(1)

    print_summary(all_rows)

    if not args.no_plot:
        plot_results(all_rows, args.output)
        plot_categories(all_rows, args.output)


if __name__ == "__main__":
    main()
