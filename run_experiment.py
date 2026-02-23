#!/usr/bin/env python3
"""Run the scaffolding ablation experiment with parallel execution.

Uses Inspect's eval() API to run all (scaffold_level x model) combinations
concurrently in a single eval() call per epoch.  After each call, eval files
are moved from a temporary flat directory into the nested structure:

    experiment_logs/sf_{N}/level_{M}/epoch_{E}/<file>.eval

Usage:
    # All levels, one model, 1 epoch
    python run_experiment.py \\
        --models openrouter/anthropic/claude-sonnet-4-20250514

    # Multiple models in parallel, 3 new epochs, higher concurrency
    python run_experiment.py \\
        --models openrouter/anthropic/claude-sonnet-4-20250514 openrouter/openai/gpt-4o \\
        --epochs 3 --max-tasks 8

    # Specific scaffold levels only
    python run_experiment.py \\
        --models openrouter/anthropic/claude-sonnet-4-20250514 \\
        --levels 0 1 2 3

    # Text-only models (skip Board Image level, disable images at levels 5-8)
    python run_experiment.py \\
        --models openrouter/meta-llama/llama-4-maverick \\
        --no-image
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
import json

from inspect_ai import eval as inspect_eval

from main_experiment import LEVEL_LABELS, chess_eval_experiment


def count_existing_epochs(level_dir: str) -> int:
    """Count epoch_* subdirectories under *level_dir*."""
    if not os.path.isdir(level_dir):
        return 0
    return sum(
        1
        for d in os.listdir(level_dir)
        if d.startswith("epoch_") and os.path.isdir(os.path.join(level_dir, d))
    )


def min_existing_epochs(base_dir: str, sf_levels: list[int], levels: list[int]) -> int:
    """Return the minimum epoch count across all (sf_level, scaffold_level) cells."""
    counts = []
    for sf in sf_levels:
        for lv in levels:
            level_dir = os.path.join(base_dir, f"sf_{sf}", f"level_{lv}")
            counts.append(count_existing_epochs(level_dir))
    return min(counts) if counts else 0


def parse_task_config(log_path: str) -> tuple[list[int], int]:
    """Extract (stockfish_levels, scaffold_level) from an eval file's header.

    Reads the task name set in main_experiment.py, e.g.
    ``chess_sf1_scaffold3``, and parses the sf and scaffold integers.
    Falls back to reading sample metadata if the name is absent.
    """
    with zipfile.ZipFile(log_path) as z:
        header = json.loads(z.read("header.json"))

    task_name: str = header.get("eval", {}).get("task", "")
    m = re.match(r"chess_sf([\d_]+)_scaffold(\d+)", task_name)
    if m:
        sf_levels = [int(x) for x in m.group(1).split("_")]
        scaffold = int(m.group(2))
        return sf_levels, scaffold

    # Fallback: read from first sample's metadata
    with zipfile.ZipFile(log_path) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                sample = json.loads(z.read(name))
                scorer_key = next(iter(sample["scores"]))
                meta = sample["scores"][scorer_key].get("metadata", {})
                return (
                    [meta.get("stockfish_level", 1)],
                    meta.get("scaffold_level", 0),
                )

    return [1], 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the scaffolding ablation experiment (parallel).",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Models to evaluate (OpenRouter format, space-separated).",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=list(range(9)),
        help="Scaffold levels to run (default: 0-8).",
    )
    parser.add_argument(
        "--sf-levels", type=int, nargs="+", default=[1],
        help="Stockfish skill levels (default: [1]).",
    )
    parser.add_argument(
        "--sf-path", default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of NEW epochs to add per configuration.",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Max concurrent eval tasks (default: num_models x num_levels).",
    )
    parser.add_argument(
        "--base-dir", default="experiment_logs",
        help="Root directory for experiment logs.",
    )
    parser.add_argument(
        "--no-image", action="store_true",
        help="Skip level 4 (Board Image) and disable images for levels 5-8. "
             "Use for text-only models that don't support image input.",
    )
    args = parser.parse_args()

    if args.no_image:
        args.levels = [lv for lv in args.levels if lv != 4]

    n_configs = len(args.sf_levels) * len(args.levels)
    n_models = len(args.models)
    max_tasks = args.max_tasks or (n_configs * n_models)

    total_evals_per_epoch = n_configs * n_models
    total_games_per_epoch = total_evals_per_epoch * 2  # white + black

    print("=" * 60)
    print("  Scaffolding Ablation Experiment (parallel)")
    print("=" * 60)
    print(f"  Models ({n_models}): {', '.join(args.models)}")
    print(f"  Scaffold levels: {args.levels}")
    print(f"  Stockfish levels: {args.sf_levels}")
    print(f"  New epochs to add: {args.epochs}")
    print(f"  Max concurrent tasks: {max_tasks}")
    print(f"  Evals per epoch: {total_evals_per_epoch}")
    print(f"  Games per epoch: {total_games_per_epoch}")
    print("=" * 60)

    # Determine the next epoch number (minimum across all cells)
    existing_epochs = min_existing_epochs(
        args.base_dir, args.sf_levels, args.levels
    )
    print(f"  Existing epochs (min across cells): {existing_epochs}")
    print()

    # Build all task objects
    tasks = [
        chess_eval_experiment(
            scaffold_level=lv,
            stockfish_levels=[sf],
            stockfish_path=args.sf_path,
            no_image=args.no_image,
        )
        for sf in args.sf_levels
        for lv in args.levels
    ]

    for ep_offset in range(args.epochs):
        epoch_num = existing_epochs + ep_offset + 1
        print(f"--- Epoch {epoch_num} ({total_evals_per_epoch} evals, "
              f"{total_games_per_epoch} games, max_tasks={max_tasks}) ---")

        with tempfile.TemporaryDirectory(prefix="chess_eval_") as tmpdir:
            logs = inspect_eval(
                tasks=tasks,
                model=args.models,
                log_dir=tmpdir,
                epochs=1,
                max_tasks=max_tasks,
            )

            moved = 0
            for log in logs:
                src = log.location
                if not src or not os.path.isfile(src):
                    print(f"  WARNING: no file for log (status={log.status})",
                          file=sys.stderr)
                    continue

                sf_levels, scaffold_level = parse_task_config(src)
                sf_tag = sf_levels[0] if sf_levels else 1
                dest_dir = os.path.join(
                    args.base_dir,
                    f"sf_{sf_tag}",
                    f"level_{scaffold_level}",
                    f"epoch_{epoch_num}",
                )
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, os.path.basename(src))
                shutil.move(src, dest)
                moved += 1

                label = LEVEL_LABELS.get(scaffold_level, f"Level {scaffold_level}")
                print(f"  [{moved}/{len(logs)}] sf={sf_tag} level={scaffold_level} "
                      f"({label}) -> {dest_dir}/")

        print(f"  Epoch {epoch_num} complete: {moved} eval files distributed.\n")

    total = args.epochs * total_evals_per_epoch
    print(f"Done. {total} eval runs across {args.epochs} epoch(s).")
    print(f"Analyze with:  python analyze_experiment.py --dir {args.base_dir}")


if __name__ == "__main__":
    main()
