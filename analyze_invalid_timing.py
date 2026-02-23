#!/usr/bin/env python3
"""Analyze when invalid moves occur within games.

Walks experiment_logs/ and extracts the move number at which each invalid move
was attempted (conversation mode, levels 2+). Outputs:
  1. Invalid move rate by game phase (opening / midgame / endgame)
  2. Invalid move rate by 5-move bins
  3. First invalid move distribution per model
  4. Per-model breakdown by phase

Usage:
    python3 analyze_invalid_timing.py
    python3 analyze_invalid_timing.py --dir experiment_logs
"""

import argparse
import os
import re
import zipfile
import json
from collections import defaultdict


def _resolve_content(content, attachments: dict) -> str:
    if isinstance(content, str):
        if content.startswith("attachment://"):
            return attachments.get(content.replace("attachment://", ""), content)
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            if text.startswith("attachment://"):
                parts.append(
                    attachments.get(text.replace("attachment://", ""), text)
                )
            elif text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def extract_invalid_timing(base_dir: str):
    """Return (invalid_events, turns_at_move, first_invalid_per_game).

    invalid_events: list of dicts with model, scaffold_level, move_num, color
    turns_at_move: dict[int, int] mapping full-move number to total LLM turns
    first_invalid_per_game: dict[model, list[int]] first invalid move per game
    """
    invalid_events = []
    turns_at_move = defaultdict(int)
    first_invalid_per_game = defaultdict(list)

    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".eval"):
                continue
            path = os.path.join(root, fname)
            try:
                with zipfile.ZipFile(path) as z:
                    header = json.loads(z.read("header.json"))
                    model = (
                        header["eval"]["model"]
                        .replace("openrouter/", "")
                        .split("/")[-1]
                    )
                    for name in z.namelist():
                        if not (
                            name.startswith("samples/")
                            and name.endswith(".json")
                        ):
                            continue
                        sample = json.loads(z.read(name))
                        sk = next(iter(sample["scores"]))
                        meta = sample["scores"][sk].get("metadata", {})
                        sl = meta.get("scaffold_level", 0)
                        if sl < 2:
                            continue
                        color = meta.get("llm_color", "?")
                        total_moves = meta.get("total_moves", 0)
                        inv_count = meta.get("invalid_moves", 0)

                        if color == "white":
                            llm_moves = list(
                                range(1, (total_moves + 2) // 2 + 1)
                            )
                        else:
                            llm_moves = list(
                                range(1, total_moves // 2 + 1)
                            )
                        for fm in llm_moves:
                            turns_at_move[fm] += 1

                        if inv_count == 0:
                            continue

                        attachments = sample.get("attachments", {})
                        events = sample.get("events", [])
                        model_evts = [
                            e for e in events if e.get("event") == "model"
                        ]

                        game_invalid_moves = []
                        for me in model_evts:
                            inp = me.get("input", [])
                            for msg in reversed(inp):
                                if msg.get("role") != "user":
                                    continue
                                content = _resolve_content(
                                    msg.get("content", ""), attachments
                                )
                                if "was invalid" not in content.lower():
                                    break
                                pgn_match = re.search(
                                    r"Moves so far:\s*(.*?)(?:\n|$)",
                                    content,
                                )
                                if not pgn_match:
                                    break
                                pgn = pgn_match.group(1).strip()
                                move_nums = re.findall(r"(\d+)\.", pgn)
                                if move_nums:
                                    full_move = int(move_nums[-1])
                                    invalid_events.append(
                                        {
                                            "model": model,
                                            "scaffold_level": sl,
                                            "move_num": full_move,
                                            "color": color,
                                        }
                                    )
                                    game_invalid_moves.append(full_move)
                                break

                        if game_invalid_moves:
                            first_invalid_per_game[model].append(
                                min(game_invalid_moves)
                            )
            except Exception:
                pass

    return invalid_events, dict(turns_at_move), dict(first_invalid_per_game)


def print_report(invalid_events, turns_at_move, first_invalid_per_game):
    total = len(invalid_events)
    if total == 0:
        print("No invalid move events found (conversation mode, levels 2+).")
        return

    print(f"Invalid move events found (conversation mode, levels 2+): {total}")

    # --- By game phase ---
    phases = [
        ("Opening (moves 1-10)", 1, 10),
        ("Midgame (moves 11-30)", 11, 30),
        ("Endgame (moves 31+)", 31, 200),
    ]

    print("\n=== Invalid Move Rate by Game Phase ===\n")
    print(f"{'Phase':<25} {'Invalid':>8} {'Turns':>8} {'Rate':>8}")
    print("-" * 52)
    for label, lo, hi in phases:
        inv = sum(
            1 for x in invalid_events if lo <= x["move_num"] <= hi
        )
        turns = sum(turns_at_move.get(m, 0) for m in range(lo, hi + 1))
        rate = inv / turns * 100 if turns else 0
        print(f"{label:<25} {inv:>8} {turns:>8} {rate:>7.1f}%")

    # --- By 5-move bins ---
    print("\n=== Invalid Move Rate by 5-Move Bins ===\n")
    print(f"{'Moves':<12} {'Invalid':>8} {'Turns':>8} {'Rate':>8}  {'':>50}")
    print("-" * 90)
    for start in range(1, 51, 5):
        end = start + 4
        inv = sum(
            1 for x in invalid_events if start <= x["move_num"] <= end
        )
        turns = sum(turns_at_move.get(m, 0) for m in range(start, end + 1))
        if turns == 0:
            continue
        rate = inv / turns * 100
        bar = "#" * int(rate * 2)
        print(f"{start:>3}-{end:<3}      {inv:>8} {turns:>8} {rate:>7.1f}%  {bar}")

    # --- By model ---
    models = sorted(set(x["model"] for x in invalid_events))
    print("\n=== Breakdown by Model ===\n")
    for model in models:
        subset = [x for x in invalid_events if x["model"] == model]
        n = len(subset)
        opening = sum(1 for x in subset if x["move_num"] <= 10)
        midgame = sum(1 for x in subset if 11 <= x["move_num"] <= 30)
        endgame = sum(1 for x in subset if x["move_num"] > 30)
        print(
            f"  {model}: "
            f"opening {opening} ({opening / n * 100:.0f}%), "
            f"midgame {midgame} ({midgame / n * 100:.0f}%), "
            f"endgame {endgame} ({endgame / n * 100:.0f}%) "
            f"[n={n}]"
        )

    # --- First invalid move ---
    print("\n=== First Invalid Move per Game ===\n")
    for model in sorted(first_invalid_per_game):
        vals = sorted(first_invalid_per_game[model])
        mean = sum(vals) / len(vals)
        print(
            f"  {model}: mean move {mean:.1f}, "
            f"min={min(vals)}, max={max(vals)}, n={len(vals)}"
        )

    # --- By move number (raw) ---
    by_move = defaultdict(int)
    for x in invalid_events:
        by_move[x["move_num"]] += 1

    print("\n=== Invalid Moves by Move Number ===\n")
    for m in sorted(by_move):
        bar = "#" * by_move[m]
        print(f"  Move {m:3d}: {bar} ({by_move[m]})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze when invalid moves occur within chess games.",
    )
    parser.add_argument(
        "--dir",
        default="experiment_logs",
        help="Base directory containing experiment logs.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Directory not found: {args.dir}")
        return

    invalid_events, turns_at_move, first_invalid = extract_invalid_timing(
        args.dir
    )
    print_report(invalid_events, turns_at_move, first_invalid)


if __name__ == "__main__":
    main()
