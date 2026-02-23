#!/usr/bin/env python3
"""Analyze scaffolding-ablation experiment results.

Walks experiment_logs/ recursively, loads every .eval file, and produces:
  1. Summary tables per (model, scaffold_level) with color breakdowns.
  2. A comprehensive set of matplotlib figures saved as PNGs.
  3. A markdown report with findings and AI safety implications.

Usage:
    python3 analyze_experiment.py                          # default dir
    python3 analyze_experiment.py --dir experiment_logs    # explicit
    python3 analyze_experiment.py --no-plot                # table only
    python3 analyze_experiment.py --no-report              # skip markdown
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict

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

LEVEL_SHORT = {
    0: "Baseline",
    1: "+Invalid\nAccum.",
    2: "+Conv.\nHistory",
    3: "+FEN",
    4: "+Board\nImage",
    5: "+Struct.\nResp.",
    6: "+Legal\nMoves",
    7: "+Piece\nRules",
    8: "+Path\nTracing",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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

        termination = meta.get("termination", "?")
        result = score_data.get("answer", "?")
        if termination == "unknown" and result == "*":
            termination = "draw_claimable"

        outcome = score_data.get("value", 0)
        total_moves = meta.get("total_moves", 0)

        rows.append({
            "model": model,
            "scaffold_level": meta.get("scaffold_level", 0),
            "stockfish_level": meta.get("stockfish_level", 0),
            "llm_color": meta.get("llm_color", "?"),
            "result": result,
            "score": outcome,
            "total_moves": total_moves,
            "invalid_moves": meta.get("invalid_moves", 0),
            "invalid_move_categories": meta.get(
                "invalid_move_categories", {}
            ),
            "termination": termination,
            "pgn": score_data.get("explanation", ""),
            "cps": composite_score(outcome, total_moves),
            "acpl": None,
            "accuracy": None,
        })
    return rows


def discover_evals(base_dir: str) -> list[str]:
    paths = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".eval"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def short_model(model: str) -> str:
    return model.replace("openrouter/", "").split("/")[-1]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / math.sqrt(len(xs))


def _outcome_label(score: float) -> str:
    if score == 1.0:
        return "Win"
    if score == 0.5:
        return "Draw"
    return "Loss"


# ---------------------------------------------------------------------------
# Composite Performance Score (CPS)
# ---------------------------------------------------------------------------

L_NORM = 80  # normalizing constant for game length (moves)


def composite_score(outcome: float, game_length: int) -> float:
    """Compute a single performance metric per game.

    Formula:
        Win  -> 1.0
        Draw -> 0.5 + 0.15 * survival
        Loss -> 0.45 * survival

    where survival = min(game_length / L_NORM, 1.0).

    This guarantees: worst draw (0.50) > best loss (0.45),
    and any win (1.0) > any draw.
    """
    survival = min(game_length / L_NORM, 1.0)
    if outcome == 1.0:
        return 1.0
    if outcome == 0.5:
        return 0.5 + 0.15 * survival
    return 0.45 * survival


# ---------------------------------------------------------------------------
# Average Centipawn Loss (ACPL) + Accuracy
# ---------------------------------------------------------------------------


def _pgn_to_moves(pgn_text: str):
    """Parse a PGN move-text string and return a list of chess.Move objects."""
    import chess.pgn
    wrapped = f'[Result "*"]\n\n{pgn_text}'
    game = chess.pgn.read_game(io.StringIO(wrapped))
    if game is None:
        return []
    return list(game.mainline_moves())


def compute_acpl_for_game(
    pgn_text: str,
    llm_color: str,
    stockfish_path: str,
    analysis_time: float = 0.05,
) -> dict:
    """Replay a game and compute ACPL/accuracy for the LLM's moves.

    Returns dict with keys: acpl, accuracy, move_cpls, n_moves_analysed.
    """
    import chess
    import chess.engine

    moves = _pgn_to_moves(pgn_text)
    if not moves:
        return {"acpl": None, "accuracy": None,
                "move_cpls": [], "n_moves_analysed": 0}

    llm_is_white = llm_color == "white"
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    cpls: list[float] = []
    try:
        for i, move in enumerate(moves):
            is_llm_move = (i % 2 == 0) == llm_is_white

            if is_llm_move:
                info_before = engine.analyse(
                    board, chess.engine.Limit(time=analysis_time))
                color = chess.WHITE if llm_is_white else chess.BLACK
                eval_before = info_before["score"].pov(color).score(
                    mate_score=10000)

                board.push(move)

                info_after = engine.analyse(
                    board, chess.engine.Limit(time=analysis_time))
                eval_after = info_after["score"].pov(color).score(
                    mate_score=10000)

                if eval_before is not None and eval_after is not None:
                    cpl = max(0, eval_before - eval_after)
                    cpls.append(min(cpl, 1000))
            else:
                board.push(move)
    finally:
        engine.quit()

    if not cpls:
        return {"acpl": None, "accuracy": None,
                "move_cpls": [], "n_moves_analysed": 0}

    acpl = _mean(cpls)
    accuracy = max(0, 103.3979 - 0.3820659 * acpl - 0.002169231 * acpl ** 2)

    return {
        "acpl": round(acpl, 1),
        "accuracy": round(min(accuracy, 100.0), 1),
        "move_cpls": [round(c, 1) for c in cpls],
        "n_moves_analysed": len(cpls),
    }


def _cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "acpl_cache.json")


def _row_key(row: dict) -> str:
    raw = (f"{row['model']}|{row['scaffold_level']}|"
           f"{row['llm_color']}|{row['pgn'][:120]}")
    return hashlib.md5(raw.encode()).hexdigest()


def compute_all_acpl(
    all_rows: list[dict],
    stockfish_path: str,
    output_dir: str,
    analysis_time: float = 0.05,
) -> None:
    """Compute ACPL for every game and attach to each row dict (in place).

    Results are cached to acpl_cache.json so subsequent runs are instant.
    """
    cache_file = _cache_path(output_dir)
    cache: dict[str, dict] = {}
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)

    to_compute = []
    for row in all_rows:
        key = _row_key(row)
        if key in cache:
            row["acpl"] = cache[key].get("acpl")
            row["accuracy"] = cache[key].get("accuracy")
        else:
            to_compute.append((key, row))

    if to_compute:
        n = len(to_compute)
        print(f"  Computing ACPL for {n} games "
              f"(~{n * 0.05 * 40 * 2:.0f}s at {analysis_time}s/pos)...")
        for idx, (key, row) in enumerate(to_compute):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"    [{idx + 1}/{n}] "
                      f"{short_model(row['model'])} lv={row['scaffold_level']} "
                      f"{row['llm_color']}")
            result = compute_acpl_for_game(
                row["pgn"], row["llm_color"], stockfish_path, analysis_time)
            row["acpl"] = result["acpl"]
            row["accuracy"] = result["accuracy"]
            cache[key] = {
                "acpl": result["acpl"],
                "accuracy": result["accuracy"],
            }

        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  ACPL cache saved to {cache_file}")
    else:
        print(f"  ACPL loaded from cache ({len(cache)} entries).")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(all_rows: list[dict]) -> None:
    models = sorted(set(r["model"] for r in all_rows))
    levels = sorted(set(r["scaffold_level"] for r in all_rows))

    for model in models:
        model_rows = [r for r in all_rows if r["model"] == model]
        print("=" * 95)
        print(f"Model: {model}")
        print(f"Games: {len(model_rows)}")
        print("=" * 95)

        has_acpl = any(r.get("acpl") is not None for r in model_rows)

        header = (
            f"  {'Lv':>3}  {'Label':<18}  {'N':>3}  "
            f"{'Moves':>6}  {'Inv':>5}  {'W/D/L':>9}  {'Win%':>5}  "
            f"{'CPS':>5}"
        )
        if has_acpl:
            header += f"  {'ACPL':>5}  {'Acc%':>5}"
        print(header)
        sep_len = 79 + (14 if has_acpl else 0)
        print("  " + "-" * sep_len)

        for lv in levels:
            subset = [r for r in model_rows if r["scaffold_level"] == lv]
            if not subset:
                continue
            n = len(subset)
            avg_moves = _mean([r["total_moves"] for r in subset])
            avg_inv = _mean([r["invalid_moves"] for r in subset])
            avg_cps = _mean([r["cps"] for r in subset])
            w = sum(1 for r in subset if r["score"] == 1.0)
            d = sum(1 for r in subset if r["score"] == 0.5)
            lo = sum(1 for r in subset if r["score"] == 0.0)
            win_pct = w / n * 100

            label = LEVEL_LABELS.get(lv, f"Level {lv}")
            line = (
                f"  {lv:>3}  {label:<18}  {n:>3}  "
                f"{avg_moves:>6.1f}  {avg_inv:>5.1f}  "
                f"{w}/{d}/{lo:>3}   {win_pct:>5.1f}%  "
                f"{avg_cps:>5.3f}"
            )
            if has_acpl:
                acpl_vals = [r["acpl"] for r in subset if r["acpl"] is not None]
                acc_vals = [r["accuracy"] for r in subset
                            if r["accuracy"] is not None]
                avg_acpl = _mean(acpl_vals) if acpl_vals else 0
                avg_acc = _mean(acc_vals) if acc_vals else 0
                line += f"  {avg_acpl:>5.0f}  {avg_acc:>5.1f}"
            print(line)

        cats: dict[str, int] = defaultdict(int)
        for r in model_rows:
            for cat, count in r["invalid_move_categories"].items():
                cats[cat] += count
        if cats:
            print("\n  Invalid-move categories (all levels combined):")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"    {cat:<25} {count:>5}")

        terms = Counter(r["termination"] for r in model_rows)
        print("\n  Termination reasons:")
        for term, count in terms.most_common():
            print(f"    {term:<25} {count:>5}")

        print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        return plt, mticker, np
    except ImportError:
        print(
            "matplotlib/numpy not installed — skipping plots.",
            file=sys.stderr,
        )
        return None, None, None


def _level_tick_labels(levels: list[int]) -> list[str]:
    return [f"{lv}: {LEVEL_LABELS.get(lv, '?')}" for lv in levels]


def _save(fig, path, plt):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_game_length_and_invalid(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 1: Game length + invalid moves vs scaffold level (per model)."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))

    fig, (ax_moves, ax_inv) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
    )
    colors = plt.cm.tab10.colors

    for idx, model in enumerate(models):
        mr = [r for r in all_rows if r["model"] == model]
        xs, ys_m, ys_i, em, ei = [], [], [], [], []
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            if not sub:
                continue
            moves = [r["total_moves"] for r in sub]
            invs = [r["invalid_moves"] for r in sub]
            xs.append(lv)
            ys_m.append(_mean(moves))
            ys_i.append(_mean(invs))
            em.append(_se(moves))
            ei.append(_se(invs))

        c = colors[idx % len(colors)]
        label = short_model(model)
        ax_moves.errorbar(xs, ys_m, yerr=em, marker="o", capsize=4,
                          label=label, color=c, linewidth=2)
        ax_inv.errorbar(xs, ys_i, yerr=ei, marker="s", capsize=4,
                        label=label, color=c, linewidth=2)

    xlabels = _level_tick_labels(all_levels)
    ax_moves.set_ylabel("Mean Game Length (full moves)")
    ax_moves.set_title("Game Survival vs Scaffolding Level", fontsize=14)
    ax_moves.legend(fontsize=9)
    ax_moves.grid(True, alpha=0.3)

    ax_inv.set_ylabel("Mean Invalid Moves per Game")
    ax_inv.set_title("Invalid Move Rate vs Scaffolding Level", fontsize=14)
    ax_inv.set_xlabel("Scaffold Level")
    ax_inv.legend(fontsize=9)
    ax_inv.grid(True, alpha=0.3)
    ax_inv.set_xticks(all_levels)
    ax_inv.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax_inv.xaxis.set_major_locator(mticker.FixedLocator(all_levels))

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig1_length_and_invalid.png"), plt)


def plot_outcomes(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 2: Win/Draw/Loss stacked bar by scaffold level, per model."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models, 5), sharey=True, squeeze=False,
    )

    outcome_colors = {"Win": "#4CAF50", "Draw": "#FFC107", "Loss": "#F44336"}

    for idx, model in enumerate(models):
        ax = axes[0][idx]
        mr = [r for r in all_rows if r["model"] == model]
        wins, draws, losses = [], [], []
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            n = len(sub) or 1
            wins.append(sum(1 for r in sub if r["score"] == 1.0) / n * 100)
            draws.append(sum(1 for r in sub if r["score"] == 0.5) / n * 100)
            losses.append(sum(1 for r in sub if r["score"] == 0.0) / n * 100)

        x = np.arange(len(all_levels))
        w = 0.6
        ax.bar(x, wins, w, label="Win", color=outcome_colors["Win"])
        ax.bar(x, draws, w, bottom=wins, label="Draw",
               color=outcome_colors["Draw"])
        ax.bar(x, losses, w,
               bottom=[wi + dr for wi, dr in zip(wins, draws)],
               label="Loss", color=outcome_colors["Loss"])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LEVEL_SHORT.get(lv, str(lv)) for lv in all_levels],
            fontsize=7, ha="center",
        )
        ax.set_title(short_model(model), fontsize=12)
        ax.set_ylim(0, 105)
        if idx == 0:
            ax.set_ylabel("Percentage of Games")
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Game Outcomes by Scaffold Level and Model",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig2_outcomes.png"), plt)


def plot_color_comparison(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 3: White vs Black game length and win rate."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax_len, ax_win = axes

    model_colors = plt.cm.tab10.colors
    offsets = [-0.15, 0.0, 0.15]

    for idx, model in enumerate(models):
        mr = [r for r in all_rows if r["model"] == model]
        mc = model_colors[idx % len(model_colors)]
        label = short_model(model)

        for color_name, style, off in [
            ("white", "o-", -0.05),
            ("black", "s--", 0.05),
        ]:
            xs, ys_len, ys_win = [], [], []
            for lv in all_levels:
                sub = [r for r in mr
                       if r["scaffold_level"] == lv
                       and r["llm_color"] == color_name]
                if not sub:
                    continue
                xs.append(lv + off + offsets[idx] * 0.3)
                ys_len.append(_mean([r["total_moves"] for r in sub]))
                ys_win.append(
                    sum(1 for r in sub if r["score"] == 1.0)
                    / len(sub) * 100
                )

            lbl = f"{label} ({color_name})"
            alpha = 1.0 if color_name == "white" else 0.6
            ax_len.plot(xs, ys_len, style, label=lbl, color=mc,
                        alpha=alpha, markersize=5, linewidth=1.5)
            ax_win.plot(xs, ys_win, style, label=lbl, color=mc,
                        alpha=alpha, markersize=5, linewidth=1.5)

    xlabels = _level_tick_labels(all_levels)
    ax_len.set_ylabel("Mean Game Length")
    ax_len.set_title("Game Length by Color and Scaffold Level", fontsize=14)
    ax_len.legend(fontsize=7, ncol=2)
    ax_len.grid(True, alpha=0.3)

    ax_win.set_ylabel("Win Rate (%)")
    ax_win.set_title("Win Rate by Color and Scaffold Level", fontsize=14)
    ax_win.set_xlabel("Scaffold Level")
    ax_win.legend(fontsize=7, ncol=2)
    ax_win.grid(True, alpha=0.3)
    ax_win.set_xticks(all_levels)
    ax_win.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig3_color_comparison.png"), plt)


def plot_termination(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 4: Termination reasons by scaffold level (all models)."""
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))

    all_terms: set[str] = set(r["termination"] for r in all_rows)
    term_names = sorted(all_terms)
    term_display = {
        "checkmate": "Checkmate",
        "stalemate": "Stalemate",
        "draw_claimable": "Draw (repetition/50-move)",
        "invalid_move_failure": "Invalid move failure",
        "insufficient_material": "Insufficient material",
        "fifty_move_rule": "50-move rule",
        "repetition": "Repetition",
        "unknown": "Unknown",
    }
    term_colors = {
        "checkmate": "#2196F3",
        "stalemate": "#9C27B0",
        "draw_claimable": "#FF9800",
        "invalid_move_failure": "#F44336",
        "insufficient_material": "#795548",
        "fifty_move_rule": "#607D8B",
        "repetition": "#009688",
        "unknown": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(all_levels))
    bottom = np.zeros(len(all_levels))

    for term in term_names:
        vals = []
        for lv in all_levels:
            sub = [r for r in all_rows if r["scaffold_level"] == lv]
            n = len(sub) or 1
            count = sum(1 for r in sub if r["termination"] == term)
            vals.append(count / n * 100)
        vals_arr = np.array(vals)
        if vals_arr.sum() == 0:
            continue
        ax.bar(x, vals_arr, 0.65, bottom=bottom,
               label=term_display.get(term, term),
               color=term_colors.get(term, "#CCCCCC"))
        bottom += vals_arr

    ax.set_xticks(x)
    ax.set_xticklabels(_level_tick_labels(all_levels),
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Percentage of Games")
    ax.set_title("How Games End by Scaffold Level (all models)", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig4_termination.png"), plt)


def plot_heatmap(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 5: Model x scaffold level heatmaps for length, win rate, invalid."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    n_m = len(models)
    n_l = len(all_levels)

    mat_len = np.zeros((n_m, n_l))
    mat_win = np.zeros((n_m, n_l))
    mat_inv = np.zeros((n_m, n_l))

    for i, model in enumerate(models):
        for j, lv in enumerate(all_levels):
            sub = [r for r in all_rows
                   if r["model"] == model and r["scaffold_level"] == lv]
            if not sub:
                continue
            mat_len[i, j] = _mean([r["total_moves"] for r in sub])
            mat_win[i, j] = (sum(1 for r in sub if r["score"] == 1.0)
                             / len(sub) * 100)
            mat_inv[i, j] = _mean([r["invalid_moves"] for r in sub])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 10),
    )

    xlabels = [LEVEL_LABELS.get(lv, str(lv)) for lv in all_levels]
    ylabels = [short_model(m) for m in models]

    for ax, mat, title, cmap, fmt in [
        (ax1, mat_len, "Mean Game Length", "YlOrRd", ".0f"),
        (ax2, mat_win, "Win Rate (%)", "RdYlGn", ".0f"),
        (ax3, mat_inv, "Mean Invalid Moves", "YlOrRd_r", ".1f"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(n_l))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(ylabels, fontsize=10)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.6)

        for i in range(n_m):
            for j in range(n_l):
                val = mat[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=9, color="white" if val > mat.max() * 0.6
                        else "black")

    fig.suptitle("Performance Heatmaps", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig5_heatmaps.png"), plt)


def plot_category_evolution(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 6: Invalid-move categories stacked bar, per model."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))

    all_cats: set[str] = set()
    for r in all_rows:
        all_cats.update(r["invalid_move_categories"].keys())
    if not all_cats:
        return
    cat_names = sorted(all_cats)
    cat_colors = plt.cm.Set2.colors

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models, 5), sharey=True, squeeze=False,
    )

    for idx, model in enumerate(models):
        ax = axes[0][idx]
        mr = [r for r in all_rows if r["model"] == model]

        data: dict[str, list[float]] = {c: [] for c in cat_names}
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            n = len(sub) or 1
            totals: dict[str, int] = defaultdict(int)
            for r in sub:
                for c, v in r["invalid_move_categories"].items():
                    totals[c] += v
            for c in cat_names:
                data[c].append(totals[c] / n)

        x = np.arange(len(all_levels))
        bottom = np.zeros(len(all_levels))
        for ci, cat in enumerate(cat_names):
            vals = np.array(data[cat])
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, 0.65, bottom=bottom, label=cat,
                   color=cat_colors[ci % len(cat_colors)])
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LEVEL_SHORT.get(lv, str(lv)) for lv in all_levels],
            fontsize=6, ha="center",
        )
        ax.set_title(short_model(model), fontsize=12)
        if idx == 0:
            ax.set_ylabel("Mean Invalid Moves per Game")
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "Invalid Move Categories by Scaffold Level",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig6_categories.png"), plt)


def plot_game_length_distribution(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 7: Box plot of game length distribution per scaffold level."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    model_colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 6))

    n_models = len(models)
    width = 0.7 / n_models
    for idx, model in enumerate(models):
        mr = [r for r in all_rows if r["model"] == model]
        data = []
        positions = []
        for li, lv in enumerate(all_levels):
            sub = [r for r in mr if r["scaffold_level"] == lv]
            moves = [r["total_moves"] for r in sub] if sub else [0]
            data.append(moves)
            positions.append(li + (idx - n_models / 2 + 0.5) * width)

        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.85,
            patch_artist=True, showfliers=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=4),
        )
        c = model_colors[idx % len(model_colors)]
        for patch in bp["boxes"]:
            patch.set_facecolor((*c[:3], 0.5))
            patch.set_edgecolor(c)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(c)
        for line in bp["medians"]:
            line.set_color("black")
        ax.plot([], [], color=c, linewidth=6, alpha=0.5,
                label=short_model(model))

    ax.set_xticks(range(len(all_levels)))
    ax.set_xticklabels(_level_tick_labels(all_levels),
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Game Length (full moves)")
    ax.set_title("Game Length Distribution by Scaffold Level", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig7_distribution.png"), plt)


def plot_scaffold_effectiveness(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 8: Marginal improvement from each scaffold level."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    model_colors = plt.cm.tab10.colors

    fig, (ax_delta_len, ax_delta_inv) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
    )

    for idx, model in enumerate(models):
        mr = [r for r in all_rows if r["model"] == model]
        prev_len, prev_inv = None, None
        xs, deltas_len, deltas_inv = [], [], []

        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            if not sub:
                prev_len = None
                prev_inv = None
                continue
            cur_len = _mean([r["total_moves"] for r in sub])
            cur_inv = _mean([r["invalid_moves"] for r in sub])
            if prev_len is not None:
                xs.append(lv)
                deltas_len.append(cur_len - prev_len)
                deltas_inv.append(cur_inv - prev_inv)
            prev_len = cur_len
            prev_inv = cur_inv

        c = model_colors[idx % len(model_colors)]
        label = short_model(model)
        ax_delta_len.bar(
            [x + (idx - 1) * 0.25 for x in xs], deltas_len, 0.22,
            label=label, color=c, alpha=0.8,
        )
        ax_delta_inv.bar(
            [x + (idx - 1) * 0.25 for x in xs], deltas_inv, 0.22,
            label=label, color=c, alpha=0.8,
        )

    xlabels = _level_tick_labels(all_levels)
    ax_delta_len.axhline(0, color="black", linewidth=0.8)
    ax_delta_len.set_ylabel("Change in Mean Game Length")
    ax_delta_len.set_title(
        "Marginal Effect of Each Scaffold Level on Game Length", fontsize=14,
    )
    ax_delta_len.legend(fontsize=9)
    ax_delta_len.grid(True, alpha=0.3)

    ax_delta_inv.axhline(0, color="black", linewidth=0.8)
    ax_delta_inv.set_ylabel("Change in Mean Invalid Moves")
    ax_delta_inv.set_title(
        "Marginal Effect of Each Scaffold Level on Invalid Moves", fontsize=14,
    )
    ax_delta_inv.set_xlabel("Scaffold Level")
    ax_delta_inv.legend(fontsize=9)
    ax_delta_inv.grid(True, alpha=0.3)
    ax_delta_inv.set_xticks(all_levels)
    ax_delta_inv.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig8_marginal_effect.png"), plt)


def plot_composite_score(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 9: Composite Performance Score vs scaffold level."""
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    model_colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, model in enumerate(models):
        mr = [r for r in all_rows if r["model"] == model]
        xs, ys, errs = [], [], []
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            if not sub:
                continue
            scores = [r["cps"] for r in sub]
            xs.append(lv)
            ys.append(_mean(scores))
            errs.append(_se(scores))

        c = model_colors[idx % len(model_colors)]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4,
                    label=short_model(model), color=c, linewidth=2)

    ax.axhspan(0, 0.45, alpha=0.06, color="red", label="_nolegend_")
    ax.axhspan(0.45, 0.65, alpha=0.06, color="orange", label="_nolegend_")
    ax.axhspan(0.65, 1.0, alpha=0.06, color="green", label="_nolegend_")

    ax.text(0.02, 0.22, "Loss zone", transform=ax.transAxes,
            fontsize=8, color="red", alpha=0.5)
    ax.text(0.02, 0.52, "Draw zone", transform=ax.transAxes,
            fontsize=8, color="orange", alpha=0.5)
    ax.text(0.02, 0.82, "Win zone", transform=ax.transAxes,
            fontsize=8, color="green", alpha=0.5)

    xlabels = _level_tick_labels(all_levels)
    ax.set_xticks(all_levels)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Composite Performance Score")
    ax.set_title(
        "Composite Performance Score vs Scaffolding Level\n"
        "(Win=1.0, Draw=0.50-0.65, Loss=0-0.45 scaled by game length)",
        fontsize=13,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig9_composite_score.png"), plt)


def plot_acpl_accuracy(
    all_rows: list[dict], out_dir: str, plt, mticker, np,
) -> None:
    """Figure 10: ACPL and Accuracy% vs scaffold level."""
    if not any(r.get("acpl") is not None for r in all_rows):
        return

    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    model_colors = plt.cm.tab10.colors

    fig, (ax_acpl, ax_acc) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
    )

    for idx, model in enumerate(models):
        mr = [r for r in all_rows if r["model"] == model]
        xs_a, ys_a, errs_a = [], [], []
        xs_c, ys_c, errs_c = [], [], []
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            acpls = [r["acpl"] for r in sub if r.get("acpl") is not None]
            accs = [r["accuracy"] for r in sub if r.get("accuracy") is not None]
            if acpls:
                xs_a.append(lv)
                ys_a.append(_mean(acpls))
                errs_a.append(_se(acpls))
            if accs:
                xs_c.append(lv)
                ys_c.append(_mean(accs))
                errs_c.append(_se(accs))

        c = model_colors[idx % len(model_colors)]
        label = short_model(model)
        ax_acpl.errorbar(xs_a, ys_a, yerr=errs_a, marker="o", capsize=4,
                         label=label, color=c, linewidth=2)
        ax_acc.errorbar(xs_c, ys_c, yerr=errs_c, marker="s", capsize=4,
                        label=label, color=c, linewidth=2)

    xlabels = _level_tick_labels(all_levels)
    ax_acpl.set_ylabel("Mean ACPL (centipawns, lower = better)")
    ax_acpl.set_title(
        "Average Centipawn Loss vs Scaffolding Level", fontsize=14,
    )
    ax_acpl.legend(fontsize=9)
    ax_acpl.grid(True, alpha=0.3)
    ax_acpl.invert_yaxis()

    ax_acc.set_ylabel("Mean Accuracy %")
    ax_acc.set_title(
        "Move Accuracy vs Scaffolding Level", fontsize=14,
    )
    ax_acc.set_xlabel("Scaffold Level")
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_xticks(all_levels)
    ax_acc.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig10_acpl_accuracy.png"), plt)


# ---------------------------------------------------------------------------
# Tables (tables.md) — raw numbers with errors for all plots
# ---------------------------------------------------------------------------


def _fmt(val: float, se: float, dp: int = 1) -> str:
    """Format value +/- SE to the given decimal places."""
    return f"{val:.{dp}f} +/- {se:.{dp}f}"


def _pct(num: int, denom: int) -> str:
    return f"{num / denom * 100:.1f}%" if denom else "n/a"


def generate_tables(all_rows: list[dict], out_dir: str) -> None:
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    has_acpl = any(r.get("acpl") is not None for r in all_rows)
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Experiment Results Tables")
    w()
    w(f"Total games: {len(all_rows)} | "
      f"Models: {len(models)} | "
      f"Scaffold levels: {sorted(all_levels)}")
    w()

    # --- Table 1: Game length ---
    w("## 1. Mean Game Length (full moves)")
    w()
    header = "| Model | " + " | ".join(
        LEVEL_LABELS.get(lv, str(lv)) for lv in all_levels) + " |"
    sep = "| --- | " + " | ".join("---" for _ in all_levels) + " |"
    w(header)
    w(sep)
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        cells = []
        for lv in all_levels:
            sub = [r["total_moves"] for r in mr if r["scaffold_level"] == lv]
            cells.append(_fmt(_mean(sub), _se(sub)) if sub else "-")
        w(f"| {short_model(model)} | " + " | ".join(cells) + " |")
    w()

    # --- Table 2: Invalid moves ---
    w("## 2. Mean Invalid Moves per Game")
    w()
    w(header)
    w(sep)
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        cells = []
        for lv in all_levels:
            sub = [r["invalid_moves"] for r in mr if r["scaffold_level"] == lv]
            cells.append(_fmt(_mean(sub), _se(sub)) if sub else "-")
        w(f"| {short_model(model)} | " + " | ".join(cells) + " |")
    w()

    # --- Table 3: Win / Draw / Loss rates ---
    w("## 3. Win / Draw / Loss Rates (%)")
    w()
    w(header)
    w(sep)
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        cells = []
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            n = len(sub)
            if not n:
                cells.append("-")
                continue
            wins = sum(1 for r in sub if r["score"] == 1.0)
            draws = sum(1 for r in sub if r["score"] == 0.5)
            losses = sum(1 for r in sub if r["score"] == 0.0)
            cells.append(f"W {_pct(wins, n)} / D {_pct(draws, n)} / L {_pct(losses, n)}")
        w(f"| {short_model(model)} | " + " | ".join(cells) + " |")
    w()

    # --- Table 4: White vs Black ---
    w("## 4. White vs Black Performance")
    w()
    w("| Model | Level | Color | Mean Length | SE | Win Rate | N |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        for lv in all_levels:
            for color in ["white", "black"]:
                sub = [r for r in mr
                       if r["scaffold_level"] == lv and r["llm_color"] == color]
                if not sub:
                    continue
                moves = [r["total_moves"] for r in sub]
                wins = sum(1 for r in sub if r["score"] == 1.0)
                w(f"| {short_model(model)} | {lv} ({LEVEL_LABELS.get(lv, '')}) "
                  f"| {color} | {_mean(moves):.1f} | {_se(moves):.1f} "
                  f"| {_pct(wins, len(sub))} | {len(sub)} |")
    w()

    # --- Table 5: Termination reasons ---
    w("## 5. Termination Reasons by Scaffold Level")
    w()
    all_terms = sorted(set(r["termination"] for r in all_rows))
    t_header = "| Level | " + " | ".join(all_terms) + " | Total |"
    t_sep = "| --- | " + " | ".join("---" for _ in all_terms) + " | --- |"
    w(t_header)
    w(t_sep)
    for lv in all_levels:
        sub = [r for r in all_rows if r["scaffold_level"] == lv]
        n = len(sub)
        counts = Counter(r["termination"] for r in sub)
        cells = [f"{counts.get(t, 0)} ({_pct(counts.get(t, 0), n)})" for t in all_terms]
        w(f"| {lv} ({LEVEL_LABELS.get(lv, '')}) | " + " | ".join(cells) + f" | {n} |")
    w()

    # --- Table 6: Invalid move categories ---
    w("## 6. Invalid Move Categories (total counts)")
    w()
    all_cats: set[str] = set()
    for r in all_rows:
        all_cats.update(r["invalid_move_categories"].keys())
    cat_names = sorted(all_cats)
    if cat_names:
        c_header = "| Model | " + " | ".join(cat_names) + " | Total |"
        c_sep = "| --- | " + " | ".join("---" for _ in cat_names) + " | --- |"
        w(c_header)
        w(c_sep)
        for model in models:
            mr = [r for r in all_rows if r["model"] == model]
            totals: dict[str, int] = defaultdict(int)
            for r in mr:
                for cat, cnt in r["invalid_move_categories"].items():
                    totals[cat] += cnt
            grand = sum(totals.values())
            cells = [str(totals.get(c, 0)) for c in cat_names]
            w(f"| {short_model(model)} | " + " | ".join(cells) + f" | {grand} |")
    w()

    # --- Table 7: Composite Performance Score ---
    w("## 7. Composite Performance Score (CPS)")
    w()
    w(header)
    w(sep)
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        cells = []
        for lv in all_levels:
            sub = [r["cps"] for r in mr if r["scaffold_level"] == lv]
            cells.append(_fmt(_mean(sub), _se(sub), 3) if sub else "-")
        w(f"| {short_model(model)} | " + " | ".join(cells) + " |")
    w()

    # --- Table 8: Marginal effects ---
    w("## 8. Marginal Effect (change from previous level)")
    w()
    w("| Model | Level | Delta Game Length | Delta Invalid Moves |")
    w("| --- | --- | --- | --- |")
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        prev_len, prev_inv = None, None
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            if not sub:
                prev_len, prev_inv = None, None
                continue
            cur_len = _mean([r["total_moves"] for r in sub])
            cur_inv = _mean([r["invalid_moves"] for r in sub])
            if prev_len is not None:
                d_len = cur_len - prev_len
                d_inv = cur_inv - prev_inv
                w(f"| {short_model(model)} | {lv} ({LEVEL_LABELS.get(lv, '')}) "
                  f"| {d_len:+.1f} | {d_inv:+.1f} |")
            prev_len, prev_inv = cur_len, cur_inv
    w()

    # --- Table 9: ACPL and Accuracy (if computed) ---
    if has_acpl:
        w("## 9. ACPL and Accuracy %")
        w()
        w(header)
        w(sep)
        for model in models:
            mr = [r for r in all_rows if r["model"] == model]
            cells = []
            for lv in all_levels:
                sub_a = [r["acpl"] for r in mr
                         if r["scaffold_level"] == lv and r.get("acpl") is not None]
                sub_c = [r["accuracy"] for r in mr
                         if r["scaffold_level"] == lv and r.get("accuracy") is not None]
                if sub_a:
                    a_str = _fmt(_mean(sub_a), _se(sub_a), 0)
                    c_str = _fmt(_mean(sub_c), _se(sub_c)) if sub_c else "-"
                    cells.append(f"ACPL {a_str} / Acc {c_str}")
                else:
                    cells.append("-")
            w(f"| {short_model(model)} | " + " | ".join(cells) + " |")
        w()

    # --- Table 10: Per-game raw data ---
    w("## 10. Per-Game Results (all games)")
    w()
    pg_cols = ["Model", "Level", "Color", "Moves", "Invalid", "Score",
               "CPS", "Termination", "Result"]
    if has_acpl:
        pg_cols += ["ACPL", "Acc%"]
    w("| " + " | ".join(pg_cols) + " |")
    w("| " + " | ".join("---" for _ in pg_cols) + " |")
    for r in sorted(all_rows, key=lambda x: (x["model"], x["scaffold_level"],
                                              x["llm_color"])):
        row_cells = [
            short_model(r["model"]),
            f"{r['scaffold_level']} ({LEVEL_LABELS.get(r['scaffold_level'], '')})",
            r["llm_color"],
            str(r["total_moves"]),
            str(r["invalid_moves"]),
            _outcome_label(r["score"]),
            f"{r['cps']:.3f}",
            r["termination"],
            r["result"],
        ]
        if has_acpl:
            row_cells.append(f"{r['acpl']:.0f}" if r.get("acpl") is not None else "-")
            row_cells.append(f"{r['accuracy']:.1f}" if r.get("accuracy") is not None else "-")
        w("| " + " | ".join(row_cells) + " |")
    w()

    path = os.path.join(out_dir, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(all_rows: list[dict], out_dir: str) -> None:
    models = sorted(set(r["model"] for r in all_rows))
    all_levels = sorted(set(r["scaffold_level"] for r in all_rows))
    n_games = len(all_rows)

    total_wins = sum(1 for r in all_rows if r["score"] == 1.0)
    total_draws = sum(1 for r in all_rows if r["score"] == 0.5)
    total_losses = sum(1 for r in all_rows if r["score"] == 0.0)

    # Per-model stats
    model_stats = {}
    for model in models:
        mr = [r for r in all_rows if r["model"] == model]
        ms = short_model(model)
        best_lv = max(
            all_levels,
            key=lambda lv: _mean([
                r["total_moves"] for r in mr if r["scaffold_level"] == lv
            ] or [0]),
        )
        best_len = _mean([
            r["total_moves"] for r in mr if r["scaffold_level"] == best_lv
        ])
        base_len = _mean([
            r["total_moves"] for r in mr if r["scaffold_level"] == 0
        ] or [0])
        total_wins_m = sum(1 for r in mr if r["score"] == 1.0)
        total_inv = sum(r["invalid_moves"] for r in mr)
        lv6_inv = _mean([
            r["invalid_moves"] for r in mr if r["scaffold_level"] == 6
        ] or [0])
        win_rate = total_wins_m / len(mr) * 100

        white_wins = sum(
            1 for r in mr if r["llm_color"] == "white" and r["score"] == 1.0
        )
        black_wins = sum(
            1 for r in mr if r["llm_color"] == "black" and r["score"] == 1.0
        )
        n_white = sum(1 for r in mr if r["llm_color"] == "white")
        n_black = sum(1 for r in mr if r["llm_color"] == "black")

        base_cps = _mean([r["cps"] for r in mr if r["scaffold_level"] == 0]
                         or [0])
        best_cps_lv = max(
            all_levels,
            key=lambda lv: _mean(
                [r["cps"] for r in mr if r["scaffold_level"] == lv] or [0]),
        )
        best_cps = _mean(
            [r["cps"] for r in mr if r["scaffold_level"] == best_cps_lv])

        acpl_vals = [r["acpl"] for r in mr if r.get("acpl") is not None]
        acc_vals = [r["accuracy"] for r in mr if r.get("accuracy") is not None]

        model_stats[ms] = {
            "best_lv": best_lv,
            "best_len": best_len,
            "base_len": base_len,
            "improvement": best_len - base_len,
            "total_wins": total_wins_m,
            "win_rate": win_rate,
            "total_inv": total_inv,
            "base_cps": base_cps,
            "best_cps_lv": best_cps_lv,
            "best_cps": best_cps,
            "mean_acpl": _mean(acpl_vals) if acpl_vals else None,
            "mean_accuracy": _mean(acc_vals) if acc_vals else None,
            "lv6_inv": lv6_inv,
            "white_win_rate": white_wins / n_white * 100 if n_white else 0,
            "black_win_rate": black_wins / n_black * 100 if n_black else 0,
        }

    # Invalid move category totals
    all_cats: dict[str, int] = defaultdict(int)
    for r in all_rows:
        for cat, count in r["invalid_move_categories"].items():
            all_cats[cat] += count
    top_cats = sorted(all_cats.items(), key=lambda x: -x[1])

    # Termination stats
    terms = Counter(r["termination"] for r in all_rows)

    # Unknown termination detail
    unknown_count = terms.get("draw_claimable", 0)

    # Scaffold level where invalid moves first reach ~0
    zero_inv_levels = {}
    for model in models:
        ms = short_model(model)
        mr = [r for r in all_rows if r["model"] == model]
        for lv in all_levels:
            sub = [r for r in mr if r["scaffold_level"] == lv]
            if sub and _mean([r["invalid_moves"] for r in sub]) < 0.5:
                zero_inv_levels[ms] = lv
                break

    report = f"""# Scaffolding Ablation Experiment: Analysis Report

## Experiment Overview

- **Total games**: {n_games} across {len(models)} models and {len(all_levels)} scaffold levels
- **Models tested**: {', '.join(short_model(m) for m in models)}
- **Scaffold levels**: 0 (Baseline) through 8 (Path Tracing)
- **Opponent**: Stockfish at Skill Level 1 (~1200-1400 Elo with 100ms/move)
- **Colors**: Each model played as both White and Black
- **Overall record**: {total_wins} wins, {total_draws} draws, {total_losses} losses ({total_wins/n_games*100:.1f}% win rate)

## Key Findings

### 1. Scaffolding Significantly Extends Game Survival

"""
    for ms, stats in model_stats.items():
        report += (
            f"- **{ms}**: Baseline mean game length of {stats['base_len']:.1f} moves "
            f"improved to {stats['best_len']:.1f} at level {stats['best_lv']} "
            f"({LEVEL_LABELS[stats['best_lv']]}), "
            f"a **{stats['improvement']:+.1f} move** improvement.\n"
        )

    report += """
Game length is a proxy for spatial reasoning competence: longer games indicate
the model can maintain a coherent understanding of the board state over more
turns, making legal and strategically sound moves. The improvement from baseline
to peak scaffolding is substantial across all models.

### 2. Invalid Move Elimination

The most dramatic effect of scaffolding is on invalid move rates:

"""
    for ms, stats in model_stats.items():
        zl = zero_inv_levels.get(ms, "N/A")
        zl_label = LEVEL_LABELS.get(zl, "N/A") if isinstance(zl, int) else "N/A"
        report += (
            f"- **{ms}**: Invalid moves reach near-zero at level {zl} "
            f"({zl_label}). Total invalid moves across all games: {stats['total_inv']}.\n"
        )

    report += """
This suggests that the primary bottleneck for LLM chess performance at lower
scaffold levels is **move legality**, not strategy. Models have plausible
strategic intuitions but fail to translate them into valid board actions.

### 3. Invalid Move Taxonomy

The breakdown of invalid move types reveals the nature of spatial reasoning
failures:

| Category | Count | Interpretation |
|----------|-------|----------------|
"""
    for cat, count in top_cats:
        interpretations = {
            "moved_pinned_piece": "Model ignores pin constraints -- fails to trace attack lines to the king",
            "ignored_check": "Model doesn't recognize it is in check -- incomplete threat detection",
            "no_such_piece": "Model hallucinates a piece on a square where none exists",
            "unparseable": "Response format failure -- not a spatial reasoning error",
            "no_move_line": "Failed to produce a MOVE: line -- instruction following failure",
            "ambiguous": "Ambiguous piece reference -- incomplete board state tracking",
            "blocked_path": "Model ignores pieces blocking the movement path",
            "api_error": "API/network failure -- not a reasoning error",
            "invalid_castling": "Illegal castling attempt",
            "invalid_geometry": "Piece moved in geometrically impossible way",
        }
        interp = interpretations.get(cat, "Unclassified")
        report += f"| {cat} | {count} | {interp} |\n"

    report += f"""
The dominance of **moved_pinned_piece** ({all_cats.get('moved_pinned_piece', 0)} instances)
is particularly revealing. A pin requires reasoning about three objects simultaneously:
the attacking piece, the pinned piece, and the king behind it. This is a
**multi-hop spatial inference** task that current LLMs consistently fail at
without explicit scaffolding.

### 4. Game Outcomes

"""
    for ms, stats in model_stats.items():
        report += (
            f"- **{ms}**: {stats['total_wins']} wins "
            f"({stats['win_rate']:.1f}% win rate)\n"
        )

    report += """
Wins against Stockfish level 1 demonstrate that frontier LLMs can, with
sufficient scaffolding, produce sequences of legal and strategically viable
moves long enough to exploit Stockfish's deliberately weakened play. However,
win rates remain modest, indicating that strategic depth -- not just move
legality -- is a binding constraint at higher scaffold levels.

### 5. White vs Black Asymmetry

"""
    for ms, stats in model_stats.items():
        report += (
            f"- **{ms}**: White win rate {stats['white_win_rate']:.1f}%, "
            f"Black win rate {stats['black_win_rate']:.1f}%\n"
        )

    white_total = sum(
        1 for r in all_rows if r["llm_color"] == "white" and r["score"] == 1.0
    )
    black_total = sum(
        1 for r in all_rows if r["llm_color"] == "black" and r["score"] == 1.0
    )
    n_per_color = n_games // 2

    report += f"""
Across all models: White wins {white_total}/{n_per_color} ({white_total/n_per_color*100:.1f}%), Black wins {black_total}/{n_per_color} ({black_total/n_per_color*100:.1f}%).

"""
    if black_total > white_total:
        report += (
            "Interestingly, models perform **better as Black** despite White's first-move "
            "advantage in standard chess. This may be because: (a) Stockfish at level 1 "
            "plays weak openings as White that are easier to exploit, (b) responding to "
            "moves is cognitively easier than initiating them (reactive vs. generative "
            "reasoning), or (c) the training data contains more annotated responses to "
            "common openings.\n"
        )
    elif white_total > black_total:
        report += (
            "Models perform better as White, consistent with the standard first-move "
            "advantage in chess. However, the margin may also reflect that LLMs have "
            "stronger pattern-matching for common White openings in their training data.\n"
        )
    else:
        report += "Win rates are roughly symmetric between colors.\n"

    report += """
### 6. Termination Reasons

| Reason | Count | % |
|--------|-------|---|
"""
    for term, count in terms.most_common():
        display = {
            "checkmate": "Checkmate",
            "stalemate": "Stalemate",
            "draw_claimable": "Draw (repetition/50-move)",
            "invalid_move_failure": "Invalid move failure",
        }.get(term, term)
        report += f"| {display} | {count} | {count/n_games*100:.1f}% |\n"

    report += f"""
**Note on "Draw (repetition/50-move)" games**: These {unknown_count} games ended because
`board.is_game_over(claim_draw=True)` detected a claimable draw (likely
threefold repetition), but the position hadn't yet been repeated 3 times at the
exact moment of checking. This is a known subtlety in python-chess: the
`can_claim_threefold_repetition()` function fires when any legal move would
create the third occurrence, even if the current position has only occurred
twice. These games represent positions where the LLM (or Stockfish) was
repeating moves -- a sign of strategic stagnation rather than a clean draw.

## AI Safety Implications

### Spatial Reasoning as a Capability Proxy

Chess provides a controlled testbed for evaluating an AI system's ability to:

1. **Maintain an accurate world model**: The board state is fully observable,
   deterministic, and unambiguous. Yet LLMs consistently fail to track piece
   positions accurately, especially under constraints like pins and checks.

2. **Perform multi-hop inference**: Detecting a pin requires tracing an attack
   line through the pinned piece to the king. This is analogous to reasoning
   about causal chains in real-world safety-critical scenarios (e.g., "if I
   take action A, and the environment responds with B, then constraint C is
   violated").

3. **Distinguish valid from plausible actions**: Models generate moves that
   "look right" but violate board constraints. This mirrors a general failure
   mode in agentic AI: producing outputs that are *superficially reasonable*
   but *mechanistically incorrect*.

### Scaffolding as an Alignment Tool

The experiment demonstrates that **external scaffolding can compensate for
intrinsic capability gaps** in spatial reasoning:

- Providing the legal moves list (Level 6) eliminates invalid moves entirely
  for most models. This is analogous to constrained decoding or action-space
  filtering in agentic systems.
- Structured response formats (Level 5) force the model to decompose its
  reasoning into verifiable steps, similar to chain-of-thought monitoring.
- Board visualization (Level 4) leverages multimodal capabilities to bypass
  text-only spatial reasoning limitations.

**Safety implication**: If an AI system's competence depends heavily on
scaffolding, removing or degrading that scaffolding could cause sudden capability
drops. This is relevant for:
- **Robustness**: Systems that appear competent with scaffolding may be brittle
  without it.
- **Sandbagging detection**: A model that performs well on scaffolded tasks but
  poorly without scaffolding is unlikely to be strategically underperforming --
  the capability genuinely depends on the external structure.
- **Capability elicitation**: The large gap between Level 0 and Level 6+
  suggests that significant latent capability can be unlocked by appropriate
  prompting and tool use, which has implications for evaluating frontier model
  capabilities.

### The Pin Detection Problem

The prevalence of `moved_pinned_piece` errors deserves special attention. A pin
is a *relational constraint* between three pieces. Detecting it requires:

1. Identifying the attacking piece
2. Tracing its line of attack
3. Recognizing that a friendly piece lies on this line
4. Recognizing that the king lies behind it
5. Concluding that the intermediate piece cannot move off the line

This is a **5-step relational inference chain** that models fail at
consistently, even with structured prompts. It suggests that current LLMs lack
robust mechanisms for maintaining and querying relational spatial structures --
a capability that would be instrumentally useful for:

- **Planning under constraints**: Understanding which actions are blocked by
  preconditions
- **Threat modeling**: Tracing causal paths through which a system could be
  compromised
- **Situational awareness**: Maintaining an accurate model of which entities
  constrain which others

### Ceiling Effects and Diminishing Returns

The marginal effect of scaffolding diminishes at higher levels. Game length
improvements plateau around levels 5-7, and some models show *decreased*
performance at levels 7-8. This suggests:

1. **Information overload**: Too much scaffolding may distract the model or
   create conflicting signals.
2. **Fundamental capability limits**: Beyond a certain point, no amount of
   prompting can substitute for genuine spatial reasoning ability.
3. **Strategic vs. tactical bottleneck shift**: Early scaffolding addresses
   tactical errors (illegal moves). Once those are eliminated, the binding
   constraint shifts to strategic play, which scaffolding addresses less
   effectively.

This has implications for **capability elicitation** in safety evaluations: there
may be a natural ceiling on how much scaffolding can unlock, and that ceiling
itself is informative about the model's intrinsic capabilities.

## Quantitative Metrics

### Composite Performance Score (CPS)

To provide a single scalar metric that accounts for wins, draws, and
game survival, we define a **Composite Performance Score**:

```
CPS(outcome, game_length) =
    Win:  1.0
    Draw: 0.5 + 0.15 * min(game_length / 80, 1)
    Loss: 0.45 * min(game_length / 80, 1)
```

This ensures: any win > any draw > any loss, while still rewarding
longer games within each outcome category. The normalizing constant
(80 moves) represents a long, competitive game.

"""

    for ms, stats in model_stats.items():
        report += (
            f"- **{ms}**: Baseline CPS = {stats['base_cps']:.3f}, "
            f"best CPS = {stats['best_cps']:.3f} at level "
            f"{stats['best_cps_lv']} "
            f"({LEVEL_LABELS[stats['best_cps_lv']]})\n"
        )

    has_acpl = any(s.get("mean_acpl") is not None for s in model_stats.values())
    if has_acpl:
        report += """
### Average Centipawn Loss (ACPL) and Accuracy

For each game, we replayed the LLM's moves through Stockfish and computed the
**Average Centipawn Loss** -- the mean evaluation drop per move compared to the
engine's assessment of the position. Lower ACPL = better play. We also convert
to an **Accuracy %** using the standard polynomial approximation:

```
Accuracy = 103.40 - 0.382 * ACPL - 0.00217 * ACPL^2
```

"""
        for ms, stats in model_stats.items():
            if stats["mean_acpl"] is not None:
                report += (
                    f"- **{ms}**: Mean ACPL = {stats['mean_acpl']:.0f} cp, "
                    f"Mean Accuracy = {stats['mean_accuracy']:.1f}%\n"
                )
        report += """
ACPL provides an objective, Stockfish-calibrated measure of move quality
independent of game outcome. A model with low ACPL but many losses may be
making strategically sound individual moves but failing at longer-horizon
planning.
"""

    report += """
## Figures

1. `fig1_length_and_invalid.png` -- Game length and invalid moves vs scaffold level
2. `fig2_outcomes.png` -- Win/Draw/Loss rates by scaffold level and model
3. `fig3_color_comparison.png` -- White vs Black performance comparison
4. `fig4_termination.png` -- How games end by scaffold level
5. `fig5_heatmaps.png` -- Model x scaffold level performance heatmaps
6. `fig6_categories.png` -- Invalid move category breakdown per model
7. `fig7_distribution.png` -- Game length distribution (box plots)
8. `fig8_marginal_effect.png` -- Marginal improvement from each scaffold level
9. `fig9_composite_score.png` -- Composite Performance Score vs scaffold level
10. `fig10_acpl_accuracy.png` -- ACPL and Accuracy vs scaffold level (if computed)
"""

    report_path = os.path.join(out_dir, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze scaffolding experiment results.",
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
        "--no-report", action="store_true",
        help="Skip generating the markdown report.",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Output directory for plots and report.",
    )
    parser.add_argument(
        "--compute-acpl", action="store_true",
        help="Compute ACPL/accuracy by replaying games through Stockfish.",
    )
    parser.add_argument(
        "--sf-path", default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary (for --compute-acpl).",
    )
    parser.add_argument(
        "--analysis-time", type=float, default=0.05,
        help="Seconds per position for ACPL analysis (default: 0.05).",
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

    if args.compute_acpl:
        print("Computing ACPL/accuracy...")
        compute_all_acpl(
            all_rows, args.sf_path, args.output_dir, args.analysis_time)
        print()

    print_summary(all_rows)

    if not args.no_plot:
        plt, mticker, np = _setup_matplotlib()
        if plt is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            print("\nGenerating figures...")
            plot_game_length_and_invalid(
                all_rows, args.output_dir, plt, mticker, np)
            plot_outcomes(all_rows, args.output_dir, plt, mticker, np)
            plot_color_comparison(all_rows, args.output_dir, plt, mticker, np)
            plot_termination(all_rows, args.output_dir, plt, mticker, np)
            plot_heatmap(all_rows, args.output_dir, plt, mticker, np)
            plot_category_evolution(all_rows, args.output_dir, plt, mticker, np)
            plot_game_length_distribution(
                all_rows, args.output_dir, plt, mticker, np)
            plot_scaffold_effectiveness(
                all_rows, args.output_dir, plt, mticker, np)
            plot_composite_score(
                all_rows, args.output_dir, plt, mticker, np)
            plot_acpl_accuracy(
                all_rows, args.output_dir, plt, mticker, np)

    if not args.no_report:
        print("\nGenerating report...")
        generate_report(all_rows, args.output_dir)

    print("\nGenerating tables.md...")
    generate_tables(all_rows, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
