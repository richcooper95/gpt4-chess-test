#!/usr/bin/env python3
"""Estimate the effective Elo of Stockfish at Skill Level N with a given
time-per-move by playing it against reference engines at known UCI_Elo ratings.

Uses maximum-likelihood estimation with the standard logistic Elo model to
produce a point estimate and 95 % confidence interval.

Reference engine time controls
-------------------------------
Stockfish's UCI_Elo was calibrated at TC 120+1.0s (~4 s/move average).
At shorter time controls the reference engine undershoots its target:

    TC 120+1.2  ->  +31 Elo over target
    TC 60+0.6   ->   +7 Elo over target
    TC 30       ->  -14 Elo
    TC 10+0.1   ->  -85 Elo           (source: Stockfish PR #2225)

The default --ref-time of 2 s/move approximates TC 60+0.6 conditions.
Increase to 4 s for closer match to the latest calibration (TC 120+1.0).

Examples
--------
    # Recommended: ~2 h, SE approx 20 Elo
    python calibrate_stockfish_elo.py --games-per-matchup 50 --ref-time 2

    # High-precision: ~4 h, SE approx 15 Elo
    python calibrate_stockfish_elo.py --games-per-matchup 100 --ref-time 4

    # Quick estimate: ~20 min, SE approx 40 Elo
    python calibrate_stockfish_elo.py --games-per-matchup 20 --ref-time 2

The minimum UCI_Elo Stockfish supports is 1320.  If the test configuration is
weaker than 1320, we can still estimate its rating via extrapolation of the
logistic model (the MLE does not require the score to be near 50 %).
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine


@dataclass
class MatchResult:
    ref_elo: int
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.total if self.total else 0.0


def play_game(
    test_engine: chess.engine.SimpleEngine,
    ref_engine: chess.engine.SimpleEngine,
    test_is_white: bool,
    test_limit: chess.engine.Limit,
    ref_limit: chess.engine.Limit,
    max_moves: int = 300,
) -> float:
    """Play one game.  Returns score from *test* engine's perspective."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > max_moves:
            return 0.5
        is_test_turn = (board.turn == chess.WHITE) == test_is_white
        eng = test_engine if is_test_turn else ref_engine
        lim = test_limit if is_test_turn else ref_limit
        result = eng.play(board, lim)
        if result.move is None:
            break
        board.push(result.move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    won_as_white = outcome.winner == chess.WHITE
    return 1.0 if won_as_white == test_is_white else 0.0


def expected_score(elo_test: float, elo_ref: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_ref - elo_test) / 400.0))


def estimate_elo(
    game_results: list[tuple[int, float]],
) -> tuple[float, float]:
    """MLE Elo estimate from (ref_elo, score) pairs.

    Finds elo_hat such that  sum(scores) == sum(expected_scores).
    Returns (elo_hat, standard_error).
    """
    if not game_results:
        return 0.0, float("inf")

    total_score = sum(s for _, s in game_results)

    lo, hi = -500.0, 4000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        exp_total = sum(expected_score(mid, r) for r, _ in game_results)
        if exp_total < total_score:
            lo = mid
        else:
            hi = mid

    elo_hat = (lo + hi) / 2.0

    k = math.log(10) / 400.0
    fisher = sum(
        expected_score(elo_hat, r) * (1 - expected_score(elo_hat, r)) * k ** 2
        for r, _ in game_results
    )
    se = 1.0 / math.sqrt(fisher) if fisher > 0 else float("inf")
    return elo_hat, se


def implied_elo(score: float, ref_elo: int) -> str:
    if score <= 0.0:
        return f"<< {ref_elo}"
    if score >= 1.0:
        return f">> {ref_elo}"
    return f"{ref_elo - 400 * math.log10((1 - score) / score):.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Stockfish Skill Level Elo at a given time/move",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stockfish-path",
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--skill-level",
        type=int,
        default=1,
        help="Skill Level for the engine under test",
    )
    parser.add_argument(
        "--test-time",
        type=float,
        default=0.1,
        help="Seconds per move for the test engine",
    )
    parser.add_argument(
        "--ref-time",
        type=float,
        default=2.0,
        help=(
            "Seconds per move for the reference engine.  UCI_Elo was "
            "calibrated at TC 120+1.0 (~4 s/move); 2 s/move approximates "
            "TC 60+0.6 which is within +7 Elo of target.  At 0.2 s/move "
            "the reference engine undershoots by ~85 Elo."
        ),
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=50,
        help="Total games per reference Elo point (half as white, half as black)",
    )
    parser.add_argument(
        "--elo-min",
        type=int,
        default=1320,
        help="Lowest reference Elo (UCI_Elo minimum is 1320)",
    )
    parser.add_argument(
        "--elo-max",
        type=int,
        default=1600,
        help="Highest reference Elo",
    )
    parser.add_argument(
        "--elo-step",
        type=int,
        default=50,
        help="Step between reference Elo points",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_results.json",
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    ref_elos = list(range(args.elo_min, args.elo_max + 1, args.elo_step))
    half = args.games_per_matchup // 2
    total_games = len(ref_elos) * args.games_per_matchup

    print("=" * 64)
    print("  Stockfish Elo Calibration")
    print("=" * 64)
    print(f"  Test config : Skill Level {args.skill_level}, {args.test_time}s/move")
    print(f"  Ref  config : UCI_LimitStrength, {args.ref_time}s/move")
    print(f"  Ref Elos    : {ref_elos}")
    print(f"  Games/match : {args.games_per_matchup}  ({half}W + {half}B)")
    print(f"  Total games : {total_games}")
    print("=" * 64)
    print()

    test_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    test_engine.configure({"Skill Level": args.skill_level})
    test_limit = chess.engine.Limit(time=args.test_time)

    all_game_results: list[tuple[int, float]] = []
    match_results: list[MatchResult] = []
    t0 = time.time()

    for ref_elo in ref_elos:
        ref_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        ref_engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": ref_elo}
        )
        ref_limit = chess.engine.Limit(time=args.ref_time)

        mr = MatchResult(ref_elo=ref_elo)

        for game_num in range(args.games_per_matchup):
            test_is_white = game_num < half
            score = play_game(
                test_engine, ref_engine, test_is_white, test_limit, ref_limit
            )
            all_game_results.append((ref_elo, score))
            if score == 1.0:
                mr.wins += 1
            elif score == 0.0:
                mr.losses += 1
            else:
                mr.draws += 1

            done = len(all_game_results)
            elapsed = time.time() - t0
            eta = elapsed / done * (total_games - done)
            print(
                f"\r  vs {ref_elo}: game {game_num + 1:>3}/{args.games_per_matchup}"
                f"  +{mr.wins} ={mr.draws} -{mr.losses}"
                f"  [{done}/{total_games}  ETA {eta / 60:.1f}m]",
                end="",
                flush=True,
            )

        ref_engine.quit()
        match_results.append(mr)

        elo_est, se = estimate_elo(all_game_results)
        print(
            f"\n  => vs {ref_elo}: +{mr.wins} ={mr.draws} -{mr.losses}"
            f"  (score {mr.score:.1%})"
            f"  |  running MLE: {elo_est:.0f} +/- {se:.0f}\n"
        )

    test_engine.quit()

    elo_est, se = estimate_elo(all_game_results)
    ci95 = 1.96 * se

    print()
    print("=" * 64)
    print("  FINAL RESULTS")
    print("=" * 64)
    print()
    print(
        f"  {'Ref Elo':>8}  {'W':>4}  {'D':>4}  {'L':>4}"
        f"  {'Score':>7}  {'Implied Elo':>12}"
    )
    print("  " + "-" * 52)
    for mr in match_results:
        print(
            f"  {mr.ref_elo:>8}  {mr.wins:>4}  {mr.draws:>4}  {mr.losses:>4}"
            f"  {mr.score:>6.1%}  {implied_elo(mr.score, mr.ref_elo):>12}"
        )

    print()
    print(f"  MLE Elo estimate : {elo_est:.0f} +/- {se:.0f}")
    print(f"  95% CI           : [{elo_est - ci95:.0f}, {elo_est + ci95:.0f}]")
    total_elapsed = time.time() - t0
    print(f"  Total games      : {len(all_game_results)}")
    print(f"  Wall time        : {total_elapsed / 60:.1f} min")
    print()

    out = {
        "test_config": {
            "skill_level": args.skill_level,
            "time_per_move": args.test_time,
        },
        "ref_config": {
            "time_per_move": args.ref_time,
        },
        "matchups": [
            {
                "ref_elo": mr.ref_elo,
                "wins": mr.wins,
                "draws": mr.draws,
                "losses": mr.losses,
                "score": round(mr.score, 4),
                "implied_elo": implied_elo(mr.score, mr.ref_elo),
            }
            for mr in match_results
        ],
        "mle_elo": round(elo_est, 1),
        "se": round(se, 1),
        "ci95_lo": round(elo_est - ci95, 1),
        "ci95_hi": round(elo_est + ci95, 1),
        "total_games": len(all_game_results),
        "wall_seconds": round(total_elapsed, 1),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  Results saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
