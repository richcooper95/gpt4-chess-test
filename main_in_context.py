"""
Chess Eval (In-Context): Test LLMs' chess ability with full conversation history.

Unlike main.py (which sends a standalone prompt each turn), this variant
maintains the entire conversation history so the model can build on its own
prior reasoning. Each turn the model is asked to analyze the position, describe
its strategy, and then provide its move.

Usage:
    inspect eval main_in_context.py --model openrouter/openai/gpt-4o --epochs 3
    inspect eval main_in_context.py --model openrouter/anthropic/claude-sonnet-4-20250514 --epochs 3

Task parameters:
    -T stockfish_levels=[1,5,10,20]
    -T stockfish_path=/opt/homebrew/bin/stockfish
"""

import logging
import re
from typing import Optional

import chess
import chess.engine
import chess.pgn

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    get_model,
)
from inspect_ai.scorer import (
    Score,
    SampleScore,
    Target,
    Value,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, TaskState, solver

logger = logging.getLogger(__name__)

MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def moves_from_game(game: chess.pgn.Game) -> str:
    """Extract the move list string from a PGN game."""
    return str(game).split("\n")[-1]


def get_termination_reason(board: chess.Board, llm_move_failed: bool) -> str:
    """Determine why the game ended."""
    if llm_move_failed:
        return "invalid_move_failure"
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_fifty_moves():
        return "fifty_move_rule"
    if board.is_repetition():
        return "repetition"
    return "unknown"


def build_system_message(llm_color: str, stockfish_level: int) -> str:
    """Build the system message that establishes the persona and format."""
    color_name = llm_color.title()
    return f"""\
You are a chess Grandmaster playing a game of chess. You are playing as \
{color_name} against Stockfish (skill level {stockfish_level}).

On each turn you will be told your opponent's latest move and the full move \
list so far.

Please respond in two parts:

1. **Analysis**: Briefly analyze the current position and describe your \
strategic thinking (2-4 sentences). Consider threats, piece activity, pawn \
structure, and your plan for the next few moves.

2. **Move**: On the LAST line of your response, write your move in exactly \
this format:
   MOVE: <your move in standard algebraic notation>

For example:
   MOVE: e4
   MOVE: Nf3
   MOVE: O-O
   MOVE: Bxc6

Rules:
- Use standard algebraic notation only (e.g. e4, Nf3, O-O, Bxc6+).
- Do not include the move number.
- Do not put your move in quotes.
- The MOVE: line must be the very last line of your response."""


def build_first_turn_message(llm_color: str, game: chess.pgn.Game) -> str:
    """Build the user message for the LLM's first turn."""
    pgn = moves_from_game(game)
    if llm_color == "white":
        return (
            "The game has begun. You are White and it is your turn to make "
            "the first move.\n\n"
            f"Moves so far: {pgn}\n\n"
            "Analyze the position and make your move."
        )
    else:
        return (
            f"The game has begun. You are Black.\n\n"
            f"Moves so far: {pgn}\n\n"
            "Analyze the position and make your move."
        )


def build_turn_message(
    opponent_move_san: str, game: chess.pgn.Game
) -> str:
    """Build the user message for a subsequent turn after the opponent moves."""
    pgn = moves_from_game(game)
    return (
        f"Your opponent played: **{opponent_move_san}**\n\n"
        f"Moves so far: {pgn}\n\n"
        "Analyze the position and make your move."
    )


def build_invalid_move_message(
    raw_move: str, reason: str, game: chess.pgn.Game
) -> str:
    """Build a user message telling the model its move was invalid."""
    pgn = moves_from_game(game)
    return (
        f'Your move "{raw_move}" was invalid ({reason}). '
        "Please reconsider and provide a legal move.\n\n"
        f"Moves so far: {pgn}\n\n"
        "Provide your corrected move (remember: MOVE: <move> on the last line)."
    )


def parse_move_from_response(text: str) -> Optional[str]:
    """Extract the move from a 'MOVE: <move>' line in the response."""
    match = re.search(r"^MOVE:\s*(.+)$", text.strip(), re.MULTILINE)
    if match:
        raw = match.group(1).strip().strip('"').strip("'").lstrip("+")
        return raw
    return None


async def get_llm_move_in_context(
    model: Model,
    messages: list[ChatMessage],
    game: chess.pgn.Game,
    board: chess.Board,
) -> tuple[Optional[chess.Move], int]:
    """Get a move from the LLM using the conversation history.

    Appends user/assistant messages to ``messages`` in place as the
    conversation progresses (including retries).

    Returns:
        A tuple of (move, invalid_attempt_count). move is None if all retries
        were exhausted without obtaining a valid move.
    """
    invalid_attempt_count = 0

    for attempt in range(MAX_RETRIES):
        try:
            response = await model.generate(messages)
            completion = response.completion

            messages.append(ChatMessageAssistant(content=completion))

            raw_move = parse_move_from_response(completion)
            if raw_move is None:
                invalid_attempt_count += 1
                messages.append(
                    ChatMessageUser(
                        content=build_invalid_move_message(
                            "(no MOVE: line found)",
                            "could not find a MOVE: line in your response",
                            game,
                        )
                    )
                )
                continue

            try:
                return board.parse_san(raw_move), invalid_attempt_count
            except chess.InvalidMoveError:
                reason = "syntactically invalid"
            except chess.IllegalMoveError:
                reason = "illegal in the current position"
            except chess.AmbiguousMoveError:
                reason = "ambiguous — please be more specific"

            logger.info(f"Invalid move from LLM: {raw_move} ({reason})")
            invalid_attempt_count += 1
            messages.append(
                ChatMessageUser(
                    content=build_invalid_move_message(raw_move, reason, game)
                )
            )

        except Exception as exc:
            logger.warning(f"Error calling LLM: {exc}")
            invalid_attempt_count += 1

    return None, invalid_attempt_count


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@metric
def mean_game_length() -> Value:
    """Average game length (full moves) across all samples."""

    def compute(scores: list[SampleScore]) -> Value:
        lengths = [
            s.score.metadata["total_moves"]
            for s in scores
            if s.score.metadata and "total_moves" in s.score.metadata
        ]
        return sum(lengths) / len(lengths) if lengths else 0.0

    return compute


@metric
def mean_invalid_moves() -> Value:
    """Average number of invalid move attempts per game."""

    def compute(scores: list[SampleScore]) -> Value:
        counts = [
            s.score.metadata["invalid_moves"]
            for s in scores
            if s.score.metadata and "invalid_moves" in s.score.metadata
        ]
        return sum(counts) / len(counts) if counts else 0.0

    return compute


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[accuracy(), stderr(), mean_game_length(), mean_invalid_moves()])
def chess_game_scorer():
    """Score a chess game based on the outcome relative to the LLM's color."""

    async def score(state: TaskState, target: Target) -> Score:
        result = state.store.get("result", "*")
        llm_color = state.store.get("llm_color", "black")

        llm_win_result = "1-0" if llm_color == "white" else "0-1"

        if result == llm_win_result:
            value = 1.0
        elif result == "1/2-1/2":
            value = 0.5
        else:
            value = 0.0

        return Score(
            value=value,
            answer=result,
            explanation=state.store.get("pgn", ""),
            metadata={
                "total_moves": state.store.get("total_moves", 0),
                "invalid_moves": state.store.get("invalid_moves", 0),
                "termination": state.store.get("termination", "unknown"),
                "stockfish_level": state.store.get("stockfish_level", 0),
                "llm_color": llm_color,
            },
        )

    return score


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def chess_game_in_context():
    """Play a full chess game with persistent conversation history.

    The model sees the entire conversation each turn, including its own prior
    analysis and reasoning, allowing it to maintain strategic coherence.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        stockfish_level: int = state.metadata["stockfish_level"]
        stockfish_path: str = state.metadata["stockfish_path"]
        llm_color: str = state.metadata["llm_color"]
        llm_plays_white = llm_color == "white"

        model = get_model()

        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        total_invalid_moves = 0
        llm_move_failed = False
        is_first_llm_turn = True

        messages: list[ChatMessage] = [
            ChatMessageSystem(
                content=build_system_message(llm_color, stockfish_level)
            ),
        ]

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": stockfish_level})

        try:
            while not board.is_game_over():
                if llm_plays_white:
                    # --- LLM turn (white) ---
                    if is_first_llm_turn:
                        messages.append(
                            ChatMessageUser(
                                content=build_first_turn_message(llm_color, game)
                            )
                        )
                        is_first_llm_turn = False

                    move, invalids = await get_llm_move_in_context(
                        model, messages, game, board
                    )
                    total_invalid_moves += invalids
                    if move is None:
                        llm_move_failed = True
                        break
                    logger.info(f"LLM moves: {board.san(move)}")
                    board.push(move)
                    node = node.add_variation(move)

                    if board.is_game_over():
                        break

                    # --- Stockfish turn (black) ---
                    sf_result = engine.play(board, chess.engine.Limit(time=0.1))
                    assert sf_result.move is not None
                    sf_san = board.san(sf_result.move)
                    logger.info(f"Stockfish moves: {sf_san}")
                    board.push(sf_result.move)
                    node = node.add_variation(sf_result.move)

                    if not board.is_game_over():
                        messages.append(
                            ChatMessageUser(
                                content=build_turn_message(sf_san, game)
                            )
                        )
                else:
                    # --- Stockfish turn (white) ---
                    sf_result = engine.play(board, chess.engine.Limit(time=0.1))
                    assert sf_result.move is not None
                    sf_san = board.san(sf_result.move)
                    logger.info(f"Stockfish moves: {sf_san}")
                    board.push(sf_result.move)
                    node = node.add_variation(sf_result.move)

                    if board.is_game_over():
                        break

                    # --- LLM turn (black) ---
                    if is_first_llm_turn:
                        messages.append(
                            ChatMessageUser(
                                content=build_first_turn_message(llm_color, game)
                            )
                        )
                        is_first_llm_turn = False
                    else:
                        messages.append(
                            ChatMessageUser(
                                content=build_turn_message(sf_san, game)
                            )
                        )

                    move, invalids = await get_llm_move_in_context(
                        model, messages, game, board
                    )
                    total_invalid_moves += invalids
                    if move is None:
                        llm_move_failed = True
                        break
                    logger.info(f"LLM moves: {board.san(move)}")
                    board.push(move)
                    node = node.add_variation(move)
        finally:
            engine.quit()

        # Store results for the scorer.
        result = board.result() if not llm_move_failed else "*"
        game.headers["Result"] = result
        game.headers["White"] = "LLM" if llm_plays_white else "Stockfish"
        game.headers["Black"] = "LLM" if not llm_plays_white else "Stockfish"

        state.store.set("pgn", moves_from_game(game))
        state.store.set("result", result)
        state.store.set("total_moves", board.fullmove_number)
        state.store.set("invalid_moves", total_invalid_moves)
        state.store.set("termination", get_termination_reason(board, llm_move_failed))
        state.store.set("stockfish_level", stockfish_level)
        state.store.set("llm_color", llm_color)

        return state

    return solve


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@task
def chess_eval_in_context(
    stockfish_levels: list[int] = [1, 5, 10, 20],
    stockfish_path: str = "/opt/homebrew/bin/stockfish",
):
    """Evaluate an LLM's chess ability with full conversation history.

    Same eval matrix as chess_eval (stockfish_level x llm_color x epochs),
    but the model maintains its reasoning in context across the full game.
    """
    samples = []
    for level in stockfish_levels:
        for llm_color in ["white", "black"]:
            samples.append(
                Sample(
                    input=f"Play chess as {llm_color.title()} against Stockfish.",
                    target="1-0" if llm_color == "white" else "0-1",
                    id=f"level-{level}-{llm_color}",
                    metadata={
                        "stockfish_level": level,
                        "stockfish_path": stockfish_path,
                        "llm_color": llm_color,
                    },
                )
            )
    return Task(
        dataset=samples,
        solver=chess_game_in_context(),
        scorer=chess_game_scorer(),
    )
