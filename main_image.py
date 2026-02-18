"""
Chess Eval (Image): Test LLMs' chess ability with board images.

Like main_in_context.py, this variant maintains the entire conversation history
so the model can build on its prior reasoning. Additionally, each turn includes
a rendered image of the current board position, oriented from the LLM's
perspective, with the last move highlighted.

Requires vision-capable models (e.g. GPT-4o+, Claude Sonnet/Opus, Gemini).

Usage:
    inspect eval main_image.py --model openrouter/openai/gpt-4o --epochs 3
    inspect eval main_image.py --model openrouter/anthropic/claude-sonnet-4-20250514 --epochs 3

Task parameters:
    -T stockfish_levels=[1,5,10,20]
    -T stockfish_path=/opt/homebrew/bin/stockfish
"""

import base64
import io
import logging
import os
import re
from typing import Optional, Union

import chess
import chess.engine
import chess.pgn
from fentoboardimage import fen_to_image, load_pieces_folder
from PIL import Image, ImageDraw, ImageFont

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
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

Content = list[Union[ContentText, ContentImage]]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIECE_SET = load_pieces_folder(os.path.join(_SCRIPT_DIR, "pieces", "letter"))

SQUARE_LENGTH = 50
LIGHT_COLOR = "#EEEED2"
DARK_COLOR = "#769656"
LAST_MOVE_LIGHT = "#F6F669"
LAST_MOVE_DARK = "#BACA2B"
COORD_MARGIN = 16
LEGEND_HEIGHT = 46


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def moves_from_game(game: chess.pgn.Game) -> str:
    """Extract the move list string from a PGN game."""
    return str(game).split("\n")[-1]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort load of a system font; falls back to Pillow default."""
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_FONT = _load_font(12)
_FONT_SM = _load_font(11)


def render_board_image(
    board: chess.Board,
    llm_color: str,
) -> str:
    """Render the board as a base64 PNG data URL.

    Uses the letter piece set on a chess.com-green board with coordinate
    labels, last-move highlighting, and a legend explaining the color coding.
    The board is oriented from the LLM's perspective.
    """
    flipped = llm_color == "black"

    last_move_arg = None
    if board.move_stack:
        lm = board.peek()
        last_move_arg = {
            "before": chess.square_name(lm.from_square),
            "after": chess.square_name(lm.to_square),
            "darkColor": LAST_MOVE_DARK,
            "lightColor": LAST_MOVE_LIGHT,
        }

    board_img = fen_to_image(
        fen=board.fen(),
        square_length=SQUARE_LENGTH,
        piece_set=_PIECE_SET,
        dark_color=DARK_COLOR,
        light_color=LIGHT_COLOR,
        flipped=flipped,
        last_move=last_move_arg,
    )

    board_w, board_h = board_img.size
    total_w = COORD_MARGIN + board_w + COORD_MARGIN
    total_h = COORD_MARGIN + board_h + COORD_MARGIN
    canvas = Image.new("RGB", (total_w, total_h + LEGEND_HEIGHT), "#FFFFFF")
    canvas.paste(board_img, (COORD_MARGIN, COORD_MARGIN))

    draw = ImageDraw.Draw(canvas)

    files = "hgfedcba" if flipped else "abcdefgh"
    ranks = "12345678" if flipped else "87654321"

    for i, letter in enumerate(files):
        x = COORD_MARGIN + i * SQUARE_LENGTH + SQUARE_LENGTH // 2
        draw.text(
            (x, COORD_MARGIN + board_h + 2),
            letter, fill="#333333", font=_FONT, anchor="mt",
        )

    for i, rank in enumerate(ranks):
        y = COORD_MARGIN + i * SQUARE_LENGTH + SQUARE_LENGTH // 2
        draw.text(
            (COORD_MARGIN - 2, y),
            rank, fill="#333333", font=_FONT, anchor="rm",
        )

    legend_y = total_h + 4
    swatch = 14
    items = [
        (LAST_MOVE_LIGHT, "Last move"),
        ("#333333", "Black pieces (filled)"),
        ("#AAAAAA", "White pieces (outline)"),
    ]
    x_cur = COORD_MARGIN
    for color, label in items:
        draw.rectangle(
            [x_cur, legend_y, x_cur + swatch, legend_y + swatch],
            fill=color, outline="#333333",
        )
        x_cur += swatch + 4
        draw.text((x_cur, legend_y + 1), label, fill="#333333", font=_FONT_SM)
        bbox = draw.textbbox((x_cur, legend_y + 1), label, font=_FONT_SM)
        x_cur = bbox[2] + 12

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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

On each turn you will be shown an image of the current board position \
(oriented from your perspective) and the full move list so far.

The board image uses letters for pieces: K=King, Q=Queen, R=Rook, \
B=Bishop, N=Knight, and a dot for Pawns. Filled/dark symbols are Black \
pieces; outline/light symbols are White pieces. Yellow-highlighted squares \
show the last move. Coordinates (a-h, 1-8) are shown along the edges.

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
- Do not start your move with a "+" symbol.
- The MOVE: line must be the very last line of your response."""


def build_first_turn_content(
    llm_color: str,
    game: chess.pgn.Game,
    board: chess.Board,
) -> Content:
    """Build the user message content for the LLM's first turn."""
    pgn = moves_from_game(game)
    board_data_url = render_board_image(board, llm_color)

    if llm_color == "white":
        text = (
            "The game has begun. You are White and it is your turn to make "
            "the first move.\n\n"
            f"Moves so far: {pgn}\n\n"
            "Analyze the position and make your move."
        )
    else:
        text = (
            f"The game has begun. You are Black.\n\n"
            f"Moves so far: {pgn}\n\n"
            "Analyze the position and make your move."
        )

    return [
        ContentImage(image=board_data_url, detail="auto"),
        ContentText(text=text),
    ]


def build_turn_content(
    opponent_move_san: str,
    game: chess.pgn.Game,
    board: chess.Board,
    llm_color: str,
) -> Content:
    """Build the user message content for a subsequent turn."""
    pgn = moves_from_game(game)
    board_data_url = render_board_image(board, llm_color)

    text = (
        f"Your opponent played: **{opponent_move_san}**\n\n"
        f"Moves so far: {pgn}\n\n"
        "Analyze the position and make your move."
    )

    return [
        ContentImage(image=board_data_url, detail="auto"),
        ContentText(text=text),
    ]


def build_invalid_move_message(
    raw_move: str,
    reason: str,
    game: chess.pgn.Game,
) -> str:
    """Build a text-only message telling the model its move was invalid.

    No board image is needed here -- the image from the current turn is
    already in context.
    """
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
    llm_color: str,
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
                    content=build_invalid_move_message(
                        raw_move, reason, game
                    )
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
def chess_game_image():
    """Play a full chess game with board images and persistent conversation.

    Each turn the model receives a rendered image of the board alongside the
    text move list.  The full conversation history (text only for past turns,
    text+image for the current turn) is sent so the model can maintain
    strategic coherence.
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
                                content=build_first_turn_content(
                                    llm_color, game, board
                                )
                            )
                        )
                        is_first_llm_turn = False

                    move, invalids = await get_llm_move_in_context(
                        model, messages, game, board, llm_color
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
                                content=build_turn_content(
                                    sf_san, game, board, llm_color
                                )
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
                                content=build_first_turn_content(
                                    llm_color, game, board
                                )
                            )
                        )
                        is_first_llm_turn = False
                    else:
                        messages.append(
                            ChatMessageUser(
                                content=build_turn_content(
                                    sf_san, game, board, llm_color
                                )
                            )
                        )

                    move, invalids = await get_llm_move_in_context(
                        model, messages, game, board, llm_color
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
def chess_eval_image(
    stockfish_levels: list[int] = [1, 5, 10, 20],
    stockfish_path: str = "/opt/homebrew/bin/stockfish",
):
    """Evaluate an LLM's chess ability with board images.

    Same eval matrix as chess_eval_in_context (stockfish_level x llm_color x
    epochs), but each turn includes a rendered board image alongside the text
    prompt.  Requires a vision-capable model.
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
        solver=chess_game_image(),
        scorer=chess_game_scorer(),
    )
