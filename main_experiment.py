"""
Chess Eval Scaffolding Experiment: Measure how incremental scaffolding
affects LLM chess ability.

A single parameterized eval with scaffold_level (0-8). Each level adds
one feature on top of all lower levels:

    0  Baseline         Standalone prompt, PGN only, basic retry
    1  Accum. Invalid   + accumulated invalid-move list on retry
    2  Conv. History    + conversation history, Analysis + MOVE format
    3  FEN String       + FEN each turn
    4  Board Image      + rendered board image (vision required)
    5  Structured Resp. + 5-part response (verify/threats/CCT/validate/move)
    6  Legal Moves      + legal-move list each turn
    7  Piece Rules      + piece movement rules in system message
    8  Path Tracing     + explicit square-by-square path enumeration

Usage:
    inspect eval main_experiment.py -T scaffold_level=0 --model openrouter/anthropic/claude-sonnet-4-20250514 --epochs 1
    inspect eval main_experiment.py -T scaffold_level=8 --model openrouter/openai/gpt-4o --epochs 1

Task parameters:
    -T scaffold_level=0          (int, 0-8)
    -T stockfish_levels=[1]      (list[int])
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


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


def _flags(level: int) -> dict[str, bool]:
    return {
        "accumulate_invalid": level >= 1,
        "use_conversation": level >= 2,
        "include_fen": level >= 3,
        "include_image": level >= 4,
        "structured_response": level >= 5,
        "include_legal_moves": level >= 6,
        "include_piece_rules": level >= 7,
        "explicit_path_tracing": level >= 8,
    }


# ---------------------------------------------------------------------------
# Board rendering (levels >= 4)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIECE_SET: object = None

SQUARE_LENGTH = 50
LIGHT_COLOR = "#EEEED2"
DARK_COLOR = "#769656"
LAST_MOVE_LIGHT = "#F6F669"
LAST_MOVE_DARK = "#BACA2B"
COORD_MARGIN = 16
LEGEND_HEIGHT = 46


def _get_piece_set():
    global _PIECE_SET
    if _PIECE_SET is None:
        _PIECE_SET = load_pieces_folder(
            os.path.join(_SCRIPT_DIR, "pieces", "letter")
        )
    return _PIECE_SET


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def render_board_image(board: chess.Board, llm_color: str) -> str:
    """Render the board as a base64 PNG data URL."""
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
        piece_set=_get_piece_set(),
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


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def moves_from_game(game: chess.pgn.Game) -> str:
    return str(game).split("\n")[-1]


def _format_legal_moves(board: chess.Board) -> str:
    return ", ".join(sorted(board.san(m) for m in board.legal_moves))


def get_termination_reason(board: chess.Board, llm_move_failed: bool) -> str:
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


def parse_move_from_response(text: str) -> Optional[str]:
    match = re.search(r"^MOVE:\s*(.+)$", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip().strip('"').strip("'").lstrip("+")
    return None


# ---------------------------------------------------------------------------
# Invalid-move diagnostics
# ---------------------------------------------------------------------------


def _parse_san_parts(san: str) -> Optional[dict]:
    """Best-effort decomposition of a SAN string into piece type + dest."""
    san = san.strip().rstrip("+#")

    if san in ("O-O", "O-O-O", "0-0", "0-0-0"):
        return {"type": "castling"}

    if "=" in san:
        san = san[: san.index("=")]

    if san and san[0] in "KQRBN":
        piece_char = san[0]
        san = san[1:]
    else:
        piece_char = "P"

    san = san.replace("x", "")

    if len(san) < 2:
        return None

    dest_str = san[-2:]
    disambig = san[:-2]

    try:
        dest = chess.parse_square(dest_str)
    except ValueError:
        return None

    piece_map = {
        "K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
        "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN,
    }
    return {
        "type": "normal",
        "piece_type": piece_map[piece_char],
        "dest": dest,
        "disambig": disambig,
    }


def _can_reach_geometrically(
    piece_type: chess.PieceType, from_sq: chess.Square, to_sq: chess.Square
) -> bool:
    """Check whether a piece could *geometrically* reach to_sq (ignoring blockers)."""
    df = abs(chess.square_file(to_sq) - chess.square_file(from_sq))
    dr = abs(chess.square_rank(to_sq) - chess.square_rank(from_sq))

    if piece_type == chess.ROOK:
        return (df == 0) != (dr == 0)
    if piece_type == chess.BISHOP:
        return df == dr and df > 0
    if piece_type == chess.QUEEN:
        return (df == 0) != (dr == 0) or (df == dr and df > 0)
    if piece_type == chess.KNIGHT:
        return sorted([df, dr]) == [1, 2]
    if piece_type == chess.KING:
        return max(df, dr) == 1
    if piece_type == chess.PAWN:
        return True  # pawn geometry is direction-dependent; accept broadly
    return False


def diagnose_invalid_move(
    board: chess.Board, raw_move: str, error_type: str
) -> str:
    """Classify *why* an LLM move was invalid.

    error_type is one of: "no_move_line", "invalid", "illegal", "ambiguous".
    """
    if error_type == "no_move_line":
        return "no_move_line"
    if error_type == "ambiguous":
        return "ambiguous"

    if error_type == "illegal":
        return "ignored_check" if board.is_check() else "moved_pinned_piece"

    parts = _parse_san_parts(raw_move)
    if parts is None:
        return "unparseable"
    if parts["type"] == "castling":
        return "invalid_castling"

    piece_type = parts["piece_type"]
    dest = parts["dest"]

    pieces = list(board.pieces(piece_type, board.turn))
    if not pieces:
        return "no_such_piece"

    disambig = parts["disambig"]
    matching: list[chess.Square] = []
    for sq in pieces:
        name = chess.square_name(sq)
        if not disambig:
            matching.append(sq)
        elif len(disambig) == 1:
            if disambig in "abcdefgh" and name[0] == disambig:
                matching.append(sq)
            elif disambig in "12345678" and name[1] == disambig:
                matching.append(sq)
        elif len(disambig) == 2 and name == disambig:
            matching.append(sq)

    if not matching:
        return "no_such_piece"

    if any(_can_reach_geometrically(piece_type, sq, dest) for sq in matching):
        return "blocked_path"

    return "invalid_geometry"


def _add_category(
    categories: dict[str, int], category: str
) -> None:
    categories[category] = categories.get(category, 0) + 1


# ---------------------------------------------------------------------------
# Prompt building — standalone mode (levels 0-1)
# ---------------------------------------------------------------------------


def _build_standalone_prompt(
    flags: dict[str, bool],
    game: chess.pgn.Game,
    board: chess.Board,
    llm_color: str,
    invalid_moves: set[str],
) -> str:
    color_name = llm_color.title()

    invalid_msg = ""
    if flags["accumulate_invalid"] and invalid_moves:
        invalid_list = "\n ".join(sorted(invalid_moves))
        invalid_msg = (
            "\nNote that the following moves are invalid, and must "
            f"not be returned:\n {invalid_list}\n"
        )

    return f"""\
You are a chess Grandmaster. You are about to be given a list of moves \
in PGN notation, and you will make the best move available to you, using \
all of your knowledge of chess. You can take all the time to think that \
you need. Think deeply - you are a Grandmaster!

You will always play as {color_name}.

You will only reply with your move in standard algebraic notation, \
e.g. "e4" or "Nf3".

If you are being asked to play a move, then there is still a valid move \
you can make. This is always the case, even if you believe the game is over.

Do not include the move number. Do not include any other words in your \
response apart from the move you wish to make. Do not return your move \
in quotes. Do not start your move with a "+" symbol. Return a single \
move for {color_name} only.
{invalid_msg}
Here are the moves so far:

{moves_from_game(game)}

Your move (which will replace the * in the line above):
"""


# ---------------------------------------------------------------------------
# Prompt building — conversation mode (levels >= 2)
# ---------------------------------------------------------------------------


def _build_system_message(
    flags: dict[str, bool], llm_color: str, stockfish_level: int
) -> str:
    color_name = llm_color.title()
    opponent_color = "Black" if llm_color == "white" else "White"
    parts: list[str] = []

    # -- opening --
    parts.append(
        f"You are a chess Grandmaster playing a game of chess. You are "
        f"playing as {color_name} against Stockfish (skill level "
        f"{stockfish_level})."
    )

    # -- what the model receives each turn --
    info = ["The full move list so far (PGN)."]
    if flags["include_fen"]:
        info.append(
            "The FEN string for the current position (an unambiguous "
            "text encoding)."
        )
    if flags["include_image"]:
        info.insert(
            0,
            "An image of the current board position (oriented from "
            "your perspective).",
        )
    if flags["include_legal_moves"]:
        info.append(
            "A complete list of legal moves in the current position."
        )
    parts.append(
        "On each turn you will be shown:\n"
        + "\n".join(f"- {i}" for i in info)
    )

    # -- image description (level >= 4) --
    if flags["include_image"]:
        parts.append(
            "The board image uses letters for pieces: K=King, Q=Queen, "
            "R=Rook, B=Bishop, N=Knight, and a dot for Pawns. Filled/dark "
            "symbols are Black pieces; outline/light symbols are White "
            "pieces. Yellow-highlighted squares show the last move. "
            "Coordinates (a-h, 1-8) are shown along the edges."
        )
        if llm_color == "black":
            parts.append(
                "IMPORTANT: The board is shown from YOUR perspective as "
                "Black. This means rank 1 is at the top and rank 8 is at "
                "the bottom; file h is on the left and file a is on the "
                "right. Keep this orientation in mind when reading piece "
                "positions from the image."
            )
        else:
            parts.append(
                "The board is shown from your perspective as White. Rank 1 "
                "is at the bottom and rank 8 is at the top; file a is on "
                "the left and file h is on the right."
            )

    # -- piece movement rules (level >= 7) --
    if flags["include_piece_rules"]:
        parts.append(
            "Piece movement rules (remember these when checking move "
            "legality):\n"
            "- **Rook**: moves in straight lines along ranks or files. "
            "BLOCKED by any piece (friendly or enemy) in its path.\n"
            "- **Bishop**: moves diagonally only. BLOCKED by any piece "
            "in its path.\n"
            "- **Queen**: moves like a Rook or Bishop (straight lines or "
            "diagonals). BLOCKED by any piece in its path.\n"
            "- **Knight**: moves in an L-shape (2 squares in one direction "
            "+ 1 square perpendicular). The ONLY piece that can jump over "
            "others.\n"
            "- **King**: moves exactly ONE square in any direction. Also "
            "can castle (O-O or O-O-O) if conditions are met.\n"
            "- **Pawn**: moves forward one square (or two from starting "
            "rank). Captures diagonally forward one square. Can promote on "
            "the 8th rank. En passant possible after opponent double-advance."
        )

    # -- response format --
    if flags["structured_response"]:
        resp: list[str] = []
        resp.append(
            f'1. **Position verification**: Cross-reference the board '
            f'image with the FEN string and the move list. Walk through '
            f'where every piece is, reading the coordinates from the '
            f'board edges carefully. For each piece, state its square '
            f'(e.g. "My Knight on f3", "Opponent\'s Bishop on c5"). '
            f'Confirm which pieces are yours ({color_name}) and which '
            f'are your opponent\'s ({opponent_color}). Do not skip this '
            f'step.'
        )
        resp.append(
            "2. **Threat assessment**: For each of your opponent's pieces, "
            "consider what squares it attacks and whether any of your "
            "pieces are under threat. Specifically:\n"
            "   - List pieces that are attacked by opponent pieces.\n"
            "   - List opponent pieces you could capture.\n"
            "   - Check whether the last move created a check, fork, pin, "
            "skewer, or discovered attack.\n"
            "   - Think about what your opponent is planning next."
        )
        resp.append(
            '3. **Analysis and candidate moves**: Plan your move using '
            '"checks, captures, threats" (CCT):\n'
            "   - **Checks**: Can you give check? Is it useful?\n"
            "   - **Captures**: Can you capture an undefended or "
            "higher-value piece?\n"
            "   - **Threats**: Can you attack an undefended piece, "
            "threaten checkmate, or gain a tempo?\n"
            "   Also consider piece activity, pawn structure, king safety, "
            "and development. List 2-3 candidate moves and briefly "
            "evaluate each."
        )

        if flags["explicit_path_tracing"]:
            resp.append(
                "4. **Move validation with explicit path tracing**: "
                "Before committing to your move:\n"
                "   - State the piece type, current square, and "
                "destination square.\n"
                "   - **Rooks/Bishops/Queens**: List EVERY intermediate "
                "square on the path. For each, state whether it is empty "
                "or occupied (and by which piece). If ANY square is "
                "occupied the path is BLOCKED — pick a different move.\n"
                '     Example: "Rook a1→a5: a2 (empty), a3 (empty), '
                'a4 (my Pawn — BLOCKED!)"\n'
                "   - **Knights**: Verify the L-shape. State the 2+1 "
                "pattern and confirm the destination is valid.\n"
                "   - **Kings**: Confirm destination is exactly one "
                "square away (or valid castling).\n"
                "   - **Pawns**: Confirm direction, distance, and whether "
                "capture or advance. If two squares, confirm starting "
                "rank and empty intermediate square.\n"
                "   - Confirm destination is not occupied by your own "
                "piece."
            )
        else:
            resp.append(
                "4. **Move validation**: Before committing to your move, "
                "verify it is legal:\n"
                "   - State the piece type and its current square.\n"
                "   - State the destination square.\n"
                "   - For Rooks/Bishops/Queens: list every square along "
                "the path and confirm each is empty.\n"
                "   - For Knights: confirm valid L-shape from current "
                "square.\n"
                "   - For Kings: confirm exactly one square away (or "
                "valid castling).\n"
                "   - For Pawns: confirm direction, distance, capture "
                "rules.\n"
                "   - Confirm destination not occupied by own piece."
            )

        if flags["include_legal_moves"]:
            resp[-1] += (
                "\n   - **Cross-check: verify your move appears in the "
                "legal moves list.** If not, pick another."
            )

        resp.append(
            "5. **Move**: On the LAST line of your response, write your "
            "chosen move in exactly this format:\n"
            "   MOVE: <your move in standard algebraic notation>"
        )

        parts.append(
            "Please respond in five parts:\n\n" + "\n\n".join(resp)
        )
    else:
        parts.append(
            "Please respond in two parts:\n\n"
            "1. **Analysis**: Briefly analyze the current position and "
            "describe your strategic thinking (2-4 sentences). Consider "
            "threats, piece activity, pawn structure, and your plan.\n\n"
            "2. **Move**: On the LAST line of your response, write your "
            "move in exactly this format:\n"
            "   MOVE: <your move in standard algebraic notation>"
        )

    # -- examples and rules --
    rules = (
        "For example:\n"
        "   MOVE: e4\n"
        "   MOVE: Nf3\n"
        "   MOVE: O-O\n"
        "   MOVE: Bxc6\n\n"
        "Rules:\n"
        "- Use standard algebraic notation only (e.g. e4, Nf3, O-O, "
        "Bxc6+).\n"
        "- Do not include the move number.\n"
        "- Do not put your move in quotes.\n"
        "- Do not start your move with a \"+\" symbol.\n"
        "- The MOVE: line must be the very last line of your response."
    )
    if flags["include_legal_moves"]:
        rules += (
            "\n- Your move MUST be one of the legal moves provided in "
            "the prompt."
        )
    parts.append(rules)

    return "\n\n".join(parts)


def _build_turn_content(
    flags: dict[str, bool],
    game: chess.pgn.Game,
    board: chess.Board,
    llm_color: str,
    *,
    opponent_move_san: Optional[str] = None,
    is_first_turn: bool = False,
) -> str | Content:
    """Build the user message content for a turn in conversation mode."""
    pgn = moves_from_game(game)
    lines: list[str] = []

    if is_first_turn:
        if llm_color == "white":
            lines.append(
                "The game has begun. You are White and it is your turn "
                "to make the first move."
            )
        else:
            lines.append("The game has begun. You are Black.")
    else:
        lines.append(f"Your opponent played: **{opponent_move_san}**")

    lines.append(f"\nMoves so far: {pgn}")

    if flags["include_fen"]:
        lines.append(f"FEN: {board.fen()}")

    if flags["include_legal_moves"]:
        lines.append(f"Legal moves: {_format_legal_moves(board)}")

    reminder = "\nAnalyze the position and make your move."
    if flags["structured_response"]:
        reminder = (
            "\nRemember: first cross-reference the board image with the "
            "move list to verify the position, then analyze, then give "
            "your move."
        )
    if flags["include_legal_moves"]:
        reminder += " Your move MUST be one of the legal moves listed above."
    lines.append(reminder)

    text = "\n".join(lines)

    if flags["include_image"]:
        return [
            ContentImage(image=render_board_image(board, llm_color), detail="auto"),
            ContentText(text=text),
        ]
    return text


def _build_invalid_retry_content(
    flags: dict[str, bool],
    raw_move: str,
    reason: str,
    game: chess.pgn.Game,
    board: chess.Board,
    llm_color: str,
    all_invalid: set[str],
) -> str:
    """Build the retry message after an invalid move (conversation mode).

    Always text-only — the board image from the current turn is already
    in context.
    """
    pgn = moves_from_game(game)
    lines: list[str] = [f'Your move "{raw_move}" was invalid ({reason}).']

    if flags["accumulate_invalid"] and all_invalid:
        formatted = ", ".join(sorted(all_invalid))
        lines.append(
            f"\nThe following moves have already been tried and are "
            f"invalid — do NOT repeat any of them: {formatted}"
        )

    lines.append(f"\nMoves so far: {pgn}")

    if flags["include_fen"]:
        lines.append(f"FEN: {board.fen()}")

    if flags["include_legal_moves"]:
        lines.append(f"Legal moves: {_format_legal_moves(board)}")

    lines.append(
        "\nProvide your corrected move (remember: MOVE: <move> on the "
        "last line)."
    )
    if flags["include_legal_moves"]:
        lines.append(
            "Your move MUST be one of the legal moves listed above."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM move functions
# ---------------------------------------------------------------------------


async def _get_llm_move_standalone(
    model: Model,
    flags: dict[str, bool],
    game: chess.pgn.Game,
    board: chess.Board,
    llm_color: str,
) -> tuple[Optional[chess.Move], int, dict[str, int]]:
    """Standalone mode (levels 0-1): fresh prompt each attempt."""
    invalid_moves: set[str] = set()
    invalid_count = 0
    categories: dict[str, int] = {}

    for _ in range(MAX_RETRIES):
        prompt = _build_standalone_prompt(
            flags, game, board, llm_color, invalid_moves
        )
        try:
            response = await model.generate(prompt)
            raw = response.completion.strip().lstrip("+")

            try:
                return board.parse_san(raw), invalid_count, categories
            except chess.InvalidMoveError:
                cat = diagnose_invalid_move(board, raw, "invalid")
            except chess.IllegalMoveError:
                cat = diagnose_invalid_move(board, raw, "illegal")
            except chess.AmbiguousMoveError:
                cat = "ambiguous"

            logger.info(f"Invalid move from LLM: {raw} ({cat})")
            invalid_count += 1
            _add_category(categories, cat)
            invalid_moves.add(raw)

        except Exception as exc:
            logger.warning(f"Error calling LLM: {exc}")
            invalid_count += 1
            _add_category(categories, "api_error")

    return None, invalid_count, categories


async def _get_llm_move_conversation(
    model: Model,
    messages: list[ChatMessage],
    flags: dict[str, bool],
    game: chess.pgn.Game,
    board: chess.Board,
    llm_color: str,
) -> tuple[Optional[chess.Move], int, dict[str, int]]:
    """Conversation mode (levels >= 2): appends to message history."""
    invalid_count = 0
    categories: dict[str, int] = {}
    tried_invalid: set[str] = set()

    for _ in range(MAX_RETRIES):
        try:
            response = await model.generate(messages)
            completion = response.completion
            messages.append(ChatMessageAssistant(content=completion))

            raw = parse_move_from_response(completion)
            if raw is None:
                invalid_count += 1
                _add_category(categories, "no_move_line")
                messages.append(
                    ChatMessageUser(
                        content=_build_invalid_retry_content(
                            flags,
                            "(no MOVE: line found)",
                            "could not find a MOVE: line in your response",
                            game, board, llm_color, tried_invalid,
                        )
                    )
                )
                continue

            try:
                return board.parse_san(raw), invalid_count, categories
            except chess.InvalidMoveError:
                cat = diagnose_invalid_move(board, raw, "invalid")
                reason = "syntactically invalid"
            except chess.IllegalMoveError:
                cat = diagnose_invalid_move(board, raw, "illegal")
                reason = "illegal in the current position"
            except chess.AmbiguousMoveError:
                cat = "ambiguous"
                reason = "ambiguous — please be more specific"

            logger.info(f"Invalid move from LLM: {raw} ({cat})")
            invalid_count += 1
            _add_category(categories, cat)
            tried_invalid.add(raw)
            messages.append(
                ChatMessageUser(
                    content=_build_invalid_retry_content(
                        flags, raw, reason,
                        game, board, llm_color, tried_invalid,
                    )
                )
            )

        except Exception as exc:
            logger.warning(f"Error calling LLM: {exc}")
            invalid_count += 1
            _add_category(categories, "api_error")

    return None, invalid_count, categories


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@metric
def mean_game_length() -> Value:
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
    async def score(state: TaskState, target: Target) -> Score:
        result = state.store.get("result", "*")
        llm_color = state.store.get("llm_color", "black")
        llm_win = "1-0" if llm_color == "white" else "0-1"

        if result == llm_win:
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
                "invalid_move_categories": state.store.get(
                    "invalid_move_categories", {}
                ),
                "termination": state.store.get("termination", "unknown"),
                "stockfish_level": state.store.get("stockfish_level", 0),
                "llm_color": llm_color,
                "scaffold_level": state.store.get("scaffold_level", 0),
            },
        )
    return score


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def chess_game_experiment():
    """Play a full chess game with configurable scaffolding level."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scaffold_level: int = state.metadata["scaffold_level"]
        stockfish_level: int = state.metadata["stockfish_level"]
        stockfish_path: str = state.metadata["stockfish_path"]
        llm_color: str = state.metadata["llm_color"]
        llm_plays_white = llm_color == "white"

        flags = _flags(scaffold_level)
        model = get_model()

        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        total_invalid_moves = 0
        all_categories: dict[str, int] = {}
        llm_move_failed = False

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": stockfish_level})

        # --- conversation setup (levels >= 2) ---
        messages: list[ChatMessage] = []
        is_first_llm_turn = True
        if flags["use_conversation"]:
            messages.append(
                ChatMessageSystem(
                    content=_build_system_message(
                        flags, llm_color, stockfish_level
                    )
                )
            )

        try:
            while not board.is_game_over(claim_draw=True):
                if llm_plays_white:
                    # --- LLM turn (white) ---
                    if flags["use_conversation"]:
                        if is_first_llm_turn:
                            content = _build_turn_content(
                                flags, game, board, llm_color,
                                is_first_turn=True,
                            )
                            messages.append(
                                ChatMessageUser(content=content)
                            )
                            is_first_llm_turn = False

                        move, inv, cats = await _get_llm_move_conversation(
                            model, messages, flags, game, board, llm_color
                        )
                    else:
                        move, inv, cats = await _get_llm_move_standalone(
                            model, flags, game, board, llm_color
                        )

                    total_invalid_moves += inv
                    for k, v in cats.items():
                        all_categories[k] = all_categories.get(k, 0) + v
                    if move is None:
                        llm_move_failed = True
                        break
                    logger.info(f"LLM moves: {board.san(move)}")
                    board.push(move)
                    node = node.add_variation(move)

                    if board.is_game_over(claim_draw=True):
                        break

                    # --- Stockfish turn (black) ---
                    sf_result = engine.play(
                        board, chess.engine.Limit(time=0.1)
                    )
                    assert sf_result.move is not None
                    sf_san = board.san(sf_result.move)
                    logger.info(f"Stockfish moves: {sf_san}")
                    board.push(sf_result.move)
                    node = node.add_variation(sf_result.move)

                    if (
                        flags["use_conversation"]
                        and not board.is_game_over(claim_draw=True)
                    ):
                        content = _build_turn_content(
                            flags, game, board, llm_color,
                            opponent_move_san=sf_san,
                        )
                        messages.append(ChatMessageUser(content=content))

                else:
                    # --- Stockfish turn (white) ---
                    sf_result = engine.play(
                        board, chess.engine.Limit(time=0.1)
                    )
                    assert sf_result.move is not None
                    sf_san = board.san(sf_result.move)
                    logger.info(f"Stockfish moves: {sf_san}")
                    board.push(sf_result.move)
                    node = node.add_variation(sf_result.move)

                    if board.is_game_over(claim_draw=True):
                        break

                    # --- LLM turn (black) ---
                    if flags["use_conversation"]:
                        if is_first_llm_turn:
                            content = _build_turn_content(
                                flags, game, board, llm_color,
                                is_first_turn=True,
                            )
                            is_first_llm_turn = False
                        else:
                            content = _build_turn_content(
                                flags, game, board, llm_color,
                                opponent_move_san=sf_san,
                            )
                        messages.append(ChatMessageUser(content=content))

                        move, inv, cats = await _get_llm_move_conversation(
                            model, messages, flags, game, board, llm_color
                        )
                    else:
                        move, inv, cats = await _get_llm_move_standalone(
                            model, flags, game, board, llm_color
                        )

                    total_invalid_moves += inv
                    for k, v in cats.items():
                        all_categories[k] = all_categories.get(k, 0) + v
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
        state.store.set("invalid_move_categories", all_categories)
        state.store.set(
            "termination", get_termination_reason(board, llm_move_failed)
        )
        state.store.set("stockfish_level", stockfish_level)
        state.store.set("llm_color", llm_color)
        state.store.set("scaffold_level", scaffold_level)

        return state

    return solve


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@task
def chess_eval_experiment(
    scaffold_level: int = 0,
    stockfish_levels: list[int] = [1],
    stockfish_path: str = "/opt/homebrew/bin/stockfish",
):
    """Evaluate LLM chess ability at a given scaffolding level.

    Each sample is a (stockfish_level, llm_color) combination.
    Use ``--epochs`` to play multiple games per configuration.
    """
    samples = []
    for level in stockfish_levels:
        for llm_color in ["white", "black"]:
            samples.append(
                Sample(
                    input=(
                        f"Play chess as {llm_color.title()} against "
                        f"Stockfish (scaffold level {scaffold_level})."
                    ),
                    target="1-0" if llm_color == "white" else "0-1",
                    id=f"sf{level}-{llm_color}-scaffold{scaffold_level}",
                    metadata={
                        "stockfish_level": level,
                        "stockfish_path": stockfish_path,
                        "llm_color": llm_color,
                        "scaffold_level": scaffold_level,
                    },
                )
            )
    sf_tag = "_".join(str(l) for l in stockfish_levels)
    return Task(
        dataset=samples,
        solver=chess_game_experiment(),
        scorer=chess_game_scorer(),
        name=f"chess_sf{sf_tag}_scaffold{scaffold_level}",
    )
