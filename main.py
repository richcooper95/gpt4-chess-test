import argparse
import chess
import chess.engine
import chess.pgn

from openai import OpenAI
from typing import Optional, Set

debug = False


def print_debug(*args, **kwargs) -> None:
    if debug:
        print(*args, **kwargs)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a game of chess between Stockfish and GPT-4."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="/opt/homebrew/bin/stockfish",
        help="Path to the Stockfish binary. Defaults to /opt/homebrew/bin/stockfish",
    )
    return parser.parse_args()


def get_stockfish_move(board: chess.Board, stockfish_path: str) -> chess.Move:
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        assert result.move is not None
        return result.move


def moves_from_game(game: chess.pgn.Game) -> str:
    return str(game).split("\n")[-1]


def get_gpt4_move_from_pgn_moves(
    client: OpenAI, game: chess.pgn.Game, *, invalid_moves: Optional[Set[str]] = None
) -> Optional[str]:
    invalid_moves_message = ""
    if invalid_moves:
        print_debug("Invalid moves:", invalid_moves)
        invalid_moves_list = "\n ".join(invalid_moves)
        invalid_moves_message = (
            "\nNote that the following moves are invalid, and must not be returned:\n"
            f" {invalid_moves_list}\n"
        )

    prompt = f"""
You are chess Grandmaster. You are about to be given a list of moves in PGN notation, and
you will make the best move available to you, using all of your knowledge of chess. You
can take all the time to think that you need. Think deeply - you are a Grandmaster!

You will always play as Black.

You will only reply with your move in standard algebraic notation, e.g. "e4" or "Nf3".

If you are being asked to play a move, then there is still a valid move you can make. This
is always the case, even if you believe the game is over.

Do not include the move number. Do not include any other words in your response apart from
the move you wish to make. Do not return your move in quotes. Do not start your move with
a "+" symbol. Return a single move for Black only.
{invalid_moves_message}
Here are the moves so far:

{moves_from_game(game)}

Your move (which will replace the * in the line above):
"""

    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
        )

        return completion.choices[0].message.content.strip()
    except Exception as exc:
        print_debug(f"Error getting move from GPT-4: {exc}")
        return None


def get_gpt4_move(
    client: OpenAI, game: chess.pgn.Game, board: chess.Board
) -> Optional[chess.Move]:
    """
    Get a move from GPT-4, retrying up to 5 times if necessary.

    Returns:
    - A chess.Move object if a valid move is returned.
    - None if an error occurs, or if no valid move is returned after 5 retries.
    """
    invalid_moves: Set[str] = set()

    def _get_gpt4_move(
        client: OpenAI, game: chess.pgn.Game, board: chess.Board
    ) -> Optional[chess.Move]:
        nonlocal invalid_moves

        gpt4_move_raw = get_gpt4_move_from_pgn_moves(
            client, game, invalid_moves=invalid_moves
        )

        if gpt4_move_raw is None:
            print_debug("Error getting move from GPT-4. Exiting.")
            return None

        # GPT-4 likes to return moves starting with "+", which is not valid, so strip it.
        gpt4_move_raw = gpt4_move_raw.lstrip("+")

        try:
            return board.parse_san(gpt4_move_raw)
        except chess.InvalidMoveError:
            print_debug(f"Syntactically invalid move from GPT-4: {gpt4_move_raw}.")
        except chess.IllegalMoveError:
            print_debug(f"Semantically invalid move from GPT-4: {gpt4_move_raw}.")
        # TODO: Handle ambiguous moves differently to invalid moves (e.g. pass them in
        # as additional context to the next prompt, rather than discarding them entirely).
        except chess.AmbiguousMoveError:
            print_debug(f"Ambiguous move from GPT-4: {gpt4_move_raw}.")

        invalid_moves.add(gpt4_move_raw.strip())
        return None

    retries = 0
    while retries < 5:
        move = _get_gpt4_move(client, game, board)
        if move is not None:
            return move

        retries += 1
    return None


def play_game(
    client: OpenAI,
    board: chess.Board,
    game: chess.pgn.Game,
    node: chess.pgn.Game,
    stockfish_path: str,
) -> None:
    while not board.is_game_over():
        # Stockfish's move
        stockfish_move = get_stockfish_move(board, stockfish_path)

        print_debug(f"Stockfish moves: {board.san(stockfish_move)}")

        board.push(stockfish_move)
        node = node.add_variation(stockfish_move)

        if board.is_game_over():
            break

        # GPT-4's move
        gpt4_move = get_gpt4_move(client, game, board)

        if gpt4_move is None:
            print_debug("Error getting move from GPT-4. Exiting.")
            break

        print_debug(f"GPT-4 moves: {board.san(gpt4_move)}")

        board.push(gpt4_move)
        node = node.add_variation(gpt4_move)

    # Game over
    if board.is_checkmate():
        print_debug("Checkmate.")
    elif board.is_stalemate():
        print_debug("Stalemate.")
    elif board.is_insufficient_material():
        print_debug("Insufficient material.")
    else:
        print_debug("Game over by other means.")

    game.headers["Result"] = board.result()
    game.headers["White"] = "Stockfish"
    game.headers["Black"] = "GPT-4"


def main() -> None:
    global debug

    args = parse_cli()
    debug = args.debug

    client = OpenAI()

    # Track the game as a PGN. This allows us to print the list of moves to pass to GPT-4,
    # and to print the final list of moves at the end.
    game = chess.pgn.Game()
    node = game

    # Initialise the board.
    board = chess.Board()

    # Play the game.
    play_game(client, board, game, node, args.stockfish_path)

    # print the list of moves
    print_debug("")
    print(moves_from_game(game))
    print_debug("")
    print_debug(
        "Paste these moves into Chess.com to watch the game: https://www.chess.com/analysis"
    )


if __name__ == "__main__":
    main()
