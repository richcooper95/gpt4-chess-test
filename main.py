import chess
import chess.engine
import chess.pgn

from openai import OpenAI

def get_stockfish_move(board):
    with chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish") as engine:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        return result.move

def get_gpt4_move_from_fen(client, board):

    prompt = f"""
You are chess Grandmaster. You are about to be given a board position in FEN notation, and
you will make the best move available to you, using all of your knowledge of chess. You
can take all the time to think that you need. Think deeply.

You will only reply with the move in UCI notation.

Here is the board position: {board.fen()}

Your move:
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
        print(f"Error getting move from GPT-4: {exc}")
        return None

# def add_stockfish_move_to_thread(thread, move):
#     thread.messages.create(
#         role="user",
#         content=f"Stockfish moves: {move}",
#     )

def get_gpt4_move_from_pgn_moves(client, game):
    moves = str(game).split("\n")[-1]

    prompt = f"""
You are chess Grandmaster. You are about to be given a list of moves in PGN notation, and
you will make the best move available to you, using all of your knowledge of chess. You
can take all the time to think that you need. Think deeply.

You will always play as Black.

You will only reply with your move in standard algebraic notation, e.g. "e4" or "Nf3".

Do not include the move number.

Here are the moves so far:
{moves}
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
        print(f"Error getting move from GPT-4: {exc}")
        return None

def get_gpt4_move(client, game, board):
    gpt4_move_raw = get_gpt4_move_from_pgn_moves(client, game)

    if gpt4_move_raw is None:
        print("Error getting move from GPT-4. Exiting.")
        return None

    try:
        return board.parse_san(gpt4_move_raw)
    except chess.InvalidMoveError:
        print(f"Syntactically invalid move from GPT-4: {gpt4_move_raw}.")
        return None
    except chess.IllegalMoveError:
        print(f"Semantically invalid move from GPT-4: {gpt4_move_raw}.")
        return None

def get_gpt4_move_with_retries(client, game, board):
    retries = 0
    while retries < 3:
        move = get_gpt4_move(client, game, board)
        if move is not None:
            return move
        retries += 1
    return None

def main():
    client = OpenAI()
    game = chess.pgn.Game()
    node = game
    board = chess.Board()

    while not board.is_game_over():
        # Stockfish's move
        stockfish_move = get_stockfish_move(board)
        stockfish_move_san = board.san(stockfish_move)
        print(f"Stockfish moves: {stockfish_move_san}")
        board.push(stockfish_move)
        node = node.add_variation(stockfish_move)

        if board.is_game_over():
            break

        # GPT-4's move
        gpt4_move = get_gpt4_move_with_retries(client, game, board)

        if gpt4_move is None:
            print("Error getting move from GPT-4. Exiting.")
            break

        print(f"GPT-4 moves: {board.san(gpt4_move)}")
        board.push(gpt4_move)
        node = node.add_variation(gpt4_move)

    # Game over
    if board.is_checkmate():
        print("Checkmate.")
    elif board.is_stalemate():
        print("Stalemate.")
    elif board.is_insufficient_material():
        print("Insufficient material.")
    else:
        print("Game over by other means.")

    game.headers["Result"] = board.result()
    game.headers["White"] = "Stockfish"
    game.headers["Black"] = "GPT-4"

    # Print the list of moves
    print("")
    print(game)

if __name__ == "__main__":
    main()
