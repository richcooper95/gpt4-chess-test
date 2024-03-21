# gpt4-chess-test

This is a simple script which plays a game of chess between Stockfish and GPT-4.

It passes through a list of moves in the game to the OpenAI API with a custom
prompt, and makes the suggested move in a game against Stockfish.

The PGN is then printed to console, so it can be pasted into a UI to analyse the game,
e.g. on [Chess.com](https://www.chess.com/analysis) or [Lichess](https://lichess.org/paste).

So far, Stockfish has (predictably) won every game. 🎣

## Usage

### Dependencies

Set the `OPENAPI_API_KEY` environment variable to an OpenAPI token from a paid account
with access to GPT-4.

Create a virtual environment, activate it, and install dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You also need to install a Stockfish engine, e.g. via `brew install stockfish` on MacOS.

### Run the script

Run the script as follows:

```
python3 main.py --debug --stockfish-path=/path/to/your/stockfish
```

The `--debug` flag will print each move, and any erroneous moves from GPT-4, in real time.

### Example

#### Console output

```
(venv) [Richs-MBP] [main] [] [gpt4-chess-test]
  > python3 main.py --debug
Stockfish moves: e4
GPT-4 moves: c5
Stockfish moves: Nc3
GPT-4 moves: d6
Stockfish moves: Nf3
GPT-4 moves: Nf6
Stockfish moves: d4
GPT-4 moves: cxd4
Stockfish moves: Nxd4
GPT-4 moves: a6
Stockfish moves: Bg5
GPT-4 moves: e6
Stockfish moves: f4
GPT-4 moves: Be7
Stockfish moves: Qf3
GPT-4 moves: Qc7
Stockfish moves: O-O-O
GPT-4 moves: Nbd7
Stockfish moves: Be2
GPT-4 moves: b5
Stockfish moves: Bxf6
GPT-4 moves: Nxf6
Stockfish moves: e5
GPT-4 moves: Bb7
Stockfish moves: exf6
GPT-4 moves: Bxf3
Stockfish moves: Bxf3
GPT-4 moves: Bxf6
Stockfish moves: Bxa8
Syntactically invalid move from GPT-4: "O-O".
Invalid moves: ['"O-O"']
GPT-4 moves: O-O
Stockfish moves: Bf3
GPT-4 moves: b4
Stockfish moves: Ne4
GPT-4 moves: Be7
Stockfish moves: Nc6
GPT-4 moves: Qxc6
Stockfish moves: Nf6+
GPT-4 moves: gxf6
Stockfish moves: Bxc6
GPT-4 moves: Rc8
Stockfish moves: Bb7
GPT-4 moves: Rc7
Stockfish moves: Bxa6
GPT-4 moves: Ra7
Stockfish moves: Bc4
GPT-4 moves: d5
Stockfish moves: Bxd5
GPT-4 moves: exd5
Stockfish moves: Kb1
GPT-4 moves: Bc5
Stockfish moves: Rxd5
GPT-4 moves: Be7
Stockfish moves: Rd3
GPT-4 moves: Kg7
Stockfish moves: Rhd1
GPT-4 moves: Bc5
Stockfish moves: Rg3+
GPT-4 moves: Kh6
Stockfish moves: Rh3+
GPT-4 moves: Kg7
Stockfish moves: Rg3+
GPT-4 moves: Kh6
Stockfish moves: Rgd3
GPT-4 moves: Kg7
Stockfish moves: Rd7
GPT-4 moves: Rxd7
Stockfish moves: Rxd7
GPT-4 moves: Kf8
Stockfish moves: f5
GPT-4 moves: Ke8
Stockfish moves: Rd3
GPT-4 moves: Ke7
Stockfish moves: Rh3
GPT-4 moves: Kd6
Stockfish moves: a4
GPT-4 moves: Ke5
Stockfish moves: Rd3
GPT-4 moves: Kxf5
Stockfish moves: a5
GPT-4 moves: Ke4
Stockfish moves: Rh3
GPT-4 moves: f5
Stockfish moves: a6
GPT-4 moves: Ba7
Stockfish moves: Rd3
GPT-4 moves: f4
Stockfish moves: Rd7
GPT-4 moves: Bc5
Stockfish moves: Rd3
Semantically invalid move from GPT-4: f5-f3.
Invalid moves: ['f5-f3']
GPT-4 moves: Ba7
Stockfish moves: Rd7
GPT-4 moves: Bc5
Stockfish moves: Rc7
GPT-4 moves: Bb6
Stockfish moves: Rb7
GPT-4 moves: Ba7
Stockfish moves: Rxa7
GPT-4 moves: f5
Stockfish moves: Rxh7
GPT-4 moves: f3
Stockfish moves: gxf3+
GPT-4 moves: Kxf3
Stockfish moves: a7
GPT-4 moves: f4
Stockfish moves: a8=Q+
GPT-4 moves: Ke2
Stockfish moves: Qg2+
GPT-4 moves: Ke1
Stockfish moves: Re7+
GPT-4 moves: Kd1
Stockfish moves: Qe2#
Checkmate.

1. e4 c5 2. Nc3 d6 3. Nf3 Nf6 4. d4 cxd4 5. Nxd4 a6 6. Bg5 e6 7. f4 Be7 8. Qf3 Qc7 9. O-O-O Nbd7 10. Be2 b5 11. Bxf6 Nxf6 12. e5 Bb7 13. exf6 Bxf3 14. Bxf3 Bxf6 15. Bxa8 O-O 16. Bf3 b4 17. Ne4 Be7 18. Nc6 Qxc6 19. Nf6+ gxf6 20. Bxc6 Rc8 21. Bb7 Rc7 22. Bxa6 Ra7 23. Bc4 d5 24. Bxd5 exd5 25. Kb1 Bc5 26. Rxd5 Be7 27. Rd3 Kg7 28. Rhd1 Bc5 29. Rg3+ Kh6 30. Rh3+ Kg7 31. Rg3+ Kh6 32. Rgd3 Kg7 33. Rd7 Rxd7 34. Rxd7 Kf8 35. f5 Ke8 36. Rd3 Ke7 37. Rh3 Kd6 38. a4 Ke5 39. Rd3 Kxf5 40. a5 Ke4 41. Rh3 f5 42. a6 Ba7 43. Rd3 f4 44. Rd7 Bc5 45. Rd3 Ba7 46. Rd7 Bc5 47. Rc7 Bb6 48. Rb7 Ba7 49. Rxa7 f5 50. Rxh7 f3 51. gxf3+ Kxf3 52. a7 f4 53. a8=Q+ Ke2 54. Qg2+ Ke1 55. Re7+ Kd1 56. Qe2# 1-0
```

#### Watching the game

The video below shows Stockfish (white) vs. GPT-4 (black). This was one of the closer games, where GPT-4 actually held its own for a while!

https://github.com/richcooper95/gpt4-chess-test/assets/58304039/7372b758-9e93-4f9b-a07d-fa64c227f9d1
