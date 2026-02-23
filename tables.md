# Experiment Results Tables

Total games: 162 | Models: 3 | Scaffold levels: [0, 1, 2, 3, 4, 5, 6, 7, 8]

## 1. Mean Game Length (full moves)

| Model | Baseline | Accum. Invalid | Conv. History | FEN String | Board Image | Structured Resp. | Legal Moves | Piece Rules | Path Tracing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 21.3 +/- 2.8 | 26.0 +/- 3.9 | 34.0 +/- 5.1 | 26.7 +/- 2.4 | 34.5 +/- 2.6 | 27.5 +/- 2.6 | 25.8 +/- 3.9 | 25.5 +/- 2.2 | 26.5 +/- 3.5 |
| gemini-3-pro-preview | 16.0 +/- 1.5 | 17.8 +/- 5.5 | 42.7 +/- 4.6 | 49.8 +/- 7.1 | 41.3 +/- 7.3 | 36.0 +/- 6.4 | 62.5 +/- 9.8 | 47.7 +/- 9.1 | 59.2 +/- 6.7 |
| gpt-5.2 | 45.2 +/- 8.9 | 50.7 +/- 13.0 | 40.0 +/- 1.2 | 51.0 +/- 7.6 | 61.7 +/- 6.6 | 51.5 +/- 9.9 | 47.7 +/- 7.4 | 33.5 +/- 3.0 | 45.0 +/- 8.8 |

## 2. Mean Invalid Moves per Game

| Model | Baseline | Accum. Invalid | Conv. History | FEN String | Board Image | Structured Resp. | Legal Moves | Piece Rules | Path Tracing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 4.2 +/- 0.8 | 5.7 +/- 1.0 | 3.2 +/- 0.4 | 2.3 +/- 1.0 | 3.7 +/- 1.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| gemini-3-pro-preview | 11.2 +/- 1.1 | 14.0 +/- 5.2 | 2.3 +/- 1.1 | 1.3 +/- 0.5 | 0.8 +/- 0.6 | 0.2 +/- 0.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.2 +/- 0.2 |
| gpt-5.2 | 7.0 +/- 1.6 | 10.2 +/- 4.0 | 4.3 +/- 0.3 | 1.8 +/- 0.5 | 4.7 +/- 1.0 | 1.7 +/- 0.5 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.7 +/- 0.5 |

## 3. Win / Draw / Loss Rates (%)

| Model | Baseline | Accum. Invalid | Conv. History | FEN String | Board Image | Structured Resp. | Legal Moves | Piece Rules | Path Tracing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% |
| gemini-3-pro-preview | W 16.7% / D 0.0% / L 83.3% | W 0.0% / D 0.0% / L 100.0% | W 66.7% / D 0.0% / L 33.3% | W 66.7% / D 0.0% / L 33.3% | W 83.3% / D 0.0% / L 16.7% | W 50.0% / D 0.0% / L 50.0% | W 66.7% / D 16.7% / L 16.7% | W 50.0% / D 0.0% / L 50.0% | W 16.7% / D 16.7% / L 66.7% |
| gpt-5.2 | W 16.7% / D 0.0% / L 83.3% | W 16.7% / D 0.0% / L 83.3% | W 0.0% / D 0.0% / L 100.0% | W 16.7% / D 0.0% / L 83.3% | W 16.7% / D 0.0% / L 83.3% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 0.0% / D 0.0% / L 100.0% | W 33.3% / D 0.0% / L 66.7% |

## 4. White vs Black Performance

| Model | Level | Color | Mean Length | SE | Win Rate | N |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 0 (Baseline) | white | 24.7 | 2.7 | 0.0% | 3 |
| claude-opus-4-6 | 0 (Baseline) | black | 18.0 | 4.1 | 0.0% | 3 |
| claude-opus-4-6 | 1 (Accum. Invalid) | white | 31.0 | 6.6 | 0.0% | 3 |
| claude-opus-4-6 | 1 (Accum. Invalid) | black | 21.0 | 0.5 | 0.0% | 3 |
| claude-opus-4-6 | 2 (Conv. History) | white | 39.7 | 9.0 | 0.0% | 3 |
| claude-opus-4-6 | 2 (Conv. History) | black | 28.3 | 0.7 | 0.0% | 3 |
| claude-opus-4-6 | 3 (FEN String) | white | 28.7 | 3.0 | 0.0% | 3 |
| claude-opus-4-6 | 3 (FEN String) | black | 24.7 | 3.3 | 0.0% | 3 |
| claude-opus-4-6 | 4 (Board Image) | white | 34.7 | 4.3 | 0.0% | 3 |
| claude-opus-4-6 | 4 (Board Image) | black | 34.3 | 3.0 | 0.0% | 3 |
| claude-opus-4-6 | 5 (Structured Resp.) | white | 22.7 | 2.9 | 0.0% | 3 |
| claude-opus-4-6 | 5 (Structured Resp.) | black | 32.3 | 1.5 | 0.0% | 3 |
| claude-opus-4-6 | 6 (Legal Moves) | white | 31.3 | 6.1 | 0.0% | 3 |
| claude-opus-4-6 | 6 (Legal Moves) | black | 20.3 | 1.5 | 0.0% | 3 |
| claude-opus-4-6 | 7 (Piece Rules) | white | 22.7 | 2.4 | 0.0% | 3 |
| claude-opus-4-6 | 7 (Piece Rules) | black | 28.3 | 2.8 | 0.0% | 3 |
| claude-opus-4-6 | 8 (Path Tracing) | white | 31.3 | 4.5 | 0.0% | 3 |
| claude-opus-4-6 | 8 (Path Tracing) | black | 21.7 | 3.7 | 0.0% | 3 |
| gemini-3-pro-preview | 0 (Baseline) | white | 12.7 | 0.3 | 0.0% | 3 |
| gemini-3-pro-preview | 0 (Baseline) | black | 19.3 | 1.0 | 33.3% | 3 |
| gemini-3-pro-preview | 1 (Accum. Invalid) | white | 14.7 | 1.7 | 0.0% | 3 |
| gemini-3-pro-preview | 1 (Accum. Invalid) | black | 21.0 | 10.7 | 0.0% | 3 |
| gemini-3-pro-preview | 2 (Conv. History) | white | 45.3 | 3.9 | 66.7% | 3 |
| gemini-3-pro-preview | 2 (Conv. History) | black | 40.0 | 7.9 | 66.7% | 3 |
| gemini-3-pro-preview | 3 (FEN String) | white | 60.3 | 10.4 | 33.3% | 3 |
| gemini-3-pro-preview | 3 (FEN String) | black | 39.3 | 4.6 | 100.0% | 3 |
| gemini-3-pro-preview | 4 (Board Image) | white | 31.3 | 2.2 | 66.7% | 3 |
| gemini-3-pro-preview | 4 (Board Image) | black | 51.3 | 12.0 | 100.0% | 3 |
| gemini-3-pro-preview | 5 (Structured Resp.) | white | 46.0 | 9.7 | 33.3% | 3 |
| gemini-3-pro-preview | 5 (Structured Resp.) | black | 26.0 | 2.1 | 66.7% | 3 |
| gemini-3-pro-preview | 6 (Legal Moves) | white | 79.3 | 11.4 | 66.7% | 3 |
| gemini-3-pro-preview | 6 (Legal Moves) | black | 45.7 | 7.9 | 66.7% | 3 |
| gemini-3-pro-preview | 7 (Piece Rules) | white | 53.3 | 14.4 | 33.3% | 3 |
| gemini-3-pro-preview | 7 (Piece Rules) | black | 42.0 | 10.2 | 66.7% | 3 |
| gemini-3-pro-preview | 8 (Path Tracing) | white | 66.7 | 10.1 | 0.0% | 3 |
| gemini-3-pro-preview | 8 (Path Tracing) | black | 51.7 | 6.4 | 33.3% | 3 |
| gpt-5.2 | 0 (Baseline) | white | 40.7 | 6.5 | 0.0% | 3 |
| gpt-5.2 | 0 (Baseline) | black | 49.7 | 16.0 | 33.3% | 3 |
| gpt-5.2 | 1 (Accum. Invalid) | white | 66.7 | 21.3 | 33.3% | 3 |
| gpt-5.2 | 1 (Accum. Invalid) | black | 34.7 | 6.7 | 0.0% | 3 |
| gpt-5.2 | 2 (Conv. History) | white | 42.7 | 1.1 | 0.0% | 3 |
| gpt-5.2 | 2 (Conv. History) | black | 37.3 | 0.3 | 0.0% | 3 |
| gpt-5.2 | 3 (FEN String) | white | 38.0 | 2.5 | 33.3% | 3 |
| gpt-5.2 | 3 (FEN String) | black | 64.0 | 10.7 | 0.0% | 3 |
| gpt-5.2 | 4 (Board Image) | white | 48.7 | 7.7 | 33.3% | 3 |
| gpt-5.2 | 4 (Board Image) | black | 74.7 | 2.2 | 0.0% | 3 |
| gpt-5.2 | 5 (Structured Resp.) | white | 69.0 | 2.1 | 0.0% | 3 |
| gpt-5.2 | 5 (Structured Resp.) | black | 34.0 | 13.5 | 0.0% | 3 |
| gpt-5.2 | 6 (Legal Moves) | white | 54.0 | 11.7 | 0.0% | 3 |
| gpt-5.2 | 6 (Legal Moves) | black | 41.3 | 7.5 | 0.0% | 3 |
| gpt-5.2 | 7 (Piece Rules) | white | 38.7 | 3.4 | 0.0% | 3 |
| gpt-5.2 | 7 (Piece Rules) | black | 28.3 | 2.3 | 0.0% | 3 |
| gpt-5.2 | 8 (Path Tracing) | white | 58.0 | 12.2 | 66.7% | 3 |
| gpt-5.2 | 8 (Path Tracing) | black | 32.0 | 7.1 | 0.0% | 3 |

## 5. Termination Reasons by Scaffold Level

| Level | checkmate | draw_claimable | invalid_move_failure | stalemate | Total |
| --- | --- | --- | --- | --- | --- |
| 0 (Baseline) | 6 (33.3%) | 0 (0.0%) | 12 (66.7%) | 0 (0.0%) | 18 |
| 1 (Accum. Invalid) | 7 (38.9%) | 1 (5.6%) | 10 (55.6%) | 0 (0.0%) | 18 |
| 2 (Conv. History) | 16 (88.9%) | 2 (11.1%) | 0 (0.0%) | 0 (0.0%) | 18 |
| 3 (FEN String) | 17 (94.4%) | 1 (5.6%) | 0 (0.0%) | 0 (0.0%) | 18 |
| 4 (Board Image) | 15 (83.3%) | 3 (16.7%) | 0 (0.0%) | 0 (0.0%) | 18 |
| 5 (Structured Resp.) | 14 (77.8%) | 4 (22.2%) | 0 (0.0%) | 0 (0.0%) | 18 |
| 6 (Legal Moves) | 17 (94.4%) | 0 (0.0%) | 0 (0.0%) | 1 (5.6%) | 18 |
| 7 (Piece Rules) | 17 (94.4%) | 1 (5.6%) | 0 (0.0%) | 0 (0.0%) | 18 |
| 8 (Path Tracing) | 14 (77.8%) | 3 (16.7%) | 0 (0.0%) | 1 (5.6%) | 18 |

## 6. Invalid Move Categories (total counts)

| Model | ambiguous | api_error | ignored_check | moved_pinned_piece | no_move_line | no_such_piece | unparseable | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 1 | 0 | 17 | 82 | 0 | 9 | 5 | 114 |
| gemini-3-pro-preview | 2 | 0 | 0 | 25 | 1 | 149 | 3 | 180 |
| gpt-5.2 | 1 | 2 | 11 | 155 | 8 | 0 | 5 | 182 |

## 7. Composite Performance Score (CPS)

| Model | Baseline | Accum. Invalid | Conv. History | FEN String | Board Image | Structured Resp. | Legal Moves | Piece Rules | Path Tracing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 0.120 +/- 0.016 | 0.146 +/- 0.022 | 0.191 +/- 0.029 | 0.150 +/- 0.013 | 0.194 +/- 0.015 | 0.155 +/- 0.014 | 0.145 +/- 0.022 | 0.143 +/- 0.012 | 0.149 +/- 0.020 |
| gemini-3-pro-preview | 0.238 +/- 0.139 | 0.100 +/- 0.031 | 0.771 +/- 0.132 | 0.804 +/- 0.113 | 0.858 +/- 0.130 | 0.605 +/- 0.164 | 0.835 +/- 0.101 | 0.686 +/- 0.130 | 0.498 +/- 0.106 |
| gpt-5.2 | 0.362 +/- 0.125 | 0.357 +/- 0.127 | 0.225 +/- 0.007 | 0.405 +/- 0.114 | 0.473 +/- 0.101 | 0.290 +/- 0.056 | 0.268 +/- 0.042 | 0.188 +/- 0.017 | 0.482 +/- 0.153 |

## 8. Marginal Effect (change from previous level)

| Model | Level | Delta Game Length | Delta Invalid Moves |
| --- | --- | --- | --- |
| claude-opus-4-6 | 1 (Accum. Invalid) | +4.7 | +1.5 |
| claude-opus-4-6 | 2 (Conv. History) | +8.0 | -2.5 |
| claude-opus-4-6 | 3 (FEN String) | -7.3 | -0.8 |
| claude-opus-4-6 | 4 (Board Image) | +7.8 | +1.3 |
| claude-opus-4-6 | 5 (Structured Resp.) | -7.0 | -3.7 |
| claude-opus-4-6 | 6 (Legal Moves) | -1.7 | +0.0 |
| claude-opus-4-6 | 7 (Piece Rules) | -0.3 | +0.0 |
| claude-opus-4-6 | 8 (Path Tracing) | +1.0 | +0.0 |
| gemini-3-pro-preview | 1 (Accum. Invalid) | +1.8 | +2.8 |
| gemini-3-pro-preview | 2 (Conv. History) | +24.8 | -11.7 |
| gemini-3-pro-preview | 3 (FEN String) | +7.2 | -1.0 |
| gemini-3-pro-preview | 4 (Board Image) | -8.5 | -0.5 |
| gemini-3-pro-preview | 5 (Structured Resp.) | -5.3 | -0.7 |
| gemini-3-pro-preview | 6 (Legal Moves) | +26.5 | -0.2 |
| gemini-3-pro-preview | 7 (Piece Rules) | -14.8 | +0.0 |
| gemini-3-pro-preview | 8 (Path Tracing) | +11.5 | +0.2 |
| gpt-5.2 | 1 (Accum. Invalid) | +5.5 | +3.2 |
| gpt-5.2 | 2 (Conv. History) | -10.7 | -5.8 |
| gpt-5.2 | 3 (FEN String) | +11.0 | -2.5 |
| gpt-5.2 | 4 (Board Image) | +10.7 | +2.8 |
| gpt-5.2 | 5 (Structured Resp.) | -10.2 | -3.0 |
| gpt-5.2 | 6 (Legal Moves) | -3.8 | -1.7 |
| gpt-5.2 | 7 (Piece Rules) | -14.2 | +0.0 |
| gpt-5.2 | 8 (Path Tracing) | +11.5 | +0.7 |

## 10. Per-Game Results (all games)

| Model | Level | Color | Moves | Invalid | Score | CPS | Termination | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 0 (Baseline) | black | 13 | 5 | Loss | 0.073 | invalid_move_failure | * |
| claude-opus-4-6 | 0 (Baseline) | black | 13 | 5 | Loss | 0.073 | invalid_move_failure | * |
| claude-opus-4-6 | 0 (Baseline) | black | 28 | 0 | Loss | 0.158 | checkmate | 1-0 |
| claude-opus-4-6 | 0 (Baseline) | white | 20 | 5 | Loss | 0.113 | invalid_move_failure | * |
| claude-opus-4-6 | 0 (Baseline) | white | 23 | 5 | Loss | 0.129 | invalid_move_failure | * |
| claude-opus-4-6 | 0 (Baseline) | white | 31 | 5 | Loss | 0.174 | invalid_move_failure | * |
| claude-opus-4-6 | 1 (Accum. Invalid) | black | 22 | 3 | Loss | 0.124 | checkmate | 1-0 |
| claude-opus-4-6 | 1 (Accum. Invalid) | black | 21 | 4 | Loss | 0.118 | checkmate | 1-0 |
| claude-opus-4-6 | 1 (Accum. Invalid) | black | 20 | 4 | Loss | 0.113 | checkmate | 1-0 |
| claude-opus-4-6 | 1 (Accum. Invalid) | white | 25 | 6 | Loss | 0.141 | invalid_move_failure | * |
| claude-opus-4-6 | 1 (Accum. Invalid) | white | 47 | 7 | Loss | 0.264 | invalid_move_failure | * |
| claude-opus-4-6 | 1 (Accum. Invalid) | white | 21 | 10 | Loss | 0.118 | invalid_move_failure | * |
| claude-opus-4-6 | 2 (Conv. History) | black | 27 | 2 | Loss | 0.152 | checkmate | 1-0 |
| claude-opus-4-6 | 2 (Conv. History) | black | 28 | 3 | Loss | 0.158 | checkmate | 1-0 |
| claude-opus-4-6 | 2 (Conv. History) | black | 30 | 2 | Loss | 0.169 | checkmate | 1-0 |
| claude-opus-4-6 | 2 (Conv. History) | white | 22 | 3 | Loss | 0.124 | checkmate | 0-1 |
| claude-opus-4-6 | 2 (Conv. History) | white | 37 | 5 | Loss | 0.208 | checkmate | 0-1 |
| claude-opus-4-6 | 2 (Conv. History) | white | 60 | 4 | Loss | 0.338 | checkmate | 0-1 |
| claude-opus-4-6 | 3 (FEN String) | black | 17 | 0 | Loss | 0.096 | checkmate | 1-0 |
| claude-opus-4-6 | 3 (FEN String) | black | 31 | 7 | Loss | 0.174 | checkmate | 1-0 |
| claude-opus-4-6 | 3 (FEN String) | black | 26 | 0 | Loss | 0.146 | checkmate | 1-0 |
| claude-opus-4-6 | 3 (FEN String) | white | 24 | 2 | Loss | 0.135 | checkmate | 0-1 |
| claude-opus-4-6 | 3 (FEN String) | white | 36 | 4 | Loss | 0.203 | checkmate | 0-1 |
| claude-opus-4-6 | 3 (FEN String) | white | 26 | 1 | Loss | 0.146 | checkmate | 0-1 |
| claude-opus-4-6 | 4 (Board Image) | black | 27 | 2 | Loss | 0.152 | checkmate | 1-0 |
| claude-opus-4-6 | 4 (Board Image) | black | 38 | 9 | Loss | 0.214 | checkmate | 1-0 |
| claude-opus-4-6 | 4 (Board Image) | black | 38 | 3 | Loss | 0.214 | checkmate | 1-0 |
| claude-opus-4-6 | 4 (Board Image) | white | 44 | 4 | Loss | 0.248 | draw_claimable | * |
| claude-opus-4-6 | 4 (Board Image) | white | 26 | 2 | Loss | 0.146 | checkmate | 0-1 |
| claude-opus-4-6 | 4 (Board Image) | white | 34 | 2 | Loss | 0.191 | checkmate | 0-1 |
| claude-opus-4-6 | 5 (Structured Resp.) | black | 30 | 0 | Loss | 0.169 | checkmate | 1-0 |
| claude-opus-4-6 | 5 (Structured Resp.) | black | 31 | 0 | Loss | 0.174 | checkmate | 1-0 |
| claude-opus-4-6 | 5 (Structured Resp.) | black | 36 | 0 | Loss | 0.203 | checkmate | 1-0 |
| claude-opus-4-6 | 5 (Structured Resp.) | white | 24 | 0 | Loss | 0.135 | checkmate | 0-1 |
| claude-opus-4-6 | 5 (Structured Resp.) | white | 28 | 0 | Loss | 0.158 | checkmate | 0-1 |
| claude-opus-4-6 | 5 (Structured Resp.) | white | 16 | 0 | Loss | 0.090 | checkmate | 0-1 |
| claude-opus-4-6 | 6 (Legal Moves) | black | 18 | 0 | Loss | 0.101 | checkmate | 1-0 |
| claude-opus-4-6 | 6 (Legal Moves) | black | 24 | 0 | Loss | 0.135 | checkmate | 1-0 |
| claude-opus-4-6 | 6 (Legal Moves) | black | 19 | 0 | Loss | 0.107 | checkmate | 1-0 |
| claude-opus-4-6 | 6 (Legal Moves) | white | 44 | 0 | Loss | 0.248 | checkmate | 0-1 |
| claude-opus-4-6 | 6 (Legal Moves) | white | 32 | 0 | Loss | 0.180 | checkmate | 0-1 |
| claude-opus-4-6 | 6 (Legal Moves) | white | 18 | 0 | Loss | 0.101 | checkmate | 0-1 |
| claude-opus-4-6 | 7 (Piece Rules) | black | 22 | 0 | Loss | 0.124 | checkmate | 1-0 |
| claude-opus-4-6 | 7 (Piece Rules) | black | 29 | 0 | Loss | 0.163 | checkmate | 1-0 |
| claude-opus-4-6 | 7 (Piece Rules) | black | 34 | 0 | Loss | 0.191 | checkmate | 1-0 |
| claude-opus-4-6 | 7 (Piece Rules) | white | 28 | 0 | Loss | 0.158 | checkmate | 0-1 |
| claude-opus-4-6 | 7 (Piece Rules) | white | 18 | 0 | Loss | 0.101 | checkmate | 0-1 |
| claude-opus-4-6 | 7 (Piece Rules) | white | 22 | 0 | Loss | 0.124 | checkmate | 0-1 |
| claude-opus-4-6 | 8 (Path Tracing) | black | 24 | 0 | Loss | 0.135 | checkmate | 1-0 |
| claude-opus-4-6 | 8 (Path Tracing) | black | 13 | 0 | Loss | 0.073 | checkmate | 1-0 |
| claude-opus-4-6 | 8 (Path Tracing) | black | 28 | 0 | Loss | 0.158 | checkmate | 1-0 |
| claude-opus-4-6 | 8 (Path Tracing) | white | 40 | 0 | Loss | 0.225 | draw_claimable | * |
| claude-opus-4-6 | 8 (Path Tracing) | white | 33 | 0 | Loss | 0.186 | checkmate | 0-1 |
| claude-opus-4-6 | 8 (Path Tracing) | white | 21 | 0 | Loss | 0.118 | checkmate | 0-1 |
| gemini-3-pro-preview | 0 (Baseline) | black | 21 | 16 | Loss | 0.118 | invalid_move_failure | * |
| gemini-3-pro-preview | 0 (Baseline) | black | 20 | 12 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 0 (Baseline) | black | 17 | 10 | Loss | 0.096 | invalid_move_failure | * |
| gemini-3-pro-preview | 0 (Baseline) | white | 12 | 8 | Loss | 0.068 | invalid_move_failure | * |
| gemini-3-pro-preview | 0 (Baseline) | white | 13 | 12 | Loss | 0.073 | invalid_move_failure | * |
| gemini-3-pro-preview | 0 (Baseline) | white | 13 | 9 | Loss | 0.073 | invalid_move_failure | * |
| gemini-3-pro-preview | 1 (Accum. Invalid) | black | 10 | 5 | Loss | 0.056 | invalid_move_failure | * |
| gemini-3-pro-preview | 1 (Accum. Invalid) | black | 47 | 41 | Loss | 0.264 | invalid_move_failure | * |
| gemini-3-pro-preview | 1 (Accum. Invalid) | black | 6 | 5 | Loss | 0.034 | invalid_move_failure | * |
| gemini-3-pro-preview | 1 (Accum. Invalid) | white | 18 | 8 | Loss | 0.101 | invalid_move_failure | * |
| gemini-3-pro-preview | 1 (Accum. Invalid) | white | 11 | 7 | Loss | 0.062 | invalid_move_failure | * |
| gemini-3-pro-preview | 1 (Accum. Invalid) | white | 15 | 18 | Loss | 0.084 | invalid_move_failure | * |
| gemini-3-pro-preview | 2 (Conv. History) | black | 27 | 1 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 2 (Conv. History) | black | 59 | 2 | Loss | 0.332 | draw_claimable | * |
| gemini-3-pro-preview | 2 (Conv. History) | black | 34 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 2 (Conv. History) | white | 48 | 1 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 2 (Conv. History) | white | 52 | 8 | Loss | 0.293 | checkmate | 0-1 |
| gemini-3-pro-preview | 2 (Conv. History) | white | 36 | 2 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 3 (FEN String) | black | 50 | 1 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 3 (FEN String) | black | 37 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 3 (FEN String) | black | 31 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 3 (FEN String) | white | 35 | 3 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 3 (FEN String) | white | 70 | 3 | Loss | 0.394 | draw_claimable | * |
| gemini-3-pro-preview | 3 (FEN String) | white | 76 | 1 | Loss | 0.427 | checkmate | 0-1 |
| gemini-3-pro-preview | 4 (Board Image) | black | 79 | 4 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 4 (Board Image) | black | 46 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 4 (Board Image) | black | 29 | 1 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 4 (Board Image) | white | 26 | 0 | Loss | 0.146 | draw_claimable | * |
| gemini-3-pro-preview | 4 (Board Image) | white | 34 | 0 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 4 (Board Image) | white | 34 | 0 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 5 (Structured Resp.) | black | 31 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 5 (Structured Resp.) | black | 23 | 0 | Loss | 0.129 | checkmate | 1-0 |
| gemini-3-pro-preview | 5 (Structured Resp.) | black | 24 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 5 (Structured Resp.) | white | 65 | 1 | Loss | 0.366 | draw_claimable | * |
| gemini-3-pro-preview | 5 (Structured Resp.) | white | 24 | 0 | Loss | 0.135 | checkmate | 0-1 |
| gemini-3-pro-preview | 5 (Structured Resp.) | white | 49 | 0 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 6 (Legal Moves) | black | 42 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 6 (Legal Moves) | black | 64 | 0 | Loss | 0.360 | checkmate | 1-0 |
| gemini-3-pro-preview | 6 (Legal Moves) | black | 31 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 6 (Legal Moves) | white | 62 | 0 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 6 (Legal Moves) | white | 69 | 0 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 6 (Legal Moves) | white | 107 | 0 | Draw | 0.650 | stalemate | 1/2-1/2 |
| gemini-3-pro-preview | 7 (Piece Rules) | black | 39 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 7 (Piece Rules) | black | 65 | 0 | Loss | 0.366 | draw_claimable | * |
| gemini-3-pro-preview | 7 (Piece Rules) | black | 22 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 7 (Piece Rules) | white | 84 | 0 | Loss | 0.450 | checkmate | 0-1 |
| gemini-3-pro-preview | 7 (Piece Rules) | white | 53 | 0 | Loss | 0.298 | checkmate | 0-1 |
| gemini-3-pro-preview | 7 (Piece Rules) | white | 23 | 0 | Win | 1.000 | checkmate | 1-0 |
| gemini-3-pro-preview | 8 (Path Tracing) | black | 36 | 0 | Win | 1.000 | checkmate | 0-1 |
| gemini-3-pro-preview | 8 (Path Tracing) | black | 59 | 1 | Loss | 0.332 | checkmate | 1-0 |
| gemini-3-pro-preview | 8 (Path Tracing) | black | 60 | 0 | Loss | 0.338 | checkmate | 1-0 |
| gemini-3-pro-preview | 8 (Path Tracing) | white | 81 | 0 | Draw | 0.650 | stalemate | 1/2-1/2 |
| gemini-3-pro-preview | 8 (Path Tracing) | white | 77 | 0 | Loss | 0.433 | draw_claimable | * |
| gemini-3-pro-preview | 8 (Path Tracing) | white | 42 | 0 | Loss | 0.236 | checkmate | 0-1 |
| gpt-5.2 | 0 (Baseline) | black | 75 | 13 | Loss | 0.422 | checkmate | 1-0 |
| gpt-5.2 | 0 (Baseline) | black | 11 | 5 | Loss | 0.062 | invalid_move_failure | * |
| gpt-5.2 | 0 (Baseline) | black | 63 | 5 | Win | 1.000 | checkmate | 0-1 |
| gpt-5.2 | 0 (Baseline) | white | 37 | 11 | Loss | 0.208 | invalid_move_failure | * |
| gpt-5.2 | 0 (Baseline) | white | 29 | 1 | Loss | 0.163 | checkmate | 0-1 |
| gpt-5.2 | 0 (Baseline) | white | 56 | 7 | Loss | 0.315 | checkmate | 0-1 |
| gpt-5.2 | 1 (Accum. Invalid) | black | 47 | 14 | Loss | 0.264 | invalid_move_failure | * |
| gpt-5.2 | 1 (Accum. Invalid) | black | 38 | 8 | Loss | 0.214 | checkmate | 1-0 |
| gpt-5.2 | 1 (Accum. Invalid) | black | 19 | 3 | Loss | 0.107 | checkmate | 1-0 |
| gpt-5.2 | 1 (Accum. Invalid) | white | 72 | 6 | Win | 1.000 | checkmate | 1-0 |
| gpt-5.2 | 1 (Accum. Invalid) | white | 109 | 30 | Loss | 0.450 | draw_claimable | * |
| gpt-5.2 | 1 (Accum. Invalid) | white | 19 | 0 | Loss | 0.107 | checkmate | 0-1 |
| gpt-5.2 | 2 (Conv. History) | black | 37 | 5 | Loss | 0.208 | checkmate | 1-0 |
| gpt-5.2 | 2 (Conv. History) | black | 38 | 4 | Loss | 0.214 | checkmate | 1-0 |
| gpt-5.2 | 2 (Conv. History) | black | 37 | 4 | Loss | 0.208 | checkmate | 1-0 |
| gpt-5.2 | 2 (Conv. History) | white | 44 | 5 | Loss | 0.248 | checkmate | 0-1 |
| gpt-5.2 | 2 (Conv. History) | white | 44 | 5 | Loss | 0.248 | checkmate | 0-1 |
| gpt-5.2 | 2 (Conv. History) | white | 40 | 3 | Loss | 0.225 | draw_claimable | * |
| gpt-5.2 | 3 (FEN String) | black | 53 | 0 | Loss | 0.298 | checkmate | 1-0 |
| gpt-5.2 | 3 (FEN String) | black | 49 | 1 | Loss | 0.276 | checkmate | 1-0 |
| gpt-5.2 | 3 (FEN String) | black | 90 | 4 | Loss | 0.450 | checkmate | 1-0 |
| gpt-5.2 | 3 (FEN String) | white | 32 | 1 | Loss | 0.180 | checkmate | 0-1 |
| gpt-5.2 | 3 (FEN String) | white | 42 | 2 | Win | 1.000 | checkmate | 1-0 |
| gpt-5.2 | 3 (FEN String) | white | 40 | 3 | Loss | 0.225 | checkmate | 0-1 |
| gpt-5.2 | 4 (Board Image) | black | 71 | 4 | Loss | 0.399 | checkmate | 1-0 |
| gpt-5.2 | 4 (Board Image) | black | 80 | 5 | Loss | 0.450 | draw_claimable | * |
| gpt-5.2 | 4 (Board Image) | black | 73 | 5 | Loss | 0.411 | checkmate | 1-0 |
| gpt-5.2 | 4 (Board Image) | white | 43 | 4 | Win | 1.000 | checkmate | 1-0 |
| gpt-5.2 | 4 (Board Image) | white | 67 | 9 | Loss | 0.377 | checkmate | 0-1 |
| gpt-5.2 | 4 (Board Image) | white | 36 | 1 | Loss | 0.203 | checkmate | 0-1 |
| gpt-5.2 | 5 (Structured Resp.) | black | 67 | 1 | Loss | 0.377 | draw_claimable | * |
| gpt-5.2 | 5 (Structured Resp.) | black | 17 | 0 | Loss | 0.096 | checkmate | 1-0 |
| gpt-5.2 | 5 (Structured Resp.) | black | 18 | 3 | Loss | 0.101 | checkmate | 1-0 |
| gpt-5.2 | 5 (Structured Resp.) | white | 67 | 2 | Loss | 0.377 | checkmate | 0-1 |
| gpt-5.2 | 5 (Structured Resp.) | white | 74 | 1 | Loss | 0.416 | draw_claimable | * |
| gpt-5.2 | 5 (Structured Resp.) | white | 66 | 3 | Loss | 0.371 | draw_claimable | * |
| gpt-5.2 | 6 (Legal Moves) | black | 24 | 0 | Loss | 0.135 | checkmate | 1-0 |
| gpt-5.2 | 6 (Legal Moves) | black | 45 | 0 | Loss | 0.253 | checkmate | 1-0 |
| gpt-5.2 | 6 (Legal Moves) | black | 55 | 0 | Loss | 0.309 | checkmate | 1-0 |
| gpt-5.2 | 6 (Legal Moves) | white | 63 | 0 | Loss | 0.354 | checkmate | 0-1 |
| gpt-5.2 | 6 (Legal Moves) | white | 73 | 0 | Loss | 0.411 | checkmate | 0-1 |
| gpt-5.2 | 6 (Legal Moves) | white | 26 | 0 | Loss | 0.146 | checkmate | 0-1 |
| gpt-5.2 | 7 (Piece Rules) | black | 26 | 0 | Loss | 0.146 | checkmate | 1-0 |
| gpt-5.2 | 7 (Piece Rules) | black | 25 | 0 | Loss | 0.141 | checkmate | 1-0 |
| gpt-5.2 | 7 (Piece Rules) | black | 34 | 0 | Loss | 0.191 | checkmate | 1-0 |
| gpt-5.2 | 7 (Piece Rules) | white | 34 | 0 | Loss | 0.191 | checkmate | 0-1 |
| gpt-5.2 | 7 (Piece Rules) | white | 47 | 0 | Loss | 0.264 | checkmate | 0-1 |
| gpt-5.2 | 7 (Piece Rules) | white | 35 | 0 | Loss | 0.197 | checkmate | 0-1 |
| gpt-5.2 | 8 (Path Tracing) | black | 48 | 3 | Loss | 0.270 | draw_claimable | * |
| gpt-5.2 | 8 (Path Tracing) | black | 30 | 0 | Loss | 0.169 | checkmate | 1-0 |
| gpt-5.2 | 8 (Path Tracing) | black | 18 | 0 | Loss | 0.101 | checkmate | 1-0 |
| gpt-5.2 | 8 (Path Tracing) | white | 30 | 1 | Win | 1.000 | checkmate | 1-0 |
| gpt-5.2 | 8 (Path Tracing) | white | 63 | 0 | Loss | 0.354 | checkmate | 0-1 |
| gpt-5.2 | 8 (Path Tracing) | white | 81 | 0 | Win | 1.000 | checkmate | 1-0 |
