# Scaffolding Ablation Experiment: Analysis Report

## Experiment Overview

- **Total games**: 162 across 3 models and 9 scaffold levels
- **Models tested**: Claude Opus 4.6, Gemini 3 Pro Preview, GPT-5.2
- **Scaffold levels**: 0 (Baseline) through 8 (Path Tracing)
- **Opponent**: Stockfish at Skill Level 1 with 100ms/move (~1418 Elo; see calibration note below)
- **Colors**: Each model played as both White and Black
- **Epochs**: 3 per (model, level, color) combination
- **Overall record**: 31 wins, 2 draws, 129 losses (19.1% win rate)

**Stockfish strength calibration**: We calibrated the effective Elo of our
Stockfish configuration (Skill Level 1, 100ms/move) by playing 300 games against
reference Stockfish instances at known UCI_Elo ratings (1320-1570, step 50).
Reference engines used `UCI_LimitStrength=true` with 2s/move, approximating the
TC 60+0.6 conditions under which UCI_Elo was calibrated (Stockfish PR #2225).
Maximum-likelihood estimation with the standard logistic Elo model yields
**~1418 Elo (95% CI [1377, 1459])**. This is consistent with the CCRL
calibration of ~1468 for Skill Level 1, with the small shortfall attributable
to our 100ms time limit versus the 120+1.0s used in Stockfish's latest
calibration. In practical terms, this is roughly low-intermediate club player
strength.

**Sample size caveat**: With 6 games per (model, level) cell, individual cell-level
results carry substantial uncertainty. Standard errors are reported throughout.
Trends across multiple levels and models are more reliable than any single cell.

## Key Findings

### 1. Scaffolding Extends Game Survival

Mean game length (in full moves) increased with scaffolding for all models,
though the magnitude and consistency varied:

- **Gemini 3 Pro**: Baseline 16.0 +/- 1.5 moves, peak 62.5 +/- 9.8 at level 6
  (Legal Moves) -- a roughly +46 move improvement.
- **GPT-5.2**: Baseline 45.2 +/- 8.9, peak 61.7 +/- 6.6 at level 4 (Board Image)
  -- a roughly +17 move improvement.
- **Claude Opus 4.6**: Baseline 21.3 +/- 2.8, peak 34.5 +/- 2.6 at level 4
  (Board Image) -- a roughly +13 move improvement.

Game length serves as a proxy for the model's ability to maintain a coherent
understanding of the board state over successive turns. The improvement from
baseline to peak scaffolding is substantial, though the relationship is not
monotonic -- some models show dips at intermediate levels.

### 2. Invalid Move Reduction

The most consistent effect of scaffolding is on invalid move rates:

- **Claude Opus 4.6**: Invalid moves drop to zero at level 5 (Structured
  Response) and remain there through level 8. Total invalid moves: 114.
- **Gemini 3 Pro**: Invalid moves drop to near-zero at level 5. Total: 180.
- **GPT-5.2**: Invalid moves drop to zero at level 6 (Legal Moves). Total: 182.

This suggests that at low scaffold levels, the primary bottleneck for LLM chess
performance is **move legality**, not strategy. Models appear to have plausible
strategic intuitions but fail to translate them into valid board actions without
explicit support.

### 3. Invalid Move Taxonomy

The breakdown of invalid move types across all 476 invalid moves (all models
combined) reveals the nature of spatial reasoning failures:

| Category | Count (all models) | Interpretation |
|----------|-------------------|----------------|
| moved_pinned_piece | 262 (55%) | Model moves a piece pinned to its king -- fails to trace attack lines |
| no_such_piece | 158 (33%) | Model references a piece on a square where none exists |
| ignored_check | 28 (6%) | Model doesn't recognize it is in check |
| unparseable | 13 (3%) | Response format failure -- not a spatial reasoning error |
| no_move_line | 9 (2%) | Failed to produce a MOVE: line -- instruction following failure |
| ambiguous | 4 (<1%) | Ambiguous piece reference |
| api_error | 2 (<1%) | API/network failure |

The dominance of **moved_pinned_piece** (262 instances) is notable. A pin
requires reasoning about three objects simultaneously: the attacking piece, the
pinned piece, and the king behind it. This appears to be a multi-hop spatial
inference task that current LLMs struggle with consistently.

**Example -- Claude Opus 4.6, Level 4 (playing White):**

> "...I should capture the rook immediately with Nxd8, winning the exchange.
> After Nxd8, I'll have a rook, two bishops, and a knight vs Black's queen,
> bishop, knight - the material will be much closer and my pieces are very
> active around Black's exposed king."
> **MOVE: Nxd8**
>
> *System: "Your move 'Nxd8' was invalid (illegal in the current position)."*

The strategic reasoning (capture material, evaluate the resulting position) is
sound, but the model fails to notice the knight is pinned and cannot legally move.

**Example -- Gemini 3 Pro, Level 2 (playing White, confused about pin geometry):**

> "My rook on f2 is pinned to the king on g1 by the rook on h6? No, the h6
> rook pins the knight on h2 to the king on g1 via diagonal? No. The rook on
> h6 is on the h-file. King on g1 is on the g-file. So no pin."

The model attempts explicit spatial reasoning about whether a pin exists and
produces internally contradictory conclusions. It identifies the relevant pieces
but fails to resolve the geometric relationship between them.

The distribution of invalid move categories differs by model:
- **Claude Opus 4.6**: Dominated by moved_pinned_piece (82) and ignored_check (17)
- **Gemini 3 Pro**: Dominated by no_such_piece (149) -- hallucinating pieces on
  empty squares
- **GPT-5.2**: Dominated by moved_pinned_piece (155)

This suggests different models may have different failure modes in spatial
reasoning, though with this sample size these patterns should be interpreted
cautiously.

### 4. When Do Invalid Moves Occur?

Analysing the 161 invalid move events in conversation-mode games (levels 2-8),
we can track *when* in the game invalid moves happen. The pattern is striking:

| Phase | Invalid Moves | Total LLM Turns | Rate |
|-------|--------------|-----------------|------|
| Opening (moves 1-10) | 13 | 1,247 | 1.0% |
| Midgame (moves 11-30) | 86 | 1,223 | 7.0% |
| Endgame (moves 31+) | 62 | 177 | 35.0% |

More granularly, the rate increases monotonically through the game:

| Moves | Rate | |
|-------|------|---|
| 1-5 | 0.0% | Zero invalid moves across 630 LLM turns |
| 6-10 | 2.1% | |
| 11-15 | 3.1% | |
| 16-20 | 5.9% | |
| 21-25 | 14.6% | |
| 26-30 | 12.0% | |
| 31-35 | 18.3% | |
| 36-40 | 28.6% | |
| 41-45 | 53.3% | (small sample: 8/15 turns) |

The first invalid move in a game occurred at move 15.2 on average for Claude
Opus 4.6, 23.1 for Gemini 3 Pro, and 22.5 for GPT-5.2.

**Interpretation**: The opening-phase accuracy (0% invalid rate in the first 5
moves, 1% through move 10) is consistent with models having learned common
opening patterns from training data. Standard chess openings are heavily
represented in training corpora, and the first several moves of most games
follow well-known sequences where the legal moves are relatively constrained
and the "obvious" move is usually legal.

As games progress into the midgame and especially the endgame, positions become
increasingly unique and diverge from training data. The board state grows more
complex (or, in endgames, more sparse with unusual piece configurations), and
the model's spatial reasoning must rely increasingly on genuine board
understanding rather than pattern matching from training data.

There is a caveat: **selection bias** in the endgame figures. Games that reach
move 40+ at scaffold levels where invalid moves occur are disproportionately
games where the model was already struggling. Games with strong scaffolding
(levels 5-8) have near-zero invalid moves at all phases, so the high endgame
rate is driven by games at levels 2-4 that happen to survive long enough.
Despite this, the monotonic increase through the midgame (where selection bias
is less severe) is a robust pattern.

### 5. Game Outcomes

| Model | Wins | Win Rate | Best Level Win Rate |
|-------|------|----------|-------------------|
| Gemini 3 Pro | 25/54 | 46.3% | 83.3% (Level 4) |
| GPT-5.2 | 6/54 | 11.1% | 33.3% (Levels 0, 1, 3, 4, 8) |
| Claude Opus 4.6 | 0/54 | 0.0% | -- |

Gemini 3 Pro was the only model to win consistently. Its wins appeared at
level 2 and above, suggesting that conversation history was a prerequisite
for sustained competitive play.

Claude Opus 4.6 never won despite surviving longer with scaffolding. This
illustrates that game length and strategic competence are related but distinct
-- a model can make legal moves for longer without ever gaining a winning
position. However, with only 6 games per cell, it is possible that Claude
could occasionally win with more trials.

GPT-5.2's wins were scattered across levels with no clear trend, which may
reflect the small sample size rather than a meaningful pattern.

**Example -- Gemini 3 Pro, Level 5 (playing Black, delivering checkmate):**

> "My Advantage: Mate in 1. My Queen on e3 and Bishop on d4 coordinate against
> the White King on h2. The square g1 is the weak point. The Bishop on d4
> covers g1. The Queen on e3 can move to g1. King cannot capture Queen
> (covered by Bishop). King cannot move to g2 (occupied by Bishop). King
> cannot move to h3 (occupied by Pawn)."
> **MOVE: Qg1#**

This demonstrates that with sufficient scaffolding, models can identify and
verify checkmate patterns, though such precision was inconsistent.

### 6. White vs Black Asymmetry

| Model | White Win Rate | Black Win Rate |
|-------|---------------|----------------|
| Claude Opus 4.6 | 0.0% | 0.0% |
| Gemini 3 Pro | 33.3% | 59.3% |
| GPT-5.2 | 18.5% | 3.7% |
| **Overall** | **14/81 (17.3%)** | **17/81 (21.0%)** |

An interesting pattern: Gemini 3 Pro performed substantially better as Black
(59.3% vs 33.3%), while GPT-5.2 showed the opposite pattern (3.7% vs 18.5%).

Possible explanations for the overall slight Black advantage:
- Stockfish at level 1 may play weaker openings as White that are easier
  to exploit
- Responding to moves may be easier than initiating (reactive vs. generative
  reasoning)
- Training data biases toward annotated responses to common openings

However, the model-level patterns are contradictory, so no strong conclusion
can be drawn about a general White vs. Black advantage.

### 7. Termination Reasons

| Reason | Count | % |
|--------|-------|---|
| Checkmate | 123 | 75.9% |
| Invalid move failure | 22 | 13.6% |
| Draw (claimable) | 15 | 9.3% |
| Stalemate | 2 | 1.2% |

At levels 0-1, 66.7% and 55.6% of games ended via invalid move failure (model
exhausted its retry budget). From level 2 onward, this dropped to 0%. The shift
is largely attributable to conversation history and structured response formats
helping models produce parseable, legal moves.

The 15 "draw (claimable)" games represent positions where threefold repetition
was reachable -- typically indicating strategic stagnation where neither side
could make progress. These appeared across all scaffold levels, suggesting they
reflect a strategic rather than tactical limitation.

### 8. Diminishing Returns at High Scaffold Levels

The marginal effect of scaffolding was inconsistent at levels 7-8:

- **Claude Opus 4.6**: Game length remained essentially flat from level 5 onward
  (range 25.5 to 27.5), suggesting a ceiling around ~27 moves regardless of
  additional scaffolding.
- **Gemini 3 Pro**: Showed a non-monotonic pattern with a dip at level 7
  (Piece Rules, -14.8 moves) followed by a recovery at level 8 (+11.5 moves).
- **GPT-5.2**: Also showed a dip at level 7 (-14.2 moves) with partial recovery
  at level 8 (+11.5 moves).

The parallel dip at level 7 for both Gemini and GPT-5.2 is suggestive of an
information overload effect from the piece rules, but could also be a
coincidence given the sample size. Possible explanations include:
1. Too many instructions creating conflicting signals
2. Verbose prompts consuming context window, reducing reasoning capacity
3. A shift from tactical errors (which scaffolding addresses) to strategic
   weakness (which scaffolding addresses less directly)

## Qualitative Analysis: Model Reasoning

### When Models Get It Wrong

**Confident strategic reasoning, illegal execution** is the most common failure
pattern. Models articulate plausible plans but propose moves that violate board
constraints they haven't verified:

> **GPT-5.2**, Level 4 (playing White):
> **MOVE: Bxe7**
>
> *System: "Your move 'Bxe7' was invalid (illegal in the current position)."*

> **Claude Opus 4.6**, Level 4 (playing Black):
> **MOVE: Bxf2+**
>
> *System: "Your move 'Bxf2+' was invalid (illegal in the current position)."*

In both cases, the models attempted to capture with a bishop that was either
pinned or not on the expected square -- demonstrating the gap between
*strategic plausibility* and *spatial accuracy*.

**Confused spatial reasoning** occurs when models attempt to reason explicitly
about board geometry:

> **Gemini 3 Pro**, Level 2 (playing White, move 50 of a 52-move game):
> "Analysis: Black's king escapes to d6. My rook on f2 is pinned to the king
> on g1 by the rook on h6? No, the h6 rook pins the knight on h2 to the king
> on g1 via diagonal? No. The rook on h6 is on the h-file. King on g1 is on
> the g-file. So no pin."

The model cycles through multiple hypotheses about pin geometry, reaching
contradictory conclusions. This suggests that explicit spatial reasoning over
text representations is unreliable even when the model "knows" to check for pins.

### When Models Get It Right

At higher scaffold levels, models occasionally demonstrate genuine tactical
competence:

> **Gemini 3 Pro**, Level 5 (playing Black, calculating a forced mate):
> "Mate in 1. My Queen on e3 and Bishop on d4 coordinate against the White
> King on h2. The square g1 is the weak point. The Bishop on d4 covers g1.
> The Queen on e3 can move to g1. King cannot capture Queen (covered by
> Bishop). King cannot move to g2 (occupied by Bishop). King cannot move
> to h3 (occupied by Pawn). King cannot move to g3 (occupied by Pawn)."
> **MOVE: Qg1#**

> **GPT-5.2**, Level 4 (playing White, converting a won endgame):
> "The pawn on d7 is ready to promote, and Black has no pieces to stop it.
> We just need to promote safely, preferably to a queen, and avoid any
> accidental stalemate."
> **MOVE: d8=Q+**

> **Gemini 3 Pro**, Level 3 (playing White, delivering a back-rank mate):
> "I have a forced checkmate in one move. The black king is trapped in the
> corner at a8. My bishop on d6 covers the escape square a7 and b8, while
> my rook on d7 covers b7. By playing Rxc8#, I capture the pinned bishop
> and deliver checkmate."
> **MOVE: Rxc8#**

These examples show that when the model's spatial representation happens to be
accurate, it can produce sophisticated tactical reasoning. The challenge is
consistency -- this level of accuracy was not reliable even at the highest
scaffold levels.

## AI Safety Implications

### Spatial Reasoning as a Capability Probe

Chess provides a controlled setting to evaluate an AI system's ability to:

1. **Maintain an accurate world model**: The board state is fully observable,
   deterministic, and unambiguous. Yet LLMs often fail to track piece
   positions accurately, especially under constraints like pins and checks.

2. **Perform multi-hop inference**: Detecting a pin requires tracing an attack
   line through the pinned piece to the king. This is loosely analogous to
   reasoning about causal chains in other domains (e.g., "if I take action A,
   and the environment responds with B, then constraint C may be violated").
   The analogy has limits -- chess is far simpler than most real-world planning
   domains -- but the consistent failure mode is informative.

3. **Distinguish valid from plausible actions**: Models generate moves that
   "look right" but violate board constraints. This pattern -- producing outputs
   that are superficially reasonable but mechanistically incorrect -- is
   potentially concerning in contexts where plausibility alone is insufficient.

### Scaffolding as Capability Elicitation

The experiment suggests that **external scaffolding can partially compensate
for intrinsic spatial reasoning limitations**:

- Providing the legal moves list (Level 6) eliminates invalid moves for most
  models. This is analogous to constrained decoding or action-space filtering.
- Structured response formats (Level 5) appear to help models decompose their
  reasoning into verifiable steps.
- Board visualization (Level 4) provides an alternative input modality that
  may bypass some text-only spatial reasoning limitations.

**Implication for safety evaluations**: If model competence depends heavily on
scaffolding, unscaffolded benchmarks may underestimate capability. Conversely,
scaffolded evaluations may overestimate the model's intrinsic reasoning ability.
Both perspectives matter for capability assessment.

### The Pin Detection Problem

The prevalence of `moved_pinned_piece` errors (262 instances, 55% of all
invalid moves) warrants specific attention. Detecting a pin requires a
relational inference chain:

1. Identifying the attacking piece
2. Tracing its line of attack
3. Recognizing that a friendly piece lies on this line
4. Recognizing that the king lies behind it
5. Concluding that the intermediate piece cannot move off the line

Models fail at this consistently, even when explicitly prompted to check for
pins (levels 5-8, structured response). This suggests the limitation may be
deeper than prompt engineering can address -- though we note that the structured
response format did eliminate invalid moves entirely for some models, likely
by routing through the legal moves check rather than improving pin detection
per se.

**Possible safety relevance**: The pin detection failure can be read as evidence
that current LLMs lack robust mechanisms for maintaining and querying relational
spatial structures. Whether this limitation extends to non-spatial relational
reasoning (e.g., reasoning about permission hierarchies, causal chains, or
constraint satisfaction) is an open question that merits further investigation.

### Scaffolding Fragility

A model that performs well only with scaffolding raises questions about
robustness:

- **Deployment risk**: Removing or degrading scaffolding could cause capability
  drops that are difficult to predict from scaffolded evaluations alone.
- **Sandbagging**: Large scaffolding-dependent performance gaps are *consistent
  with* genuine capability limitations, though they do not definitively rule out
  strategic underperformance. A model that cannot play legal chess moves without
  a legal-moves list is unlikely to be deliberately hiding chess ability.
- **Capability ceilings**: The diminishing returns at levels 7-8 suggest a
  natural ceiling on scaffolding-driven improvement. The existence and level of
  this ceiling may itself be informative about intrinsic model capability.

### Limitations and Caveats

- **Small sample size**: 6 games per (model, level) cell is insufficient for
  strong statistical claims. Many observed patterns could reflect noise.
- **Single opponent**: All games were against Stockfish Level 1. Results may
  not generalize to other opponents or difficulty levels.
- **Confounded scaffolding**: Each level adds features cumulatively, so it is
  not possible to isolate the contribution of any single feature. A factorial
  design would be needed for causal claims.
- **Chess specificity**: Chess is a narrow domain. The relationship between
  chess spatial reasoning and broader AI capabilities is analogical, not
  established.
- **Model versioning**: Results are specific to the model versions tested
  (February 2026). Future model updates may change the picture substantially.

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
(80 moves) represents an approximate long, competitive game.

| Model | Baseline CPS | Best CPS | Best Level |
|-------|-------------|----------|------------|
| Gemini 3 Pro | 0.238 +/- 0.139 | 0.858 +/- 0.130 | Level 4 (Board Image) |
| GPT-5.2 | 0.362 +/- 0.125 | 0.482 +/- 0.153 | Level 8 (Path Tracing) |
| Claude Opus 4.6 | 0.120 +/- 0.016 | 0.194 +/- 0.015 | Level 4 (Board Image) |

### Average Centipawn Loss (ACPL) and Accuracy

For each game, we replayed the LLM's moves through Stockfish and computed the
**Average Centipawn Loss** -- the mean evaluation drop per move compared to the
engine's assessment. Lower ACPL = better play. We also convert to an
**Accuracy %** using the standard polynomial approximation:

```
Accuracy = 103.40 - 0.382 * ACPL - 0.00217 * ACPL^2
```

| Model | Mean ACPL | Mean Accuracy |
|-------|-----------|--------------|
| Gemini 3 Pro | 48 cp | 78.1% |
| GPT-5.2 | 87 cp | 53.0% |
| Claude Opus 4.6 | 119 cp | 29.8% |

For rough context, a 1200-rated human typically averages ~150 cp ACPL, though
direct comparisons between engine and human ACPL distributions should be
treated with caution.

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

See `tables.md` for complete numerical results with standard errors.
