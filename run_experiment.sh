#!/usr/bin/env bash
#
# DEPRECATED: This script runs models sequentially. Use run_experiment.py
# instead, which runs all (model × scaffold_level) combinations in parallel
# via Inspect's eval() API:
#
#   python run_experiment.py \
#       --models openrouter/anthropic/claude-sonnet-4-20250514 openrouter/openai/gpt-4o \
#       --levels 0 1 2 3 4 5 6 7 8 \
#       --epochs 3 --max-tasks 8
#
# ---
#
# Legacy scaffolding ablation experiment runner (sequential).
#
# Sweeps scaffold levels × models × Stockfish levels. Each invocation adds
# --new-epochs new games per (model, sf_level, scaffold_level) cell.
# Existing epoch directories are detected automatically so the script is
# safe to re-run — it will pick up where it left off.
#
# Usage:
#   ./run_experiment.sh                       # 1 new epoch, defaults
#   ./run_experiment.sh --new-epochs 3        # add 3 epochs
#   ./run_experiment.sh --levels 0,1,2        # only levels 0-2
#   ./run_experiment.sh --models "openrouter/anthropic/claude-sonnet-4-20250514,openrouter/openai/gpt-4o"
#
set -euo pipefail

# ---- defaults ----
NEW_EPOCHS=1
SCAFFOLD_LEVELS="0,1,2,3,4,5,6,7,8"
SF_LEVELS="1"
STOCKFISH_PATH="/opt/homebrew/bin/stockfish"
MODELS="openrouter/anthropic/claude-sonnet-4-20250514"
BASE_DIR="experiment_logs"

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --new-epochs)   NEW_EPOCHS="$2"; shift 2 ;;
        --levels)       SCAFFOLD_LEVELS="$2"; shift 2 ;;
        --sf-levels)    SF_LEVELS="$2"; shift 2 ;;
        --sf-path)      STOCKFISH_PATH="$2"; shift 2 ;;
        --models)       MODELS="$2"; shift 2 ;;
        --base-dir)     BASE_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--new-epochs N] [--levels 0,1,...] [--sf-levels 1,5,...] [--models m1,m2] [--sf-path PATH] [--base-dir DIR]" >&2
            exit 1
            ;;
    esac
done

# ---- helpers ----
model_slug() {
    # openrouter/anthropic/claude-sonnet-4-20250514 -> anthropic--claude-sonnet-4-20250514
    echo "$1" | sed 's|^openrouter/||' | sed 's|/|--|g'
}

IFS=',' read -ra LEVEL_ARR <<< "$SCAFFOLD_LEVELS"
IFS=',' read -ra SF_ARR <<< "$SF_LEVELS"
IFS=',' read -ra MODEL_ARR <<< "$MODELS"

total_runs=0
skipped=0

echo "============================================="
echo "  Scaffolding Ablation Experiment"
echo "============================================="
echo "  Models:           ${MODEL_ARR[*]}"
echo "  Scaffold levels:  ${LEVEL_ARR[*]}"
echo "  Stockfish levels: ${SF_ARR[*]}"
echo "  New epochs:       $NEW_EPOCHS"
echo "  Base directory:   $BASE_DIR"
echo "============================================="
echo ""

for model in "${MODEL_ARR[@]}"; do
    slug=$(model_slug "$model")
    for sf_level in "${SF_ARR[@]}"; do
        for scaffold_level in "${LEVEL_ARR[@]}"; do
            epoch_parent="${BASE_DIR}/${slug}/sf_${sf_level}/level_${scaffold_level}"

            # count existing epochs
            existing=0
            if [[ -d "$epoch_parent" ]]; then
                existing=$(find "$epoch_parent" -maxdepth 1 -type d -name 'epoch_*' | wc -l | tr -d ' ')
            fi
            next_epoch=$((existing + 1))
            last_epoch=$((existing + NEW_EPOCHS))

            for epoch_num in $(seq "$next_epoch" "$last_epoch"); do
                log_dir="${epoch_parent}/epoch_${epoch_num}"
                echo "[run] model=${slug}  sf=${sf_level}  level=${scaffold_level}  epoch=${epoch_num}  -> ${log_dir}"

                inspect eval main_experiment.py \
                    --model "$model" \
                    -T "scaffold_level=${scaffold_level}" \
                    -T "stockfish_levels=[${sf_level}]" \
                    -T "stockfish_path=${STOCKFISH_PATH}" \
                    --epochs 1 \
                    --log-dir "$log_dir"

                total_runs=$((total_runs + 1))
            done
        done
    done
done

echo ""
echo "Done. Completed ${total_runs} eval runs."
