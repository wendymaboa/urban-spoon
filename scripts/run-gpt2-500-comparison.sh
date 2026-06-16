#!/usr/bin/env bash
#
# Run both safe-rlhf GPT-2 conditions for the 500-step comparison:
#   1. Baseline — PPO + KL (0.2) + Detoxify
#   2. Minmax   — Algorithm 1 + KL anchor (0.01) + Detoxify
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-500}"

if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
	echo "========== GPT-2 baseline (safe-rlhf) — ${MAX_TRAINING_STEPS} steps =========="
	MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS}" bash "${SCRIPT_DIR}/ppo-gpt2-baseline.sh"
fi

if [[ "${SKIP_MINMAX:-0}" != "1" ]]; then
	echo ""
	echo "========== GPT-2 Minmax (safe-rlhf) — ${MAX_TRAINING_STEPS} steps =========="
	MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS}" bash "${SCRIPT_DIR}/ppo-gpt2-minmax.sh"
fi

echo ""
echo "Done. Logs:"
echo "  output/gpt2-baseline-500/baseline_training_log.csv"
echo "  output/gpt2-minmax-500/minmax_training_log.csv"
