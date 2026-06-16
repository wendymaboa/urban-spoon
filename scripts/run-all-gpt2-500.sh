#!/usr/bin/env bash
#
# Run all four GPT-2 500-step conditions in one repo:
#   TRL:       baseline + Minmax  (ppo_minmax_experiment/)
#   safe-rlhf: baseline + Minmax  (DeepSpeed scripts/)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-500}"

cd "${ROOT_DIR}"

echo "=============================================="
echo " GPT-2 experiment — ${MAX_TRAINING_STEPS} steps each"
echo " Repo root: ${ROOT_DIR}"
echo "=============================================="

if [[ "${RUN_TRL:-1}" == "1" ]]; then
	echo ""
	echo "========== TRL (ppo_minmax_experiment) =========="
	TRL_ARGS=(--only train --steps "${MAX_TRAINING_STEPS}")
	[[ "${SKIP_TRL_BASELINE:-0}" == "1" ]] && TRL_ARGS+=(--skip-baseline)
	[[ "${SKIP_TRL_MINMAX:-0}" == "1" ]] && TRL_ARGS+=(--skip-minmax)
	python ppo_minmax_experiment/run_experiment.py "${TRL_ARGS[@]}"
fi

if [[ "${RUN_SAFE_RLHF:-1}" == "1" ]]; then
	echo ""
	echo "========== safe-rlhf (DeepSpeed) =========="
	export MAX_TRAINING_STEPS
	export SKIP_BASELINE="${SKIP_SAFE_RLHF_BASELINE:-0}"
	export SKIP_MINMAX="${SKIP_SAFE_RLHF_MINMAX:-0}"
	bash "${SCRIPT_DIR}/run-gpt2-500-comparison.sh"
fi

echo ""
echo "=============================================="
echo " All runs complete. Training logs:"
echo "   TRL baseline:      ppo_minmax_experiment/results/baseline_training_log.csv"
echo "   TRL Minmax:         ppo_minmax_experiment/results/minmax_training_log.csv"
echo "   safe-rlhf baseline: output/gpt2-baseline-500/baseline_training_log.csv"
echo "   safe-rlhf Minmax:   output/gpt2-minmax-500/minmax_training_log.csv"
echo "=============================================="
