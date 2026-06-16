#!/usr/bin/env bash
#
# GPT-2 baseline (PPO + KL + Detoxify) — matches ppo_minmax_experiment TRL settings.
# Run on Colab GPU via notebooks/colab_gpt2_minmax_500steps.ipynb
#
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

ACTOR_MODEL_NAME_OR_PATH="${ACTOR_MODEL_NAME_OR_PATH:-gpt2}"
REWARD_MODEL_NAME_OR_PATH="${REWARD_MODEL_NAME_OR_PATH:-gpt2}"
REWARD_CRITIC_MODEL_NAME_OR_PATH="${REWARD_CRITIC_MODEL_NAME_OR_PATH:-gpt2}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/gpt2-baseline-500}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-500}"
# BeaverTails = TRL experiment parity. Use PKU-SafeRLHF/train for native PKU PPO.
TRAIN_DATASET="${TRAIN_DATASET:-BeaverTails}"
ZERO_STAGE="${ZERO_STAGE:-2}"
OFFLOAD="${OFFLOAD:-none}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
echo '*' >"${OUTPUT_DIR}/.gitignore"
cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [ -z "${WANDB_API_KEY:-}" ]; then
	export WANDB_MODE="offline"
fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed --num_gpus=1 \
	--module safe_rlhf.algorithms.ppo \
	--train_datasets "${TRAIN_DATASET}" \
	--actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--reward_model_name_or_path "${REWARD_MODEL_NAME_OR_PATH}" \
	--reward_critic_model_name_or_path "${REWARD_CRITIC_MODEL_NAME_OR_PATH}" \
	--use_detoxify_reward True \
	--use_minmax False \
	--safety_threshold -0.3 \
	--max_length 160 \
	--temperature 1.0 \
	--num_return_sequences 1 \
	--repetition_penalty 1.0 \
	--trust_remote_code True \
	--epochs 1 \
	--update_iters 1 \
	--max_training_steps "${MAX_TRAINING_STEPS}" \
	--per_device_prompt_batch_size 4 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 1 \
	--actor_lr 1.41e-5 \
	--actor_weight_decay 0.0 \
	--actor_lr_scheduler_type constant \
	--actor_lr_warmup_ratio 0.0 \
	--critic_lr 1.41e-5 \
	--critic_weight_decay 0.0 \
	--critic_lr_scheduler_type constant \
	--critic_lr_warmup_ratio 0.0 \
	--normalize_reward False \
	--seed 42 \
	--kl_coeff 0.2 \
	--clip_range_ratio 0.2 \
	--clip_range_score 50.0 \
	--clip_range_value 0.2 \
	--ptx_coeff 0.0 \
	--training_log_csv "${OUTPUT_DIR}/baseline_training_log.csv" \
	--output_dir "${OUTPUT_DIR}" \
	--log_type tensorboard \
	--log_dir "${OUTPUT_DIR}/runs" \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True
