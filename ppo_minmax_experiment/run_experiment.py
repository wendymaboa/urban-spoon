"""
Run the full GPT-2 Minmax experiment: train baseline, train Minmax, evaluate, plot.

Usage:
    python run_experiment.py
    python run_experiment.py --steps 1000
    python run_experiment.py --steps 100 --skip-plots
    python run_experiment.py --only eval
    python run_experiment.py --only train --steps 500

All paths are relative to this folder (ppo_minmax_experiment/).
Part of the urban-spoon repo — Colab runs via notebooks/colab_gpt2_minmax_500steps.ipynb.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train both conditions, evaluate, and plot results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--steps", type=int, default=500, help="PPO training steps for both conditions.")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size.")
    parser.add_argument("--mini-batch", type=int, default=2, help="PPO mini-batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Training generation length.")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name.")
    parser.add_argument("--output-dir", default="results", help="Directory for logs and eval CSVs.")
    parser.add_argument("--learning-rate", type=float, default=1.41e-5, help="PPO learning rate.")
    parser.add_argument("--baseline-kl-coef", type=float, default=0.2, help="KL coef for baseline.")
    parser.add_argument("--minmax-kl-coef", type=float, default=0.01, help="KL anchor for Minmax.")
    parser.add_argument("--safety-threshold", type=float, default=-0.3, help="Minmax unsafe reward threshold.")
    parser.add_argument("--penalty-floor", type=float, default=-1.0, help="Minmax penalty floor.")

    # Evaluation
    parser.add_argument("--eval-max-new-tokens", type=int, default=64, help="Eval generation length.")
    parser.add_argument("--harm-threshold", type=float, default=0.3, help="Detoxify harm threshold.")
    parser.add_argument("--eval-limit", type=int, default=700, help="Number of eval prompts.")

    # Stage control
    parser.add_argument(
        "--only",
        choices=["all", "train", "eval", "plot"],
        default="all",
        help="Run a subset of stages.",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline training.")
    parser.add_argument("--skip-minmax", action="store_true", help="Skip Minmax training.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip BeaverTails evaluation.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")

    return parser.parse_args()


def run_stage(name, script_rel, extra_args):
    script = ROOT / script_rel
    cmd = [PYTHON, str(script), *extra_args]
    print(f"\n{'=' * 60}")
    print(f"START {name}")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED {name} (exit {result.returncode}, {elapsed / 60:.1f} min)")
        sys.exit(result.returncode)

    print(f"\nDONE {name} ({elapsed / 60:.1f} min)")


def training_args(args):
    return [
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--mini-batch", str(args.mini_batch),
        "--max-new-tokens", str(args.max_new_tokens),
        "--model", args.model,
        "--output-dir", args.output_dir,
        "--learning-rate", str(args.learning_rate),
    ]


def main():
    args = parse_args()

    run_train = args.only in ("all", "train")
    run_eval = args.only in ("all", "eval") and not args.skip_eval
    run_plot = args.only in ("all", "plot") and not args.skip_plots

    print(f"Experiment root: {ROOT}")
    print(f"Steps: {args.steps} | Batch: {args.batch_size} | Model: {args.model}")

    if run_train:
        if not args.skip_baseline:
            run_stage(
                "baseline training",
                "Train/train_baseline.py",
                training_args(args) + ["--kl-coef", str(args.baseline_kl_coef)],
            )
        if not args.skip_minmax:
            run_stage(
                "Minmax training",
                "Train/train_minmax_gpt2.py",
                training_args(args) + [
                    "--kl-coef", str(args.minmax_kl_coef),
                    "--safety-threshold", str(args.safety_threshold),
                    "--penalty-floor", str(args.penalty_floor),
                ],
            )

    if run_eval:
        run_stage(
            "BeaverTails evaluation",
            "Evaluation/eval_beavertails.py",
            [
                "--output-dir", args.output_dir,
                "--max-new-tokens", str(args.eval_max_new_tokens),
                "--harm-threshold", str(args.harm_threshold),
                "--eval-limit", str(args.eval_limit),
            ],
        )

    if run_plot:
        run_stage("plot results", "Evaluation/plot_results.py", [])

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
