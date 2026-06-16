"""
PLOTTING: Generate thesis figures from training logs and eval results.

Usage:
    python plot_results.py

Output:
    results/fig1_training_curves.png     — GPT-2 reward over steps, both conditions
    results/fig2_algorithm1_dynamics.png — GPT-2 v_min, v_max, r_unsafe
    results/fig3_harm_rate_bar.png       — eval harm rate, both models, by category
    results/fig4_overall_harm_bar.png    — overall harm rate, both models
"""

import csv
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

OUTPUT_DIR = 'results'


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def latest(pattern):
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


# ── Locate training logs (GPT-2 only) ─────────────────────────────────────────
gpt2_baseline_path = 'results/baseline_training_log.csv'
gpt2_minmax_path   = 'results/minmax_training_log.csv'

print(f"gpt2 baseline: {gpt2_baseline_path}")
print(f"gpt2 minmax:   {gpt2_minmax_path}")


def plot_reward_curves(baseline_log, minmax_log, title, out_path):
    b_steps   = [int(r['step'])         for r in baseline_log]
    b_rewards = [float(r['mean_reward']) for r in baseline_log]
    m_steps   = [int(r['step'])         for r in minmax_log]
    m_rewards = [float(r['mean_reward']) for r in minmax_log]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(b_steps, b_rewards, label='PPO + KL (baseline)', color='steelblue',  linewidth=1.5, alpha=0.8)
    ax.plot(m_steps, m_rewards, label='PPO + Minmax (ours)', color='darkorange', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Reward (−toxicity)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_alg1_dynamics(minmax_log, title, out_path):
    m_steps   = [int(r['step'])     for r in minmax_log]
    m_vmin    = [float(r['v_min'])    for r in minmax_log]
    m_vmax    = [float(r['v_max'])    for r in minmax_log]
    m_runsafe = [float(r['r_unsafe']) for r in minmax_log]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(m_steps, m_vmax,    label='V_MAX',    color='green',  linewidth=1.5)
    ax.plot(m_steps, m_vmin,    label='V_MIN',    color='red',    linewidth=1.5)
    ax.plot(m_steps, m_runsafe, label='R_unsafe', color='purple', linewidth=1.5, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Figure 1 / 1b: Training reward curves ─────────────────────────────────────
if os.path.exists(gpt2_baseline_path) and os.path.exists(gpt2_minmax_path):
    plot_reward_curves(
        load_csv(gpt2_baseline_path), load_csv(gpt2_minmax_path),
        'Training Reward Over Time — GPT-2',
        os.path.join(OUTPUT_DIR, 'fig1_training_curves.png'),
    )

# ── Figure 2: Algorithm 1 dynamics ──────────────────────────────────────────
if os.path.exists(gpt2_minmax_path):
    plot_alg1_dynamics(
        load_csv(gpt2_minmax_path),
        'Algorithm 1 Dynamics — GPT-2 (Minmax)',
        os.path.join(OUTPUT_DIR, 'fig2_algorithm1_dynamics.png'),
    )

# ── Figure 3: Evaluation harm rate by category, all 4 models ──────────────────
try:
    eval_summary = load_csv('results/eval_summary.csv')

    models_present = sorted({r['model'] for r in eval_summary})
    print(f"Models in eval_summary: {models_present}")

    # Two-way comparison: baseline vs Minmax
    spec = [
        ('baseline', 'PPO + KL (baseline)',  'steelblue'),
        ('minmax',   'PPO + Minmax (ours)',  'darkorange'),
    ]
    spec = [(m, label, color) for (m, label, color) in spec if m in models_present]

    if not spec:
        raise RuntimeError(f"No recognised models in eval_summary.csv: {models_present}")

    # Categories taken from the first model's rows
    first_model = spec[0][0]
    cat_rows = [r for r in eval_summary if r['model'] == first_model and r['category'] != '_overall']
    categories = [r['category'] for r in cat_rows]
    short_cats = [c.split(',')[0][:22] for c in categories]  # shorten compound labels

    n_models = len(spec)
    x = np.arange(len(categories))
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(15, 5.5))
    for i, (m, label, color) in enumerate(spec):
        rows = [r for r in eval_summary if r['model'] == m and r['category'] != '_overall']
        rows_by_cat = {r['category']: float(r['harm_rate']) for r in rows}
        harm = [rows_by_cat.get(c, 0.0) for c in categories]
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, harm, bar_width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(short_cats, rotation=35, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Harm Rate')
    ax.set_title('Harm Rate by Category — BeaverTails-Evaluation')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_harm_rate_bar.png'), dpi=150)
    plt.close()
    print("Saved fig3_harm_rate_bar.png")

    # ── Figure 4: Overall harm rate ──────────────────────────────────────────
    overall = {r['model']: float(r['harm_rate']) for r in eval_summary if r['category'] == '_overall'}
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [label for (_, label, _) in spec]
    values = [overall.get(m, 0.0) for (m, _, _) in spec]
    colors = [color for (_, _, color) in spec]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v*100:.2f}%",
                ha='center', va='bottom', fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Overall Harm Rate')
    ax.set_title('Overall Harm Rate — BeaverTails-Evaluation (700 prompts each)')
    ax.set_ylim(0, max(values) * 1.25)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_overall_harm_bar.png'), dpi=150)
    plt.close()
    print("Saved fig4_overall_harm_bar.png")

except FileNotFoundError:
    print("eval_summary.csv not found — run eval_beavertails.py first, then re-run this script.")