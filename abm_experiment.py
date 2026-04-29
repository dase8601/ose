"""
abm_experiment.py — Run and compare all A-B-M conditions.

Paper 2 (PPO): autonomous | fixed | ppo_only
Paper 3 (MPC): autonomous | fixed | mpc_only | random

Usage:
    # MiniWorld Paper 3 (DINO-WM / MPC)
    python abm_experiment.py --all --device cuda --env miniworld --steps 200000
    python abm_experiment.py --condition autonomous --device cuda --env miniworld

    # DoorKey / Crafter (Paper 2, PPO)
    python abm_experiment.py --all --device cuda
    python abm_experiment.py --all --device cuda --env crafter --steps 1000000
"""

import argparse
import importlib
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Plotting config ────────────────────────────────────────────────────────
COLORS = {
    "autonomous": "#2ecc71",
    "fixed":      "#3498db",
    "ppo_only":   "#aaaaaa",
    "mpc_only":   "#e67e22",
    "random":     "#95a5a6",
}
LABELS = {
    "autonomous": "Autonomous System M\n(MPC + plateau-triggered)",
    "fixed":      "Fixed-schedule System M\n(MPC + every 10K steps)",
    "ppo_only":   "PPO baseline\n(raw pixels, no LeWM)",
    "mpc_only":   "MPC only\n(no world model, random shooting)",
    "random":     "Random baseline\n(no planning, no world model)",
}

# All valid conditions — Paper 2 uses ppo_only; Paper 3 uses mpc_only + random
ALL_CONDITIONS_P2  = ["autonomous", "fixed", "ppo_only"]
ALL_CONDITIONS_P3  = ["autonomous", "fixed", "mpc_only", "random"]

TIER_COLORS = {
    "tier1_basic":    "#27ae60",
    "tier2_tools":    "#2980b9",
    "tier3_advanced": "#e67e22",
    "tier4_hard":     "#c0392b",
}
TIER_LABELS = {
    "tier1_basic":    "Tier 1 — Basic survival",
    "tier2_tools":    "Tier 2 — First tools",
    "tier3_advanced": "Tier 3 — Advanced crafting",
    "tier4_hard":     "Tier 4 — Hard achievements",
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _smooth(x, w=5):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(results: dict, save_dir: Path, env_type: str) -> Path:
    """Success rate (DoorKey) or achievement score (Crafter) vs env steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_label = ("Achievement Score (fraction of 22 unlocked)"
               if env_type == "crafter" else
               "Success Rate (50-episode eval)")
    env_names = {"crafter": "Crafter", "doorkey": "MiniGrid-DoorKey-6x6",
                  "miniworld": "MiniWorld-MazeS3", "dmcontrol": "dm_control Walker-Walk",
                  "habitat": "Habitat PointNav"}
    env_name = env_names.get(env_type, env_type)
    if env_type == "crafter":
        target, target_label = 0.15, "15% score target"
    elif env_type == "miniworld":
        target, target_label = 0.50, "50% nav success target"
    elif env_type == "dmcontrol":
        target, target_label = 0.50, "50% avg reward target"
    elif env_type == "habitat":
        target, target_label = 0.30, "30% nav success target"
    else:
        target, target_label = 0.80, "80% target"

    for cond, data in results.items():
        steps = data["env_steps"]
        sr    = data["success_rate"]
        if not steps:
            continue
        color = COLORS.get(cond, "#888888")
        ax.plot(steps, sr, color=color, lw=2, label=LABELS.get(cond, cond), alpha=0.9)

        if cond == "autonomous":
            for sw in data.get("switch_log", []):
                ax.axvline(sw["env_step"], color=color, linestyle=":", lw=0.8, alpha=0.5)

    ax.axhline(target, color="red", linestyle="--", lw=1.2, label=target_label)
    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    subtitle = ("DINOv2 ViT-B/14 + CEM MPC (DINO-WM style)" if env_type in ("miniworld", "dmcontrol")
                 else "LeWM (1.5M params) + LSTM-PPO")
    ax.set_title(
        f"A-B-M Loop: Autonomous vs Fixed-Schedule Mode Switching\n"
        f"{env_name}  |  {subtitle}",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 0.5 if env_type == "crafter" else 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = save_dir / "learning_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Learning curves → {path}")
    return path


def plot_ssl_loss(results: dict, save_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False

    for cond in ("autonomous", "fixed"):
        data = results.get(cond)
        if data is None:
            continue
        steps  = data["env_steps"]
        losses = data["ssl_loss_ewa"]
        if not any(l > 0 for l in losses):
            continue
        ax.plot(steps, losses, color=COLORS[cond], lw=1.8,
                label=LABELS[cond].split("\n")[0])
        plotted = True

    if not plotted:
        plt.close(fig)
        return save_dir / "ssl_loss_curve.png"

    ax.set_xlabel("Environment Steps", fontsize=10)
    ax.set_ylabel("LeWM SSL Loss (EWA)", fontsize=10)
    ax.set_title("System A (LeWM) Training Progress", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = save_dir / "ssl_loss_curve.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SSL loss curve → {path}")
    return path


def plot_mode_switches(results: dict, save_dir: Path) -> Path:
    data = results.get("autonomous")
    if data is None or not data.get("switch_log"):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No mode switches recorded",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(save_dir / "mode_switches.png", dpi=120)
        plt.close(fig)
        return save_dir / "mode_switches.png"

    fig, ax = plt.subplots(figsize=(10, 2.5))
    max_step = max(data["env_steps"]) if data["env_steps"] else 1

    current_mode = "OBSERVE"
    prev_step = 0

    for sw in data["switch_log"]:
        color = "#2ecc71" if current_mode == "OBSERVE" else "#3498db"
        ax.barh(0, sw["env_step"] - prev_step, left=prev_step, height=0.6,
                color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        prev_step = sw["env_step"]
        current_mode = sw["to"]

    color = "#2ecc71" if current_mode == "OBSERVE" else "#3498db"
    ax.barh(0, max_step - prev_step, left=prev_step, height=0.6,
            color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

    obs_patch = mpatches.Patch(color="#2ecc71", label="OBSERVE (LeWM trains)")
    act_patch = mpatches.Patch(color="#3498db", label="ACT (PPO trains)")
    ax.legend(handles=[obs_patch, act_patch], fontsize=9, loc="upper right")
    ax.set_xlabel("Environment Steps", fontsize=10)
    ax.set_xlim(0, max_step)
    ax.set_yticks([])
    ax.set_title(
        f"Autonomous System M — Mode Timeline  ({len(data['switch_log'])} switches)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = save_dir / "mode_switches.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Mode switch timeline → {path}")
    return path


def plot_comparison_bar(results: dict, save_dir: Path, env_type: str) -> Path:
    """Bar chart: steps to reach 80% (DoorKey) or peak score (Crafter)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    conds = [c for c in ("autonomous", "fixed", "ppo_only", "mpc_only", "random")
             if c in results]

    if env_type == "crafter":
        # For Crafter, bar = peak achievement score
        vals   = [max(results[c]["success_rate"]) if results[c]["success_rate"] else 0
                  for c in conds]
        ylabel = "Peak Achievement Score"
        title  = "Peak Achievement Score by Condition"
        fmt    = lambda v: f"{v:.1%}"
        def bar_label(v, c): return fmt(v)
    else:
        vals   = []
        for c in conds:
            s = results[c].get("steps_to_80pct")
            vals.append(s if s is not None else
                        (results[c]["env_steps"][-1] if results[c]["env_steps"] else 0))
        ylabel = "Environment Steps to 80% Success Rate"
        title  = "Sample Efficiency Comparison\n(* = 80% not reached, showing max steps)"
        fmt    = lambda v: f"{int(v):,}"
        def bar_label(v, c):
            return fmt(v) if results[c].get("steps_to_80pct") else f">{fmt(v)}*"

    colors = [COLORS[c] for c in conds]
    labels = [LABELS[c].split("\n")[0] for c in conds]
    bars   = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)

    for bar, v, c in zip(bars, vals, conds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                bar_label(v, c),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    if env_type == "crafter":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = save_dir / "comparison_bar.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison bar → {path}")
    return path


def plot_crafter_tiers(results: dict, save_dir: Path) -> Path:
    """
    Crafter-only: per-tier achievement fraction for autonomous vs fixed.
    This is the key figure that shows whether autonomous System M front-loads
    OBSERVE time for harder tiers (the core Crafter hypothesis).
    """
    from abm.crafter_env import ACHIEVEMENT_TIERS

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    tiers = list(ACHIEVEMENT_TIERS.keys())

    for ax, cond in zip(axes, ("autonomous", "fixed")):
        data = results.get(cond)
        if data is None or not data.get("per_tier"):
            ax.text(0.5, 0.5, f"No data for {cond}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(LABELS[cond].split("\n")[0])
            continue

        # Use the final eval's per_tier dict
        final_tier = data["per_tier"][-1] if data["per_tier"] else {}
        vals   = [final_tier.get(t, 0) for t in tiers]
        colors = [TIER_COLORS[t] for t in tiers]
        labels = [TIER_LABELS[t] for t in tiers]

        bars = ax.bar(range(len(tiers)), vals, color=colors, alpha=0.85, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([TIER_LABELS[t].split("—")[0].strip() for t in tiers],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Fraction of Tier Achieved", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title(LABELS[cond].split("\n")[0], fontsize=11, color=COLORS[cond])
        ax.grid(axis="y", alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=TIER_LABELS[t]) for t in tiers]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Crafter: Achievement Progress by Tech-Tree Tier\n"
        "(Autonomous System M should unlock harder tiers earlier — "
        "more OBSERVE time when world model is most useful)",
        fontsize=10,
    )
    fig.tight_layout()
    path = save_dir / "crafter_tiers.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Crafter tier plot → {path}")
    return path


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _img_tag(path: Path, max_width: int = 900) -> str:
    import base64
    if not path.exists():
        return f"<p><em>{path.name} not generated</em></p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:{max_width}px; margin:8px; display:block;">'


def write_report(save_dir: Path, results: dict, plot_paths: dict,
                 env_type: str) -> Path:
    env_names  = {"crafter": "Crafter", "doorkey": "MiniGrid-DoorKey-6x6",
                   "miniworld": "MiniWorld-MazeS3", "dmcontrol": "dm_control Walker-Walk",
                   "habitat": "Habitat PointNav"}
    env_name   = env_names.get(env_type, env_type)
    if env_type == "crafter":
        metric_col = "Steps to 15% score"
    elif env_type == "miniworld":
        metric_col = "Steps to 50% nav success"
    elif env_type == "dmcontrol":
        metric_col = "Avg reward (0-1)"
    elif env_type == "habitat":
        metric_col = "Steps to 30% nav success"
    else:
        metric_col = "Steps to 80%"

    rows = ""
    for cond in ("autonomous", "fixed", "ppo_only", "mpc_only", "random"):
        d = results.get(cond)
        if d is None:
            continue
        s80      = d.get("steps_to_80pct")
        sr80     = f"{s80:,}" if s80 else "not reached"
        n_sw     = d.get("n_switches", 0)
        t        = d.get("total_time_s", 0)
        peak     = max(d["success_rate"]) if d["success_rate"] else 0
        a_steps  = d.get("act_steps", 0)
        o_steps  = d.get("observe_steps", 0)
        total_s  = a_steps + o_steps
        act_pct  = f"{a_steps/total_s:.0%}" if total_s > 0 else "—"
        label    = LABELS.get(cond, cond).replace(chr(10), ' ')
        rows += f"""
        <tr>
          <td><b>{label}</b></td>
          <td>{sr80}</td>
          <td>{peak:.1%}</td>
          <td>{n_sw}</td>
          <td>{a_steps:,} ({act_pct})</td>
          <td>{t:.0f}s</td>
        </tr>"""

    tier_section = ""
    if env_type == "crafter" and "crafter_tiers" in plot_paths:
        tier_section = f"""
  <div class="card">
    <h2>Figure 5 — Achievement Progress by Tech-Tree Tier</h2>
    <p>Key test of the Crafter hypothesis: autonomous System M should allocate
    more OBSERVE time to harder tiers, resulting in higher achievement fractions
    for tier 3 and 4 relative to fixed-schedule.</p>
    <figure>{_img_tag(plot_paths["crafter_tiers"], 1000)}</figure>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>A-B-M Experiment — {env_name}</title>
  <style>
    body   {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding:28px; background:#f4f6f8; color:#222; }}
    h1     {{ color:#2c3e50; }}
    h2     {{ color:#34495e; border-bottom:1px solid #ddd; padding-bottom:6px; }}
    .card  {{ background:#fff; padding:20px 28px; border-radius:8px; margin:18px 0; box-shadow:0 1px 5px rgba(0,0,0,.12); }}
    table  {{ border-collapse:collapse; width:100%; }}
    th, td {{ padding:10px 16px; border-bottom:1px solid #e0e0e0; text-align:left; }}
    th     {{ background:#3498db; color:#fff; }}
    tr:nth-child(even) {{ background:#f9f9f9; }}
    figure {{ margin:0; text-align:center; }}
    figcaption {{ font-size:0.85em; color:#555; margin-top:4px; }}
  </style>
</head>
<body>
  <h1>A-B-M Experiment: Autonomous vs Fixed-Schedule Mode Switching</h1>
  <p>
    Empirical test of the Dupoux/LeCun/Malik A-B-M architecture (arXiv 2603.15381)
    on <b>{env_name}</b>.
    {"System A = DINOv2 ViT-B/14 (frozen) + action-conditioned world model predictor. System B = CEM MPC (DINO-WM style — Yann LeCun: abandon RL, use MPC)." if env_type in ("miniworld", "dmcontrol") else "System A = LeWM (~1.5M params, trains from pixels). System B = LSTM-PPO."}
    System M = autonomous plateau-detect FSM vs fixed-schedule timer.
  </p>
  <div class="card">
    <h2>Results Summary</h2>
    <table>
      <tr><th>Condition</th><th>{metric_col}</th><th>Peak score</th><th>Mode switches</th><th>ACT steps (% of budget)</th><th>Wall time</th></tr>
      {rows}
    </table>
  </div>
  <div class="card">
    <h2>Figure 1 — Learning Curves</h2>
    <figure>{_img_tag(plot_paths.get("learning_curves", save_dir/"learning_curves.png"), 950)}</figure>
  </div>
  <div class="card">
    <h2>Figure 2 — Mode Switch Timeline (Autonomous)</h2>
    <figure>{_img_tag(plot_paths.get("mode_switches", save_dir/"mode_switches.png"), 950)}</figure>
  </div>
  <div class="card">
    <h2>Figure 3 — SSL Loss (LeWM training)</h2>
    <figure>{_img_tag(plot_paths.get("ssl_loss", save_dir/"ssl_loss_curve.png"), 800)}</figure>
  </div>
  <div class="card">
    <h2>Figure 4 — Performance Bar</h2>
    <figure>{_img_tag(plot_paths.get("bar", save_dir/"comparison_bar.png"), 650)}</figure>
  </div>
  {tier_section}
</body>
</html>"""

    path = save_dir / "report.html"
    path.write_text(html)
    logger.info(f"HTML report → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A-B-M mode-switching experiment")
    parser.add_argument("--condition",
                        choices=["autonomous", "fixed", "ppo_only", "mpc_only", "random",
                                 "planner_only", "curiosity_observe", "her_goals",
                                 "subgoals", "curiosity_her", "short_horizon", "scripted_seed",
                                 "protected_seed", "post_door_neg", "dinov2_frozen",
                                 "vjepa2_frozen", "vjepa2_symbolic", "vjepa2_adapter",
                                 "vjepa2_symbolic_scaled", "vjepa2_adapter_late_ebm",
                                 "symbolic_only", "vjepa2_symbolic_scaled_late_ebm",
                                 "symbolic_l2_stage3", "symbolic_bce_ebm",
                                 "symbolic_large_margin", "symbolic_two_phase_ebm",
                                 "symbolic_horizon12_s2", "symbolic_scripted_stage3",
                                 "symbolic_frozen_pred_stage3", "symbolic_ppo_stage3",
                                 "symbolic_short_horizon_s2", "symbolic_exact_goal_s2",
                                 "lewm_doorkey_pixels",
                                 "lewm_crafter_pixels"],
                        help="Single condition to run")
    parser.add_argument("--all",    action="store_true",
                        help="Run all conditions (Paper 3: autonomous+fixed+mpc_only+random for miniworld)")
    parser.add_argument("--device", default="auto",       help="auto | mps | cpu | cuda")
    parser.add_argument("--env",    default="doorkey",
                        choices=["doorkey", "crafter", "miniworld", "dmcontrol", "habitat"],
                        help="Environment: doorkey | crafter | miniworld | dmcontrol | habitat")
    parser.add_argument("--steps",  type=int, default=800_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--observe-steps", type=int, default=None,
                        help="Override initial OBSERVE steps (for MPC experiments)")
    parser.add_argument("--use-mpc", action="store_true",
                        help="Use MPC planning instead of PPO in ACT mode")
    parser.add_argument("--no-rl", action="store_true",
                        help="Disable RL (PPO) — MPC only, no policy gradient")
    parser.add_argument("--loop-module", default="abm.loop_ppo_lewm",
                        help="Python module exposing run_abm_loop, e.g. abm.loop_ppo_lewm, abm.loop_mpc_crafter, abm.loop_mpc_tiered_crafter, abm.loop_mpc_doorkey_run12, abm.loop_mpc_doorkey_run13")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    logger.info(f"Device: {args.device}  |  Env: {args.env}")

    loop_module = importlib.import_module(args.loop_module)
    run_abm_loop = loop_module.run_abm_loop

    save_dir = Path(f"results/{args.env}")
    save_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    if args.all:
        if args.env in ("miniworld", "dmcontrol", "habitat"):
            conditions = ALL_CONDITIONS_P3   # autonomous, fixed, mpc_only, random
        else:
            conditions = ALL_CONDITIONS_P2   # autonomous, fixed, ppo_only
    elif args.condition:
        conditions = [args.condition]
    else:
        parser.error("Specify --condition or --all")

    all_results = {}

    for cond in conditions:
        json_path = save_dir / f"metrics_{cond}.json"
        if json_path.exists():
            logger.info(f"Loading cached results for {cond} from {json_path}")
            with open(json_path) as f:
                all_results[cond] = json.load(f)
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Running condition: {cond.upper()}")
        logger.info(f"{'='*60}\n")

        result = run_abm_loop(
            condition     = cond,
            device        = args.device,
            max_steps     = args.steps,
            seed          = args.seed,
            n_envs        = args.n_envs,
            env_type      = args.env,
            observe_steps = args.observe_steps,
            use_mpc       = args.use_mpc,
            use_rl        = not args.no_rl,
        )
        all_results[cond] = result

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved {cond} metrics → {json_path}")

    plot_paths = {}
    plot_paths["learning_curves"] = plot_learning_curves(all_results, save_dir, args.env)
    plot_paths["ssl_loss"]        = plot_ssl_loss(all_results, save_dir)
    plot_paths["mode_switches"]   = plot_mode_switches(all_results, save_dir)
    plot_paths["bar"]             = plot_comparison_bar(all_results, save_dir, args.env)

    if args.env == "crafter":
        plot_paths["crafter_tiers"] = plot_crafter_tiers(all_results, save_dir)

    write_report(save_dir, all_results, plot_paths, args.env)

    logger.info(f"\nAll outputs → {save_dir}/")

    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    for cond, data in all_results.items():
        s80  = data.get("steps_to_80pct")
        peak = max(data["success_rate"]) if data["success_rate"] else 0
        a_steps = data.get("act_steps", 0)
        o_steps = data.get("observe_steps", 0)
        total_s = a_steps + o_steps
        act_pct = f"{a_steps/total_s:.0%}" if total_s > 0 else "—"
        metric_label = "steps_to_50" if args.env in ("miniworld", "dmcontrol") else "steps_to_80"
        logger.info(
            f"  {cond:12s}: {metric_label}={s80 or 'N/A':>7}  |  "
            f"peak={peak:.1%}  |  switches={data.get('n_switches', 0)}  |  "
            f"act_steps={a_steps:,} ({act_pct})"
        )

    # Disabled for long runs: avoid auto-push failures interrupting result handling.
    # push_results_to_github(save_dir, args.env, args.condition)


def push_results_to_github(save_dir: Path, env_type: str, condition: str = None):
    """Auto-commit and push results after each experiment run."""
    import subprocess
    import shutil
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path("tests") / f"{env_type}_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)

    for f in save_dir.glob("*"):
        if f.suffix in (".json", ".png", ".html"):
            shutil.copy2(f, test_dir / f.name)

    summary = {
        "env_type": env_type,
        "condition": condition or "all",
        "timestamp": timestamp,
        "files": [f.name for f in test_dir.iterdir()],
    }
    with open(test_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    try:
        subprocess.run(["git", "add", str(test_dir)], check=True)
        subprocess.run([
            "git", "commit", "-m",
            f"Test results: {env_type} {condition or 'all'} {timestamp}"
        ], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logger.info(f"Results pushed to GitHub: {test_dir}/")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to push results to GitHub: {e}")


if __name__ == "__main__":
    main()
