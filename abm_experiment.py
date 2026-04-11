"""
abm_experiment.py — Run and compare all three A-B-M conditions.

Usage:
    python abm_experiment.py --condition autonomous --device mps
    python abm_experiment.py --condition fixed      --device mps
    python abm_experiment.py --condition ppo_only   --device mps
    python abm_experiment.py --all                  --device mps
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from abm.loop import run_abm_loop

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
}
LABELS = {
    "autonomous": "Autonomous System M\n(plateau-triggered)",
    "fixed":      "Fixed-schedule System M\n(every 1000 steps)",
    "ppo_only":   "PPO baseline\n(raw pixels, no LeWM)",
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _smooth(x, w=5):
    """Simple moving-average smoothing."""
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(results: dict, save_dir: Path) -> Path:
    """
    Primary figure: episode success rate vs env steps for all conditions.
    Vertical dotted lines mark autonomous mode switches.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, data in results.items():
        steps = data["env_steps"]
        sr    = data["success_rate"]
        if not steps:
            continue

        color = COLORS[cond]
        label = LABELS[cond]
        ax.plot(steps, sr, color=color, lw=2, label=label, alpha=0.9)

        # Mode-switch tick marks for autonomous
        if cond == "autonomous":
            for sw in data.get("switch_log", []):
                ax.axvline(sw["env_step"], color=color, linestyle=":", lw=0.8, alpha=0.5)

    ax.axhline(0.80, color="red", linestyle="--", lw=1.2, label="80% target")
    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel("Success Rate (50-episode eval)", fontsize=11)
    ax.set_title(
        "A-B-M Loop: Autonomous vs Fixed-Schedule Mode Switching\n"
        "MiniGrid-DoorKey-6x6  |  LeWM (1.5M params) + PPO",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.05)
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
    """LeWM SSL loss during OBSERVE phases (autonomous + fixed)."""
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
        ax.plot(steps, losses, color=COLORS[cond], lw=1.8, label=LABELS[cond].split("\n")[0])
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
    """Timeline of mode switches for the autonomous condition."""
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

    # Last segment
    color = "#2ecc71" if current_mode == "OBSERVE" else "#3498db"
    ax.barh(0, max_step - prev_step, left=prev_step, height=0.6,
            color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

    obs_patch = mpatches.Patch(color="#2ecc71", label="OBSERVE (LeWM trains)")
    act_patch = mpatches.Patch(color="#3498db", label="ACT (PPO trains)")
    ax.legend(handles=[obs_patch, act_patch], fontsize=9, loc="upper right")
    ax.set_xlabel("Environment Steps", fontsize=10)
    ax.set_xlim(0, max_step)
    ax.set_yticks([])
    ax.set_title(f"Autonomous System M — Mode Timeline  ({len(data['switch_log'])} switches)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = save_dir / "mode_switches.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Mode switch timeline → {path}")
    return path


def plot_comparison_bar(results: dict, save_dir: Path) -> Path:
    """Bar chart of steps-to-80pct for each condition."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    conds = [c for c in ("autonomous", "fixed", "ppo_only") if c in results]
    steps = []
    for c in conds:
        s = results[c].get("steps_to_80pct")
        steps.append(s if s is not None else results[c]["env_steps"][-1] if results[c]["env_steps"] else 0)

    colors = [COLORS[c] for c in conds]
    labels = [LABELS[c].split("\n")[0] for c in conds]
    bars   = ax.bar(labels, steps, color=colors, alpha=0.85, width=0.5)

    for bar, s, c in zip(bars, steps, conds):
        txt = f"{s:,}" if results[c].get("steps_to_80pct") else f">{s:,}*"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                txt, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Environment Steps to 80% Success Rate", fontsize=10)
    ax.set_title("Sample Efficiency Comparison\n(* = 80% not reached, showing max steps)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = save_dir / "comparison_bar.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison bar → {path}")
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


def write_report(save_dir: Path, results: dict, plot_paths: dict) -> Path:
    rows = ""
    for cond in ("autonomous", "fixed", "ppo_only"):
        d = results.get(cond)
        if d is None:
            continue
        s80  = d.get("steps_to_80pct")
        sr80 = f"{s80:,}" if s80 else "not reached"
        n_sw = d.get("n_switches", 0)
        t    = d.get("total_time_s", 0)
        peak = max(d["success_rate"]) if d["success_rate"] else 0
        rows += f"""
        <tr>
          <td><b>{LABELS[cond].replace(chr(10),' ')}</b></td>
          <td>{sr80}</td>
          <td>{peak:.1%}</td>
          <td>{n_sw}</td>
          <td>{t:.0f}s</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>A-B-M Experiment — Autonomous vs Fixed Switching</title>
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
    on MiniGrid-DoorKey-6x6.  System A = LeWM (~1.5M params, trains from pixels).
    System B = PPO.  System M = autonomous plateau-detect FSM vs fixed-schedule timer.
  </p>
  <div class="card">
    <h2>Results Summary</h2>
    <table>
      <tr><th>Condition</th><th>Steps to 80%</th><th>Peak success</th><th>Mode switches</th><th>Wall time</th></tr>
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
    <h2>Figure 4 — Sample Efficiency Bar</h2>
    <figure>{_img_tag(plot_paths.get("bar", save_dir/"comparison_bar.png"), 650)}</figure>
  </div>
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
    parser.add_argument("--condition", choices=["autonomous", "fixed", "ppo_only"],
                        help="Single condition to run")
    parser.add_argument("--all",    action="store_true", help="Run all three conditions")
    parser.add_argument("--device", default="auto",       help="auto | mps | cpu | cuda")
    parser.add_argument("--steps",  type=int, default=400_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16,  help="parallel envs (16 for GPU, 4 for CPU/MPS)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    logger.info(f"Device: {args.device}")

    save_dir = Path("results/abm")
    save_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    if args.all:
        conditions = ["autonomous", "fixed", "ppo_only"]
    elif args.condition:
        conditions = [args.condition]
    else:
        parser.error("Specify --condition or --all")

    # ── Run conditions ──────────────────────────────────────────────────────
    all_results = {}

    # Load existing results so --all can be resumed
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
            condition  = cond,
            device     = args.device,
            max_steps  = args.steps,
            seed       = args.seed,
            n_envs     = args.n_envs,
        )
        all_results[cond] = result

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved {cond} metrics → {json_path}")

    # ── Generate plots ──────────────────────────────────────────────────────
    plot_paths = {}
    plot_paths["learning_curves"] = plot_learning_curves(all_results, save_dir)
    plot_paths["ssl_loss"]        = plot_ssl_loss(all_results, save_dir)
    plot_paths["mode_switches"]   = plot_mode_switches(all_results, save_dir)
    plot_paths["bar"]             = plot_comparison_bar(all_results, save_dir)

    write_report(save_dir, all_results, plot_paths)

    logger.info(f"\nAll outputs → {save_dir}/")

    # ── Print key result ────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    for cond, data in all_results.items():
        s80  = data.get("steps_to_80pct")
        peak = max(data["success_rate"]) if data["success_rate"] else 0
        logger.info(
            f"  {cond:12s}: steps_to_80={s80 or 'N/A':>7}  |  "
            f"peak={peak:.1%}  |  switches={data.get('n_switches', 0)}"
        )


if __name__ == "__main__":
    main()
