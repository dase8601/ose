"""
abm/loop_mpc_crafter_run33.py — Run 33: Curiosity-driven goal selection (no manager, no RL)

Run 32 conclusion: REINFORCE is the fundamental bottleneck. Four runs confirmed
that sparse achievement reward cannot drive a manager toward prerequisite ordering,
regardless of horizon, intrinsic reward, or codebook quality.

This run drops REINFORCE entirely. Goal selection is replaced with curiosity:
at each goal-selection step, sample M=32 candidate z embeddings from replay and
pick the one MOST DIFFERENT from the current state (maximum cosine distance).

Intuition: if the agent is in a tier1/2 state (wood, stone), the most visually
different state in the replay is likely a crafting/tier3 state — because crafted
items look unlike raw environment observations. The agent perpetually aims for
the most novel-looking state it has in memory.

No manager policy. No REINFORCE. No codebook. No achievement reward signal.
Pure self-supervised exploration via curiosity-selected goals + CEM planning.

Key differences from Run 32:
  - No SubgoalManager, no REINFORCE, no codebook
  - Goal selection: argmax cosine_distance(z_cur, replay_sample[k]) over M=32 candidates
  - H_GOAL = 150 (re-select goal every 150 primitive steps, same cadence as H_MANAGER)
  - N_CANDIDATES = 32 (candidates sampled per goal selection)
  - Encoder stays frozen in ACT (CEM stability)

Condition:   lewm_crafter_curiosity
Loop module: abm.loop_mpc_crafter_run33
RunPod:
  pip install timm crafter scikit-learn wandb moviepy
  python abm_experiment.py --loop-module abm.loop_mpc_crafter_run33 \\
    --condition lewm_crafter_curiosity --device cuda --env crafter \\
    --steps 600000 --n-envs 8
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .cem_planner import CEMPlanner
from .crafter_env import ACHIEVEMENTS, ACHIEVEMENT_TIERS, make_crafter_env, make_crafter_vec_env
from .world_model import Predictor, sigreg

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE        = 64
Z_DIM           = 256
N_ACTIONS       = 17
N_ENVS          = 8
REPLAY_CAP      = 30_000
GOAL_BUF_CAP    = 2_000
TRAIN_FREQ      = 16
BATCH_SIZE      = 256
SIGREG_LAMBDA   = 0.05
PRED_LR         = 3e-4
OBSERVE_DEFAULT = 300_000
EP_MAX_STEPS    = 1_000
EVAL_INTERVAL   = 10_000
EVAL_N_EPS      = 10
CEM_SAMPLES     = 512
CEM_ELITES      = 50
CEM_ITERS       = 10
TRAIN_WARMUP    = BATCH_SIZE
H_GOAL          = 150    # primitive steps between goal re-selections
N_CANDIDATES    = 32     # replay samples evaluated per goal selection
VIDEO_LOG_INTERVAL = 50_000


# ── ViT-Tiny encoder ───────────────────────────────────────────────────────────

class ViTTinyEncoder(nn.Module):
    def __init__(self, img_size: int = IMG_SIZE, z_dim: int = Z_DIM):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False,
            img_size=img_size, num_classes=0,
        )
        self.proj = nn.Linear(192, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.vit(x))


# ── Pixel replay buffer ────────────────────────────────────────────────────────

class PixelReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE):
        self.capacity  = capacity
        self._obs_t    = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._obs_next = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._ptr      = 0
        self._size     = 0

    def push(self, obs_t, action, obs_next):
        self._obs_t[self._ptr]    = obs_t
        self._actions[self._ptr]  = action
        self._obs_next[self._ptr] = obs_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n, device):
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        def to_t(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return (to_t(self._obs_t[idx]),
                torch.from_numpy(self._actions[idx]).long().to(device),
                to_t(self._obs_next[idx]))

    def sample_raw(self, n):
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return self._obs_t[idx]

    def __len__(self):
        return self._size


# ── Goal buffer ────────────────────────────────────────────────────────────────

class GoalPixelBuffer:
    def __init__(self, capacity: int = GOAL_BUF_CAP):
        self._buf: deque = deque(maxlen=capacity)

    def add(self, pix_hwc):
        self._buf.append(pix_hwc.copy())

    def sample_raw(self, n=1):
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self):
        return len(self._buf)


# ── Pixel helpers ──────────────────────────────────────────────────────────────

def _extract_pix(obs):
    if isinstance(obs, dict) and "image" in obs:
        return obs["image"]
    return obs

def _pix_batch_to_tensor(pix_np, device):
    return torch.from_numpy(pix_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)

def _pix_to_tensor(pix, device):
    return torch.from_numpy(pix.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


# ── Curiosity goal selection ───────────────────────────────────────────────────

def _select_curious_goal(
    z_cur: torch.Tensor,
    replay: PixelReplayBuffer,
    goal_buf: GoalPixelBuffer,
    encoder: nn.Module,
    device: str,
    n_candidates: int = N_CANDIDATES,
) -> Optional[torch.Tensor]:
    """
    Select the goal most different from z_cur among N_CANDIDATES replay samples.
    70% chance to draw candidates from goal_buf (achievement-positive),
    30% from replay. Among candidates, pick argmax cosine_distance to z_cur.

    Intuition: the most visually different state from current is likely a higher-tier
    state — crafted items look unlike raw environment observations.
    """
    use_goal = len(goal_buf) >= n_candidates // 2 and random.random() < 0.7
    raw = goal_buf.sample_raw(n_candidates) if use_goal else replay.sample_raw(n_candidates)
    if raw is None:
        raw = replay.sample_raw(n_candidates)
    if raw is None:
        return None

    with torch.no_grad():
        cand_z = encoder(_pix_batch_to_tensor(raw, device))        # (N, Z_DIM)
        cos_sim = F.cosine_similarity(
            z_cur.expand(len(cand_z), -1), cand_z, dim=-1
        )  # (N,) — similarity to current state
        # Pick most DISSIMILAR candidate (highest novelty)
        best = cos_sim.argmin().item()
    return cand_z[best:best+1]   # (1, Z_DIM)


# ── World model training ───────────────────────────────────────────────────────

def _train_step(encoder, predictor, opt, replay, device):
    if len(replay) < TRAIN_WARMUP:
        return None, None
    obs_t, actions, obs_next = replay.sample(BATCH_SIZE, device)
    z_t    = encoder(obs_t)
    z_next = encoder(obs_next).detach()
    a_oh   = F.one_hot(actions, N_ACTIONS).float()
    z_pred = predictor(z_t, a_oh)
    pred_loss = F.mse_loss(z_pred, z_next)
    reg_loss  = sigreg(z_t)
    loss = pred_loss + SIGREG_LAMBDA * reg_loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
    )
    opt.step()
    return pred_loss.item(), reg_loss.item()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_crafter(
    encoder, predictor, goal_buf, replay, device,
    replay_size=0, seed_offset=1000, n_eps=EVAL_N_EPS,
):
    if replay_size < TRAIN_WARMUP:
        return 0.0, {}

    was_enc  = encoder.training
    was_pred = predictor.training
    encoder.eval(); predictor.eval()

    mpc = CEMPlanner(
        predictor, n_actions=N_ACTIONS, horizon=5,
        n_samples=256, n_elites=32, n_iters=5,
        device=device, distance="cosine",
    )
    ever_unlocked = {k: 0 for k in ACHIEVEMENTS}

    for ep in range(n_eps):
        env = make_crafter_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_steps, goal_steps = False, 0, H_GOAL
        z_goal = None

        while not done and ep_steps < EP_MAX_STEPS:
            pix = _extract_pix(obs)
            with torch.no_grad():
                z_cur = encoder(_pix_to_tensor(pix, device))

            if goal_steps >= H_GOAL or z_goal is None:
                z_goal = _select_curious_goal(z_cur, replay, goal_buf, encoder, device)
                goal_steps = 0

            action = (mpc.plan_single(z_cur, z_goal)
                      if z_goal is not None else env.action_space.sample())
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_steps += 1
            goal_steps += 1

            for k, v in info.get("achievements", {}).items():
                if v and k in ever_unlocked:
                    ever_unlocked[k] = 1
            if reward > 0:
                goal_buf.add(_extract_pix(obs))

        env.close()

    score    = sum(ever_unlocked.values()) / len(ACHIEVEMENTS)
    per_tier = {
        tier: sum(ever_unlocked.get(a, 0) for a in ach_list) / len(ach_list)
        for tier, ach_list in ACHIEVEMENT_TIERS.items()
    }
    if was_enc:  encoder.train(); predictor.train()
    return score, per_tier


# ── Video recording ────────────────────────────────────────────────────────────

def _record_episode(encoder, predictor, goal_buf, replay, device, seed=7777):
    was_enc  = encoder.training
    was_pred = predictor.training
    encoder.eval(); predictor.eval()

    mpc = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=5,
                     n_samples=256, n_elites=32, n_iters=5,
                     device=device, distance="cosine")
    env = make_crafter_env(seed=seed)
    obs, _ = env.reset(seed=seed)

    frames, done, steps, goal_steps = [], False, 0, H_GOAL
    z_goal = None
    while not done and steps < EP_MAX_STEPS:
        pix = _extract_pix(obs)
        frames.append(pix.copy())
        with torch.no_grad():
            z_cur = encoder(_pix_to_tensor(pix, device))
        if goal_steps >= H_GOAL or z_goal is None:
            z_goal = _select_curious_goal(z_cur, replay, goal_buf, encoder, device)
            goal_steps = 0
        action = mpc.plan_single(z_cur, z_goal) if z_goal is not None else env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc; steps += 1; goal_steps += 1

    env.close()
    if was_enc:  encoder.train(); predictor.train()
    return np.stack(frames).transpose(0, 3, 1, 2)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_crafter_run33_loop(
    condition: str = "lewm_crafter_curiosity",
    device: str = "cuda",
    max_steps: int = 600_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "crafter",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "lewm_crafter_curiosity":
        raise ValueError(f"loop_mpc_crafter_run33 supports: lewm_crafter_curiosity — got: {condition}")
    if env_type != "crafter":
        raise ValueError("loop_mpc_crafter_run33 only supports env_type='crafter'.")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else OBSERVE_DEFAULT

    logger.info(
        f"[RUN33] Curiosity Crafter | ViT-Tiny 64×64 | cosine CEM | "
        f"H_goal={H_GOAL} n_candidates={N_CANDIDATES} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    if _wandb is not None:
        _wandb.init(
            project="lewm-crafter",
            name=f"run33-{condition}",
            config={
                "z_dim": Z_DIM, "h_goal": H_GOAL, "n_candidates": N_CANDIDATES,
                "sigreg_lambda": SIGREG_LAMBDA, "cem_distance": "cosine",
                "observe_steps": _observe_steps, "max_steps": max_steps,
                "n_envs": n_envs, "goal_selection": "curiosity_max_cos_dist",
            },
        )
        logger.info("[RUN33] wandb initialized")

    replay   = PixelReplayBuffer(REPLAY_CAP)
    goal_buf = GoalPixelBuffer(GOAL_BUF_CAP)

    encoder   = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor = Predictor(latent_dim=Z_DIM, n_actions=N_ACTIONS, hidden=512).to(device)
    opt = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=PRED_LR
    )

    # Per-env goal state
    goal_steps:    np.ndarray = np.full(n_envs, H_GOAL, dtype=np.int32)
    active_goal_z: List[Optional[torch.Tensor]] = [None] * n_envs

    envs = make_crafter_vec_env(n_envs, seed=seed, use_async=False)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    metrics: Dict = {
        "env_step": [], "crafter_score": [], "per_tier": [],
        "pred_loss_ewa": [], "sigreg_loss_ewa": [],
        "mode": [], "wall_time_s": [],
    }
    pred_ewa = sigreg_ewa = None
    env_step = act_steps = total_observe_steps = 0
    frozen = False
    best_score = 0.0

    while env_step < max_steps:
        in_observe = env_step < _observe_steps

        # ── OBSERVE → ACT transition ──
        if not in_observe and not frozen:
            for p in encoder.parameters():  p.requires_grad_(False)
            for p in predictor.parameters(): p.requires_grad_(False)
            encoder.eval(); predictor.eval()
            frozen = True
            logger.info(
                f"[RUN33] OBSERVE→ACT at step={env_step} | "
                f"encoder+predictor frozen | replay={len(replay)} goal_buf={len(goal_buf)}"
            )

        pix_cur_np = _extract_pix(obs)

        # ── Actions ──
        if in_observe:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                z_cur_t = encoder(_pix_batch_to_tensor(pix_cur_np, device))

            # Curiosity goal selection per env
            for i in range(n_envs):
                if active_goal_z[i] is None or goal_steps[i] >= H_GOAL:
                    active_goal_z[i] = _select_curious_goal(
                        z_cur_t[i:i+1], replay, goal_buf, encoder, device
                    )
                    goal_steps[i] = 0

            valid   = [i for i in range(n_envs) if active_goal_z[i] is not None]
            actions = np.random.randint(N_ACTIONS, size=n_envs)

            if valid:
                mpc = CEMPlanner(
                    predictor, n_actions=N_ACTIONS, horizon=5,
                    n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                    device=device, distance="cosine",
                )
                z_batch   = z_cur_t[valid]
                z_g_batch = torch.cat([active_goal_z[i] for i in valid], dim=0)
                actions[valid] = mpc.plan_batch(z_batch, z_g_batch)

        # ── Step ──
        obs_next, rewards, terms, truncs, infos = envs.step(actions)
        dones    = terms | truncs
        env_step += n_envs
        if in_observe:
            total_observe_steps += n_envs
        else:
            act_steps += n_envs

        pix_next_np = _extract_pix(obs_next)

        for i in range(n_envs):
            pix_next_i = pix_cur_np[i] if dones[i] else pix_next_np[i]
            replay.push(pix_cur_np[i], int(actions[i]), pix_next_i)
            if rewards[i] > 0:
                goal_buf.add(pix_next_i)
            if not in_observe:
                goal_steps[i] += 1
                if dones[i]:
                    active_goal_z[i] = None
                    goal_steps[i]    = H_GOAL

        obs = obs_next

        # ── Train world model (OBSERVE only) ──
        if in_observe and (env_step // n_envs) % TRAIN_FREQ == 0:
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa   = pl if pred_ewa   is None else 0.95 * pred_ewa   + 0.05 * pl
                sigreg_ewa = rl if sigreg_ewa is None else 0.95 * sigreg_ewa + 0.05 * rl

        # ── Heartbeat ──
        if (env_step // n_envs) % 2000 == 0 and env_step > 0:
            mode_str = "OBSERVE" if in_observe else "ACT"
            logger.info(
                f"[RUN33] step={env_step:7d} | {mode_str:7s} | "
                f"replay={len(replay)} goal_buf={len(goal_buf)} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | "
                f"{time.time()-t0:.0f}s"
            )

        # ── Eval ──
        if env_step % EVAL_INTERVAL < n_envs:
            mode_str = "OBSERVE" if in_observe else "ACT"
            score, per_tier = _eval_crafter(
                encoder, predictor, goal_buf, replay, device,
                replay_size=len(replay), seed_offset=9000 + env_step,
            )
            elapsed  = time.time() - t0
            tier_str = " ".join(f"{k.replace('tier','t')}={v:.0%}" for k, v in per_tier.items())
            logger.info(
                f"[RUN33] step={env_step:7d} | mode={mode_str:7s} | "
                f"score={score:.1%} | {tier_str} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["crafter_score"].append(score)
            metrics["per_tier"].append(per_tier)
            metrics["pred_loss_ewa"].append(pred_ewa or 0.0)
            metrics["sigreg_loss_ewa"].append(sigreg_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            if score > best_score:
                best_score = score
                logger.info(f"[RUN33] *** new best score={score:.1%} at step {env_step} ***")

            if _wandb is not None:
                wb = {
                    "crafter_score":   score,
                    "pred_loss_ewa":   pred_ewa or 0.0,
                    "sigreg_loss_ewa": sigreg_ewa or 0.0,
                    "replay_size":     len(replay),
                    "goal_buf_size":   len(goal_buf),
                    "mode":            0 if in_observe else 1,
                }
                wb.update({f"tier/{k}": v for k, v in per_tier.items()})
                _wandb.log(wb, step=env_step)

            if _wandb is not None and env_step % VIDEO_LOG_INTERVAL < n_envs:
                try:
                    frames = _record_episode(encoder, predictor, goal_buf, replay, device,
                                             seed=7777 + env_step)
                    _wandb.log({"eval_video": _wandb.Video(
                        frames, fps=10, format="mp4",
                        caption=f"step={env_step} score={score:.1%}")}, step=env_step)
                except Exception as e:
                    logger.warning(f"[RUN33] Video skipped: {e}")

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/crafter_lewm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict()},
               ckpt_dir / f"checkpoint_{condition}.pt")
    logger.info(f"[RUN33] Done | best_score={best_score:.1%} | total_time={elapsed_total:.0f}s")
    if _wandb is not None:
        _wandb.finish()

    return {
        "condition":       condition,
        "env_type":        "crafter",
        "env_steps":       metrics["env_step"],
        "crafter_score":   metrics["crafter_score"],
        "per_tier":        metrics["per_tier"],
        "pred_loss_ewa":   metrics["pred_loss_ewa"],
        "sigreg_loss_ewa": metrics["sigreg_loss_ewa"],
        "mgr_loss_ewa":    [0.0] * len(metrics["env_step"]),
        "mode":            metrics["mode"],
        "wall_time_s":     metrics["wall_time_s"],
        "best_score":      best_score,
        "n_switches":      0,
        "switch_log":      [],
        "total_time_s":    elapsed_total,
        "act_steps":       act_steps,
        "observe_steps":   total_observe_steps,
    }


run_abm_loop = run_crafter_run33_loop
