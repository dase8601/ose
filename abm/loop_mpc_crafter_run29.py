"""
abm/loop_mpc_crafter_run29.py — Run 29: LeWM on Crafter pixels

Same architecture as Run 28 (DoorKey), zero changes to the learning algorithm,
new environment. This validates that ViT-Tiny + online SIGReg + L2 CEM is not
overfitted to DoorKey's simple grid structure.

Key differences from Run 28 (DoorKey):
  - img_size=64 (Crafter native, no resize needed)
  - n_actions=17
  - No manual stages — goals sampled randomly from replay buffer
    (open-ended goal-conditioned exploration)
  - Achievement-positive obs added to goal_buf for biased sampling
  - Metric: Crafter achievement score (fraction of 22 unlocked)

Architecture (unchanged from Run 28):
  - ViT-Tiny encoder (timm, 5M, img_size=64, z_dim=256)
  - MLP predictor (Predictor from world_model.py, hidden=512)
  - SIGReg λ=0.05 (sigreg() from world_model.py)
  - CEMPlanner(distance="l2") from cem_planner.py
  - OBSERVE+ACT: 300k OBSERVE, 300k ACT

Condition:   lewm_crafter_pixels
Loop module: abm.loop_mpc_crafter_run29
RunPod:
  pip install timm crafter
  python abm_experiment.py --loop-module abm.loop_mpc_crafter_run29 \\
    --condition lewm_crafter_pixels --device cuda --env crafter \\
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

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE         = 64        # Crafter native resolution — no resize needed
Z_DIM            = 256
N_ACTIONS        = 17
N_ENVS           = 8
REPLAY_CAP       = 30_000   # 30k × 64 × 64 × 3 ≈ 368 MB uint8
GOAL_BUF_CAP     = 2_000    # achievement-positive observations
TRAIN_FREQ       = 16       # train every N vectorised steps
BATCH_SIZE       = 256
SIGREG_LAMBDA    = 0.05
PRED_LR          = 3e-4
OBSERVE_DEFAULT  = 300_000
EP_MAX_STEPS     = 1_000    # Crafter episodes can run long
EVAL_INTERVAL    = 10_000
EVAL_N_EPS       = 10
GOAL_SAMPLE_FREQ = 100      # steps per env before resampling goal
CEM_SAMPLES      = 512
CEM_ELITES       = 50
CEM_ITERS        = 10
TRAIN_WARMUP     = BATCH_SIZE


# ── ViT-Tiny encoder ───────────────────────────────────────────────────────────

class ViTTinyEncoder(nn.Module):
    """
    ViT-Tiny (timm) for 64×64 Crafter observations.
    64/16=4 → 4×4=16 spatial patches, 192-dim hidden → z_dim projection.
    """
    def __init__(self, img_size: int = IMG_SIZE, z_dim: int = Z_DIM):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,   # strip head → (B, 192)
        )
        self.proj = nn.Linear(192, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 [0,1] → (B, z_dim)"""
        return self.proj(self.vit(x))


# ── Pixel replay buffer ────────────────────────────────────────────────────────

class PixelReplayBuffer:
    """
    Circular replay buffer for Crafter pixel transitions.
    30k × 64 × 64 × 3 bytes ≈ 368 MB — manageable on A100.
    """
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE):
        self.capacity   = capacity
        self._obs_t     = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions   = np.zeros(capacity, dtype=np.int64)
        self._obs_next  = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._ptr       = 0
        self._size      = 0

    def push(self, obs_t: np.ndarray, action: int, obs_next: np.ndarray) -> None:
        self._obs_t[self._ptr]    = obs_t
        self._actions[self._ptr]  = action
        self._obs_next[self._ptr] = obs_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        def to_t(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return (to_t(self._obs_t[idx]),
                torch.from_numpy(self._actions[idx]).long().to(device),
                to_t(self._obs_next[idx]))

    def sample_raw(self, n: int) -> Optional[np.ndarray]:
        """Return n random raw pixel obs (H, W, 3) uint8 for goal sampling."""
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return self._obs_t[idx]

    def __len__(self) -> int:
        return self._size


# ── Goal buffer (flat, no bucketing — no manual stages for Crafter) ────────────

class GoalPixelBuffer:
    """
    Flat deque of raw pixel obs (H, W, 3) uint8 from achievement-positive steps.
    Used to bias goal sampling toward interesting states.
    Falls back to PixelReplayBuffer when empty.
    """
    def __init__(self, capacity: int = GOAL_BUF_CAP):
        self._buf: deque = deque(maxlen=capacity)

    def add(self, pix_hwc: np.ndarray) -> None:
        self._buf.append(pix_hwc.copy())

    def sample_raw(self, n: int = 1) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self) -> int:
        return len(self._buf)


# ── Pixel helpers ──────────────────────────────────────────────────────────────

def _extract_pix(obs) -> np.ndarray:
    """Extract (H, W, 3) or (n_envs, H, W, 3) uint8 from CrafterEnv obs dict."""
    if isinstance(obs, dict) and 'image' in obs:
        return obs['image']
    return obs


def _pix_batch_to_tensor(pix_np: np.ndarray, device: str) -> torch.Tensor:
    """(n_envs, H, W, 3) uint8 → (n_envs, 3, H, W) float32 [0,1]"""
    return torch.from_numpy(pix_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


def _pix_to_tensor(pix: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float32 [0,1]"""
    t = torch.from_numpy(pix.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def _sample_goal_z(
    goal_buf: GoalPixelBuffer,
    replay: PixelReplayBuffer,
    encoder: nn.Module,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Sample a goal z: prefer achievement-positive obs (goal_buf), fall back to replay.
    Returns (1, Z_DIM) or None if replay is empty.
    """
    # 70% from goal_buf if populated, 30% from replay (or 100% replay if goal_buf empty)
    use_goal = len(goal_buf) > 0 and random.random() < 0.7
    raw = goal_buf.sample_raw(1) if use_goal else replay.sample_raw(1)
    if raw is None:
        raw = replay.sample_raw(1)
    if raw is None:
        return None
    pix = raw[0]  # (H, W, 3)
    with torch.no_grad():
        return encoder(_pix_to_tensor(pix, device))  # (1, Z_DIM)


# ── Training ───────────────────────────────────────────────────────────────────

def _train_step(
    encoder: nn.Module,
    predictor: nn.Module,
    opt: torch.optim.Optimizer,
    replay: PixelReplayBuffer,
    device: str,
) -> Tuple[Optional[float], Optional[float]]:
    if len(replay) < TRAIN_WARMUP:
        return None, None

    obs_t, actions, obs_next = replay.sample(BATCH_SIZE, device)

    z_t    = encoder(obs_t)
    z_next = encoder(obs_next).detach()

    a_oh   = F.one_hot(actions, N_ACTIONS).float()
    z_pred = predictor(z_t, a_oh)

    pred_loss = F.mse_loss(z_pred, z_next)
    reg_loss  = sigreg(z_t)
    loss      = pred_loss + SIGREG_LAMBDA * reg_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
    )
    opt.step()

    return pred_loss.item(), reg_loss.item()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_run29(
    encoder: nn.Module,
    predictor: nn.Module,
    goal_buf: GoalPixelBuffer,
    replay: PixelReplayBuffer,
    device: str,
    replay_size: int = 0,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> Tuple[float, Dict]:
    """
    Eval with frozen encoder+predictor. L2 CEM toward random/achievement goals.
    Returns (achievement_score, per_tier_dict).
    """
    if replay_size < TRAIN_WARMUP:
        return 0.0, {}

    was_training = encoder.training
    encoder.eval(); predictor.eval()

    mpc = CEMPlanner(
        predictor, n_actions=N_ACTIONS, horizon=5,
        n_samples=256, n_elites=32, n_iters=5,   # cheaper for eval
        device=device, distance="l2",
    )

    ever_unlocked: Dict[str, int] = {k: 0 for k in ACHIEVEMENTS}

    for ep in range(n_eps):
        env = make_crafter_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_steps, goal_age = False, 0, GOAL_SAMPLE_FREQ

        z_goal = _sample_goal_z(goal_buf, replay, encoder, device)

        while not done and ep_steps < EP_MAX_STEPS:
            if goal_age >= GOAL_SAMPLE_FREQ or z_goal is None:
                z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
                goal_age = 0

            pix = _extract_pix(obs)
            z_cur = encoder(_pix_to_tensor(pix, device))

            action = (mpc.plan_single(z_cur, z_goal)
                      if z_goal is not None
                      else env.action_space.sample())

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc; ep_steps += 1; goal_age += 1

            for k, v in info.get("achievements", {}).items():
                if v and k in ever_unlocked:
                    ever_unlocked[k] = 1
            if reward > 0:
                goal_buf.add(_extract_pix(obs))
                z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
                goal_age = 0

        env.close()

    score = sum(ever_unlocked.values()) / len(ACHIEVEMENTS)
    per_tier = {
        tier: sum(ever_unlocked.get(a, 0) for a in ach_list) / len(ach_list)
        for tier, ach_list in ACHIEVEMENT_TIERS.items()
    }

    if was_training:
        encoder.train(); predictor.train()

    return score, per_tier


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_crafter_run29_loop(
    condition: str = "lewm_crafter_pixels",
    device: str = "cuda",
    max_steps: int = 600_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "crafter",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "lewm_crafter_pixels":
        raise ValueError(f"loop_mpc_crafter_run29 supports: lewm_crafter_pixels — got: {condition}")
    if env_type != "crafter":
        raise ValueError("loop_mpc_crafter_run29 only supports env_type='crafter'.")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else OBSERVE_DEFAULT

    logger.info(
        f"[RUN29] LeWM Crafter pixels | ViT-Tiny 64×64 | L2 CEM | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    # ── Buffers ──
    replay   = PixelReplayBuffer(REPLAY_CAP)
    goal_buf = GoalPixelBuffer(GOAL_BUF_CAP)

    # ── Model ──
    encoder   = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor = Predictor(latent_dim=Z_DIM, n_actions=N_ACTIONS, hidden=512).to(device)
    opt = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=PRED_LR
    )

    mpc: Optional[CEMPlanner] = None

    # ── Vectorised envs ──
    envs = make_crafter_vec_env(n_envs, seed=seed, use_async=False)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    # Per-env tracking
    goal_ages = np.full(n_envs, GOAL_SAMPLE_FREQ, dtype=np.int32)  # force first refresh
    active_goal_z: List[Optional[torch.Tensor]] = [None] * n_envs
    ep_achiev: List[Dict] = [{} for _ in range(n_envs)]

    metrics: Dict = {
        "env_step": [], "crafter_score": [], "per_tier": [],
        "pred_loss_ewa": [], "sigreg_loss_ewa": [], "mode": [], "wall_time_s": [],
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
            mpc = CEMPlanner(
                predictor, n_actions=N_ACTIONS, horizon=5,
                n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                device=device, distance="l2",
            )
            frozen = True
            logger.info(
                f"[RUN29] OBSERVE→ACT at step={env_step} | encoder+predictor frozen | "
                f"replay={len(replay)} goal_buf={len(goal_buf)}"
            )

        # ── Get current pixel obs ──
        pix_cur_np = _extract_pix(obs)  # (n_envs, 64, 64, 3)

        # ── Choose actions ──
        if in_observe:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                z_cur_t = encoder(_pix_batch_to_tensor(pix_cur_np, device))  # (n_envs, Z_DIM)

            actions = np.random.randint(N_ACTIONS, size=n_envs)

            has_goal = []
            for i in range(n_envs):
                if active_goal_z[i] is None or goal_ages[i] >= GOAL_SAMPLE_FREQ:
                    active_goal_z[i] = _sample_goal_z(goal_buf, replay, encoder, device)
                    goal_ages[i] = 0
                if active_goal_z[i] is not None:
                    has_goal.append(i)

            if has_goal and mpc is not None:
                z_batch  = z_cur_t[has_goal]
                z_g_batch = torch.cat([active_goal_z[i] for i in has_goal], dim=0)
                actions[has_goal] = mpc.plan_batch(z_batch, z_g_batch)

        # ── Step ──
        obs_next, rewards, terms, truncs, infos = envs.step(actions)
        dones = terms | truncs
        env_step += n_envs
        if in_observe:
            total_observe_steps += n_envs
        else:
            act_steps += n_envs

        pix_next_np = _extract_pix(obs_next)

        # ── Per-env post-step ──
        for i in range(n_envs):
            pix_next_i = pix_cur_np[i] if dones[i] else pix_next_np[i]
            replay.push(pix_cur_np[i], int(actions[i]), pix_next_i)

            if rewards[i] > 0:
                goal_buf.add(pix_next_i)
                if not in_observe:
                    active_goal_z[i] = _sample_goal_z(goal_buf, replay, encoder, device)
                    goal_ages[i] = 0

            if dones[i]:
                if not in_observe:
                    active_goal_z[i] = None
                    goal_ages[i] = GOAL_SAMPLE_FREQ
            elif not in_observe:
                goal_ages[i] += 1

        obs = obs_next

        # ── Train (OBSERVE only) ──
        if in_observe and (env_step // n_envs) % TRAIN_FREQ == 0:
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa   = pl if pred_ewa   is None else 0.95 * pred_ewa   + 0.05 * pl
                sigreg_ewa = rl if sigreg_ewa is None else 0.95 * sigreg_ewa + 0.05 * rl

        # ── Heartbeat ──
        if (env_step // n_envs) % 2000 == 0 and env_step > 0:
            mode_str = "OBSERVE" if in_observe else "ACT"
            logger.info(
                f"[RUN29] step={env_step:7d} | {mode_str:7s} | replay={len(replay)} goal_buf={len(goal_buf)} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | {time.time()-t0:.0f}s"
            )

        # ── Eval ──
        if env_step % EVAL_INTERVAL < n_envs:
            mode_str = "OBSERVE" if in_observe else "ACT"
            score, per_tier = _eval_run29(
                encoder, predictor, goal_buf, replay, device,
                replay_size=len(replay),
                seed_offset=9000 + env_step, n_eps=EVAL_N_EPS,
            )
            elapsed = time.time() - t0
            tier_str = " ".join(f"{k.replace('tier','t')}={v:.0%}" for k, v in per_tier.items())
            logger.info(
                f"[RUN29] step={env_step:7d} | mode={mode_str:7s} | "
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
                logger.info(f"[RUN29] *** new best score={score:.1%} at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/crafter_lewm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"condition": condition, "encoder": encoder.state_dict(),
         "predictor": predictor.state_dict()},
        ckpt_dir / f"checkpoint_{condition}.pt",
    )
    logger.info(f"[RUN29] Done | best_score={best_score:.1%} | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "crafter",
        "env_steps": metrics["env_step"],
        "crafter_score": metrics["crafter_score"],
        "per_tier": metrics["per_tier"],
        "pred_loss_ewa": metrics["pred_loss_ewa"],
        "sigreg_loss_ewa": metrics["sigreg_loss_ewa"],
        "mode": metrics["mode"], "wall_time_s": metrics["wall_time_s"],
        "best_score": best_score, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps,
        "observe_steps": total_observe_steps,
    }


run_abm_loop = run_crafter_run29_loop
