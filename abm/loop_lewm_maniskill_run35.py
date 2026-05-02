"""
abm/loop_lewm_maniskill_run35.py — Run 35: LeWM + Continuous CEM on FetchPickAndPlace-v3

First test of online SIGReg world model + CEM planning on continuous robot manipulation.
Uses gymnasium[mujoco] FetchPickAndPlace-v3 — MuJoCo 3.x bundles its own libEGL so
no system OpenGL/Vulkan is needed on headless RunPod CUDA pods.

Rendering history:
  ManiSkill3 → needs Vulkan (not available on RunPod)
  dm_control → needs system libOpenGL/libOSMesa (not available on RunPod)
  gymnasium[mujoco] → uses MuJoCo's bundled libEGL, works headless with MUJOCO_GL=egl

Architecture:
  - ViT-Tiny encoder trained online from 64×64 RGB pixels (same as Crafter runs)
  - MLP Predictor with 4-dim continuous action input (FetchPickAndPlace: dx,dy,dz,dg)
  - ContinuousCEMPlanner: Gaussian CEM over H=8 step sequences
  - SIGReg M=1024, λ=0.1 (LeWM paper values)
  - OBSERVE/ACT freeze protocol (same as Runs 29-34)

Goal buffer strategy:
  - reward > -0.1 (dense, near 0 = gripper near goal) → add to goal buffer
  - info["is_success"] == True → also add to goal buffer
  - 70% goal buffer / 30% replay mix

Condition:   lewm_maniskill_pickcube
Loop module: abm.loop_lewm_maniskill_run35

RunPod full run:
  pip install "gymnasium[mujoco]" gymnasium-robotics imageio
  MUJOCO_GL=egl python abm_experiment.py --loop-module abm.loop_lewm_maniskill_run35 \\
    --condition lewm_maniskill_pickcube --device cuda --env maniskill \\
    --steps 400000 --n-envs 4
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .cem_continuous import ContinuousCEMPlanner
from .world_model import Predictor, sigreg

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE        = 64
Z_DIM           = 256
REPLAY_CAP      = 50_000
GOAL_BUF_CAP    = 2_000
TRAIN_FREQ      = 16
BATCH_SIZE      = 256
SIGREG_LAMBDA   = 0.1
N_PROJ_SIGREG   = 1024
PRED_LR         = 3e-4
OBSERVE_DEFAULT = 200_000
EP_MAX_STEPS    = 50      # FetchPickAndPlace-v3 default episode length
EVAL_INTERVAL   = 10_000
EVAL_N_EPS      = 5
TRAIN_WARMUP    = BATCH_SIZE
# CEM (LeWM paper values)
CEM_H           = 8
CEM_K           = 300
CEM_ELITES      = 30
CEM_ITERS       = 30
# Goal buffer thresholds (dense reward in FetchPickAndPlace is distance-based,
# near 0 = at goal, negative = away; -0.1 is "close enough to store as goal")
REWARD_GOAL_THRESH = -0.1


# ── ViT-Tiny encoder (identical to Crafter runs) ───────────────────────────────

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


# ── Replay buffer (continuous actions stored as float32) ───────────────────────

class ContinuousReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE, a_dim: int = 4):
        self.capacity  = capacity
        self.a_dim     = a_dim
        self._obs_t    = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions  = np.zeros((capacity, a_dim), dtype=np.float32)
        self._obs_next = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._ptr      = 0
        self._size     = 0

    def push(self, obs_t: np.ndarray, action: np.ndarray, obs_next: np.ndarray):
        self._obs_t[self._ptr]    = obs_t
        self._actions[self._ptr]  = action
        self._obs_next[self._ptr] = obs_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, device: str):
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        def to_t(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return (
            to_t(self._obs_t[idx]),
            torch.from_numpy(self._actions[idx]).float().to(device),
            to_t(self._obs_next[idx]),
        )

    def sample_raw(self, n: int) -> Optional[np.ndarray]:
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return self._obs_t[idx]

    def __len__(self):
        return self._size


# ── Goal pixel buffer ──────────────────────────────────────────────────────────

class GoalPixelBuffer:
    def __init__(self, capacity: int = GOAL_BUF_CAP):
        self._buf: deque = deque(maxlen=capacity)

    def add(self, pix_hwc: np.ndarray):
        self._buf.append(pix_hwc.copy())

    def sample_raw(self, n: int = 1) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self):
        return len(self._buf)


# ── Pixel helpers ──────────────────────────────────────────────────────────────

def _pix_to_tensor(pix_hwc: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float tensor on device."""
    import cv2
    if pix_hwc.shape[0] != IMG_SIZE or pix_hwc.shape[1] != IMG_SIZE:
        pix_hwc = cv2.resize(pix_hwc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(pix_hwc.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


# ── gymnasium[mujoco] environment helpers ─────────────────────────────────────

def _make_single_env(seed: int = 42):
    """
    Create a FetchPickAndPlace-v3 env with RGB pixel rendering.
    Uses MuJoCo's native renderer (no system OpenGL needed when MUJOCO_GL=egl).
    """
    import gymnasium_robotics  # registers Fetch envs
    gymnasium_robotics.register_robotics_envs()
    import gymnasium as gym
    env = gym.make(
        "FetchPickAndPlace-v4",  # v3 deprecated
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=EP_MAX_STEPS,
    )
    return env


def _get_pixels(env) -> np.ndarray:
    """Render current frame as (IMG_SIZE, IMG_SIZE, 3) uint8."""
    import cv2
    frame = env.render()
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame.astype(np.uint8)
    if frame.shape[0] != IMG_SIZE or frame.shape[1] != IMG_SIZE:
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame


def _get_action_dim(env) -> int:
    return int(env.action_space.shape[0])


# ── World model training step ──────────────────────────────────────────────────

def _train_step(encoder, predictor, opt, replay: ContinuousReplayBuffer, device: str):
    if len(replay) < TRAIN_WARMUP:
        return None, None
    obs_t, actions, obs_next = replay.sample(BATCH_SIZE, device)
    z_t    = encoder(obs_t)
    z_next = encoder(obs_next).detach()
    z_pred    = predictor(z_t, actions)
    pred_loss = F.mse_loss(z_pred, z_next)
    reg_loss  = sigreg(z_t, n_proj=N_PROJ_SIGREG)
    loss = pred_loss + SIGREG_LAMBDA * reg_loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
    opt.step()
    return pred_loss.item(), reg_loss.item()


# ── Goal sampling ──────────────────────────────────────────────────────────────

def _sample_goal_z(
    goal_buf: GoalPixelBuffer,
    replay: ContinuousReplayBuffer,
    encoder: nn.Module,
    device: str,
) -> Optional[torch.Tensor]:
    use_goal = len(goal_buf) > 0 and random.random() < 0.7
    raw = goal_buf.sample_raw(1) if use_goal else replay.sample_raw(1)
    if raw is None:
        raw = replay.sample_raw(1)
    if raw is None:
        return None
    with torch.no_grad():
        return encoder(_pix_to_tensor(raw[0], device))


# ── Evaluation with video recording ───────────────────────────────────────────

def _eval_fetch(
    encoder: nn.Module,
    predictor,
    goal_buf: GoalPixelBuffer,
    replay: ContinuousReplayBuffer,
    a_dim: int,
    device: str,
    step: int,
    n_eps: int = EVAL_N_EPS,
    save_video: bool = True,
) -> dict:
    if len(replay) < TRAIN_WARMUP:
        return {"success_rate": 0.0, "near_rate": 0.0}

    encoder.eval()
    predictor.eval()

    planner = ContinuousCEMPlanner(
        predictor, a_dim=a_dim, H=CEM_H, K=CEM_K,
        n_elites=CEM_ELITES, n_iters=CEM_ITERS, device=device,
    )

    successes    = 0
    nears        = 0
    video_frames: List[np.ndarray] = []

    for ep in range(n_eps):
        env = _make_single_env(seed=1000 + ep)
        obs, _ = env.reset()
        planner.reset()

        rgb = _get_pixels(env)
        z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
        if z_goal is None:
            with torch.no_grad():
                z_goal = encoder(_pix_to_tensor(rgb, device))

        ep_success  = False
        ep_near     = False
        record      = save_video and ep == 0

        for _ in range(EP_MAX_STEPS):
            rgb = _get_pixels(env)
            if record:
                video_frames.append(rgb.copy())

            with torch.no_grad():
                z_cur = encoder(_pix_to_tensor(rgb, device))

            action = planner.plan(z_cur, z_goal)
            obs, reward, terminated, truncated, info = env.step(action)

            if info.get("is_success", False):
                ep_success = True
            if reward > REWARD_GOAL_THRESH:
                ep_near = True

            if terminated or truncated:
                break

        env.close()
        if ep_success:
            successes += 1
        if ep_near:
            nears += 1

    if save_video and video_frames:
        try:
            import imageio
            media_dir = Path("media")
            media_dir.mkdir(exist_ok=True)
            vid_path = media_dir / f"maniskill_run35_eval_step{step:07d}.mp4"
            imageio.mimsave(str(vid_path), video_frames, fps=20)
            logger.info(f"[RUN35] Saved eval video: {vid_path} ({len(video_frames)} frames)")
        except Exception as e:
            logger.warning(f"[RUN35] Video save failed: {e}")

    encoder.train()
    predictor.train()

    return {
        "success_rate": successes / n_eps,
        "near_rate":    nears     / n_eps,
    }


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_abm_loop(
    condition: str = "lewm_maniskill_pickcube",
    device: str = "cuda",
    max_steps: int = 400_000,
    seed: int = 42,
    n_envs: int = 4,
    env_type: str = "maniskill",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:

    observe_steps = observe_steps or OBSERVE_DEFAULT
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    logger.info(f"[RUN35] condition={condition} device={device} "
                f"observe={observe_steps} total={max_steps} n_envs={n_envs}")

    # ── Environments ────────────────────────────────────────────────────────────
    envs = [_make_single_env(seed=seed + i) for i in range(n_envs)]
    obs_list = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=seed + i)
        obs_list.append(obs)

    a_dim = _get_action_dim(envs[0])
    logger.info(f"[RUN35] FetchPickAndPlace-v4 | a_dim={a_dim}")

    # ── Model ───────────────────────────────────────────────────────────────────
    encoder   = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor = Predictor(latent_dim=Z_DIM, n_actions=a_dim, hidden=512).to(device)
    params    = list(encoder.parameters()) + list(predictor.parameters())
    opt       = optim.Adam(params, lr=PRED_LR)

    replay   = ContinuousReplayBuffer(capacity=REPLAY_CAP, img_size=IMG_SIZE, a_dim=a_dim)
    goal_buf = GoalPixelBuffer(capacity=GOAL_BUF_CAP)

    # ── Metrics ─────────────────────────────────────────────────────────────────
    results: Dict = {
        "condition":       condition,
        "env_type":        "fetch_picknplace",
        "env_steps":       [],
        "success_rate":    [],
        "near_rate":       [],
        "pred_loss_ewa":   [],
        "sigreg_loss_ewa": [],
        "mode":            [],
        "wall_time_s":     [],
        "best_success":    0.0,
        "total_time_s":    0.0,
        "act_steps":       0,
        "observe_steps":   observe_steps,
    }

    pred_ewa  = 0.0
    sig_ewa   = 0.0
    ewa_alpha = 0.01
    t_start   = time.time()
    phase     = "OBSERVE"
    planner: Optional[ContinuousCEMPlanner] = None

    global_step  = 0
    train_ticker = 0

    # Initial pixel batch — render after reset
    rgb_batch = np.stack([_get_pixels(env) for env in envs])  # (n_envs, H, W, 3)

    while global_step < max_steps:

        # ── Phase transition ────────────────────────────────────────────────────
        if phase == "OBSERVE" and global_step >= observe_steps:
            phase = "ACT"
            encoder.eval()
            predictor.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
            for p in predictor.parameters():
                p.requires_grad_(False)
            planner = ContinuousCEMPlanner(
                predictor, a_dim=a_dim, H=CEM_H, K=CEM_K,
                n_elites=CEM_ELITES, n_iters=CEM_ITERS, device=device,
            )
            logger.info(f"[RUN35] → ACT phase | pred_ewa={pred_ewa:.4f} sig_ewa={sig_ewa:.4f} "
                        f"replay={len(replay)} goal_buf={len(goal_buf)}")

        # ── Action selection ────────────────────────────────────────────────────
        if phase == "OBSERVE" or planner is None:
            actions_np = np.stack([envs[0].action_space.sample() for _ in range(n_envs)])
        else:
            z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
            if z_goal is None:
                actions_np = np.stack([envs[0].action_space.sample() for _ in range(n_envs)])
            else:
                actions_np = np.zeros((n_envs, a_dim), dtype=np.float32)
                for i in range(n_envs):
                    with torch.no_grad():
                        z_cur = encoder(_pix_to_tensor(rgb_batch[i], device))
                    actions_np[i] = planner.plan(z_cur, z_goal)

        # ── Environment step ────────────────────────────────────────────────────
        next_rgb = np.zeros_like(rgb_batch)
        for i in range(n_envs):
            obs, reward, terminated, truncated, info = envs[i].step(actions_np[i])
            pix_next = _get_pixels(envs[i])
            next_rgb[i] = pix_next

            replay.push(rgb_batch[i], actions_np[i], pix_next)
            if reward > REWARD_GOAL_THRESH or info.get("is_success", False):
                goal_buf.add(pix_next)

            if terminated or truncated:
                obs, _ = envs[i].reset()
                next_rgb[i] = _get_pixels(envs[i])
                if planner is not None:
                    planner.reset()

        global_step  += n_envs
        train_ticker += n_envs

        # ── Training (OBSERVE only) ─────────────────────────────────────────────
        if phase == "OBSERVE" and train_ticker >= TRAIN_FREQ:
            train_ticker = 0
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa = (1 - ewa_alpha) * pred_ewa + ewa_alpha * pl
                sig_ewa  = (1 - ewa_alpha) * sig_ewa  + ewa_alpha * rl

        rgb_batch = next_rgb

        # ── Evaluation & logging ────────────────────────────────────────────────
        if global_step % EVAL_INTERVAL < n_envs:
            metrics = _eval_fetch(
                encoder, predictor, goal_buf, replay, a_dim, device,
                step=global_step, save_video=True,
            )
            elapsed = time.time() - t_start
            results["env_steps"].append(global_step)
            results["success_rate"].append(metrics["success_rate"])
            results["near_rate"].append(metrics["near_rate"])
            results["pred_loss_ewa"].append(pred_ewa)
            results["sigreg_loss_ewa"].append(sig_ewa)
            results["mode"].append(phase)
            results["wall_time_s"].append(elapsed)

            if metrics["success_rate"] > results["best_success"]:
                results["best_success"] = metrics["success_rate"]

            logger.info(
                f"[RUN35] step={global_step:>7} | {phase:7} | "
                f"success={metrics['success_rate']:.1%} near={metrics['near_rate']:.1%} | "
                f"pred_ewa={pred_ewa:.4f} sig_ewa={sig_ewa:.4f} | "
                f"replay={len(replay)} goal={len(goal_buf)} | "
                f"t={elapsed:.0f}s"
            )

    for env in envs:
        env.close()
    results["total_time_s"] = time.time() - t_start
    results["act_steps"]    = max(0, global_step - observe_steps)
    logger.info(f"[RUN35] Done | best_success={results['best_success']:.1%} "
                f"total_t={results['total_time_s']:.0f}s")
    return results
