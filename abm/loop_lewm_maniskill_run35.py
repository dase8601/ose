"""
abm/loop_lewm_maniskill_run35.py — Run 35: LeWM + Continuous CEM on ManiSkill3 PickCube-v1

First test of online SIGReg world model + CEM planning on continuous robot manipulation.

Architecture:
  - ViT-Tiny encoder trained online from 64×64 RGB pixels (same as Crafter runs)
  - MLP Predictor with 6-dim continuous action input (arm_pd_ee_delta_pose)
  - ContinuousCEMPlanner: Gaussian CEM over H=8 step sequences
  - SIGReg M=1024, λ=0.1 (LeWM paper values)
  - OBSERVE/ACT freeze protocol (same as Runs 29-34)

Goal buffer strategy:
  - During OBSERVE: store observations where cube height > 0.05m (lifted) or
    info['success']=True — these are the states we want to reach
  - 70% goal buffer / 30% replay mix (same as Crafter)

Key difference from Crafter: actions are continuous 6-dim ∈ [-1, 1] rather
than discrete one-hot. The MLP Predictor treats the 6-dim action as a raw
float input (n_actions=6).

Condition:   lewm_maniskill_pickcube
Loop module: abm.loop_lewm_maniskill_run35

Local smoke test (CPU, ~3-5 min):
  python abm_experiment.py --loop-module abm.loop_lewm_maniskill_run35 \\
    --condition lewm_maniskill_pickcube --device cpu --env maniskill \\
    --steps 2000 --n-envs 1

RunPod full run (A100, ~2-3 hrs):
  pip install mani_skill imageio
  python abm_experiment.py --loop-module abm.loop_lewm_maniskill_run35 \\
    --condition lewm_maniskill_pickcube --device cuda --env maniskill \\
    --steps 400000 --n-envs 4
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
SIGREG_LAMBDA   = 0.1     # LeWM paper value (vs 0.05 in Crafter runs)
N_PROJ_SIGREG   = 1024    # LeWM paper value (vs 512 in Crafter runs)
PRED_LR         = 3e-4
OBSERVE_DEFAULT = 200_000
EP_MAX_STEPS    = 200     # PickCube-v1 max episode length
EVAL_INTERVAL   = 10_000
EVAL_N_EPS      = 5
TRAIN_WARMUP    = BATCH_SIZE
# CEM (LeWM paper values for continuous control)
CEM_H           = 8
CEM_K           = 300
CEM_ELITES      = 30
CEM_ITERS       = 30
# Goal buffer thresholds
CUBE_LIFT_THRESH = 0.05   # meters — cube considered "lifted" above this height


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
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE, a_dim: int = 6):
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
        obs_t    = to_t(self._obs_t[idx])
        actions  = torch.from_numpy(self._actions[idx]).float().to(device)   # (B, a_dim)
        obs_next = to_t(self._obs_next[idx])
        return obs_t, actions, obs_next

    def sample_raw(self, n: int) -> Optional[np.ndarray]:
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return self._obs_t[idx]

    def __len__(self):
        return self._size


# ── Goal pixel buffer (identical to Crafter runs) ─────────────────────────────

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


def _batch_pix_to_tensor(pix_np: np.ndarray, device: str) -> torch.Tensor:
    """(B, H, W, 3) uint8 → (B, 3, H, W) float tensor on device."""
    return torch.from_numpy(pix_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


# ── ManiSkill3 environment helpers ─────────────────────────────────────────────

def _make_env(n_envs: int, seed: int = 42, render_mode: Optional[str] = None):
    """Create a ManiSkill3 PickCube-v1 environment."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers environments

    kwargs = dict(
        num_envs=n_envs,
        obs_mode="state_dict+rgb",
        control_mode="arm_pd_ee_delta_pose",
        reward_mode="none",   # no reward signal used
        render_mode=render_mode or "rgb_array",
        sensor_configs=dict(width=IMG_SIZE, height=IMG_SIZE),
    )
    env = gym.make("PickCube-v1", **kwargs)
    return env


def _extract_rgb(obs: dict) -> np.ndarray:
    """
    Extract (n_envs, H, W, 3) uint8 RGB from ManiSkill3 observation dict.
    Handles both tensor and numpy outputs from the environment.
    """
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    return rgb.astype(np.uint8)  # (n_envs, H, W, 3)


def _extract_cube_height(obs: dict, env_idx: int = 0) -> float:
    """Extract cube Z-position from state_dict obs."""
    try:
        obj_pose = obs["extra"]["obj_pose"]
        if isinstance(obj_pose, torch.Tensor):
            obj_pose = obj_pose.cpu().numpy()
        return float(obj_pose[env_idx, 2])  # Z component
    except (KeyError, IndexError):
        return 0.0


def _extract_success(info: dict) -> np.ndarray:
    """Extract per-env success flags as boolean array."""
    success = info.get("success", False)
    if isinstance(success, torch.Tensor):
        return success.cpu().numpy().astype(bool)
    if isinstance(success, np.ndarray):
        return success.astype(bool)
    return np.array([bool(success)])


# ── World model training step ──────────────────────────────────────────────────

def _train_step(encoder, predictor, opt, replay: ContinuousReplayBuffer, device: str):
    if len(replay) < TRAIN_WARMUP:
        return None, None
    obs_t, actions, obs_next = replay.sample(BATCH_SIZE, device)
    z_t    = encoder(obs_t)
    z_next = encoder(obs_next).detach()
    # actions: (B, a_dim) continuous float — passed directly (no one-hot)
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
    """Sample a goal embedding: 70% from goal_buf, 30% from replay."""
    use_goal = len(goal_buf) > 0 and random.random() < 0.7
    raw = goal_buf.sample_raw(1) if use_goal else replay.sample_raw(1)
    if raw is None:
        raw = replay.sample_raw(1)
    if raw is None:
        return None
    with torch.no_grad():
        return encoder(_pix_to_tensor(raw[0], device))


# ── Evaluation with video recording ───────────────────────────────────────────

def _eval_maniskill(
    encoder: nn.Module,
    predictor,
    goal_buf: GoalPixelBuffer,
    replay: ContinuousReplayBuffer,
    a_dim: int,
    device: str,
    step: int,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
    save_video: bool = True,
) -> dict:
    """Run n_eps evaluation episodes and return metrics."""
    if len(replay) < TRAIN_WARMUP:
        return {"success_rate": 0.0, "lift_rate": 0.0}

    encoder.eval()
    predictor.eval()

    planner = ContinuousCEMPlanner(
        predictor, a_dim=a_dim, H=CEM_H, K=CEM_K,
        n_elites=CEM_ELITES, n_iters=CEM_ITERS, device=device,
    )

    successes = 0
    lifts     = 0
    video_frames: List[np.ndarray] = []
    record_ep = 0  # record the first episode

    for ep in range(n_eps):
        env = _make_env(1, seed=seed_offset + ep,
                        render_mode="rgb_array" if (save_video and ep == record_ep) else None)
        obs, _ = env.reset(seed=seed_offset + ep)
        planner.reset()

        z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
        if z_goal is None:
            with torch.no_grad():
                rgb = _extract_rgb(obs)[0]
                z_goal = encoder(_pix_to_tensor(rgb, device))

        ep_success = False
        ep_lift    = False

        for step_i in range(EP_MAX_STEPS):
            rgb = _extract_rgb(obs)[0]  # (H, W, 3)

            if save_video and ep == record_ep:
                frame = env.render()
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame is not None:
                    if frame.ndim == 4:
                        frame = frame[0]
                    video_frames.append(frame.astype(np.uint8))

            with torch.no_grad():
                z_cur = encoder(_pix_to_tensor(rgb, device))

            action = planner.plan(z_cur, z_goal)  # (a_dim,)
            obs, _, terminated, truncated, info = env.step(action[None])  # add batch dim

            success_arr = _extract_success(info)
            if success_arr[0]:
                ep_success = True

            cube_h = _extract_cube_height(obs, env_idx=0)
            if cube_h > CUBE_LIFT_THRESH:
                ep_lift = True

            if terminated[0] if hasattr(terminated, '__getitem__') else terminated:
                break

        env.close()
        if ep_success:
            successes += 1
        if ep_lift:
            lifts += 1

    # Save video
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
        "lift_rate":    lifts     / n_eps,
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

    # ── Environment ─────────────────────────────────────────────────────────────
    env = _make_env(n_envs, seed=seed)
    obs, _ = env.reset(seed=seed)

    a_dim = int(env.action_space.shape[-1])
    logger.info(f"[RUN35] Action dim: {a_dim}  Obs keys: {list(obs.keys())}")

    # ── Model ───────────────────────────────────────────────────────────────────
    encoder  = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    # Predictor with n_actions=a_dim — continuous action passed directly as float
    predictor = Predictor(latent_dim=Z_DIM, n_actions=a_dim, hidden=512).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = optim.Adam(params, lr=PRED_LR)

    replay   = ContinuousReplayBuffer(capacity=REPLAY_CAP, img_size=IMG_SIZE, a_dim=a_dim)
    goal_buf = GoalPixelBuffer(capacity=GOAL_BUF_CAP)

    # ── Metrics ─────────────────────────────────────────────────────────────────
    results: Dict = {
        "condition":        condition,
        "env_type":         "maniskill",
        "env_steps":        [],
        "success_rate":     [],
        "lift_rate":        [],
        "pred_loss_ewa":    [],
        "sigreg_loss_ewa":  [],
        "mode":             [],
        "wall_time_s":      [],
        "best_success":     0.0,
        "total_time_s":     0.0,
        "act_steps":        0,
        "observe_steps":    observe_steps,
    }

    pred_ewa  = 0.0
    sig_ewa   = 0.0
    ewa_alpha = 0.01
    t_start   = time.time()
    phase     = "OBSERVE"

    # CEM planner (created after OBSERVE freeze)
    planner: Optional[ContinuousCEMPlanner] = None

    global_step = 0
    train_ticker = 0

    # ── Collect first obs ───────────────────────────────────────────────────────
    rgb_batch = _extract_rgb(obs)  # (n_envs, H, W, 3)

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
            # Random exploration
            actions_np = env.action_space.sample()  # (n_envs, a_dim) or (a_dim,)
            if actions_np.ndim == 1:
                actions_np = actions_np[None].repeat(n_envs, axis=0)
        else:
            # CEM planning per environment
            actions_np = np.zeros((n_envs, a_dim), dtype=np.float32)
            z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
            if z_goal is None:
                actions_np = env.action_space.sample()
                if actions_np.ndim == 1:
                    actions_np = actions_np[None].repeat(n_envs, axis=0)
            else:
                for i in range(n_envs):
                    with torch.no_grad():
                        z_cur = encoder(_pix_to_tensor(rgb_batch[i], device))
                    actions_np[i] = planner.plan(z_cur, z_goal)

        # ── Environment step ────────────────────────────────────────────────────
        next_obs, _, terminated, truncated, info = env.step(actions_np)
        next_rgb = _extract_rgb(next_obs)  # (n_envs, H, W, 3)

        # ── Store transitions ───────────────────────────────────────────────────
        success_arr = _extract_success(info)
        for i in range(n_envs):
            replay.push(rgb_batch[i], actions_np[i], next_rgb[i])

            # Goal buffer: cube lifted or task succeeded
            cube_h = _extract_cube_height(next_obs, env_idx=i)
            if cube_h > CUBE_LIFT_THRESH or (i < len(success_arr) and success_arr[i]):
                goal_buf.add(next_rgb[i])

        global_step += n_envs
        train_ticker += n_envs

        # ── Training (OBSERVE only) ─────────────────────────────────────────────
        if phase == "OBSERVE" and train_ticker >= TRAIN_FREQ:
            train_ticker = 0
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa = (1 - ewa_alpha) * pred_ewa + ewa_alpha * pl
                sig_ewa  = (1 - ewa_alpha) * sig_ewa  + ewa_alpha * rl

        # ── Handle episode resets ───────────────────────────────────────────────
        done_any = terminated if isinstance(terminated, bool) else terminated.any()
        if done_any:
            if planner is not None:
                planner.reset()
            obs, _ = env.reset()
            next_rgb = _extract_rgb(obs)

        rgb_batch = next_rgb

        # ── Evaluation & logging ────────────────────────────────────────────────
        if global_step % EVAL_INTERVAL < n_envs:
            metrics = _eval_maniskill(
                encoder, predictor, goal_buf, replay, a_dim, device,
                step=global_step, save_video=True,
            )
            elapsed = time.time() - t_start
            results["env_steps"].append(global_step)
            results["success_rate"].append(metrics["success_rate"])
            results["lift_rate"].append(metrics["lift_rate"])
            results["pred_loss_ewa"].append(pred_ewa)
            results["sigreg_loss_ewa"].append(sig_ewa)
            results["mode"].append(phase)
            results["wall_time_s"].append(elapsed)

            if metrics["success_rate"] > results["best_success"]:
                results["best_success"] = metrics["success_rate"]

            logger.info(
                f"[RUN35] step={global_step:>7} | {phase:7} | "
                f"success={metrics['success_rate']:.1%} lift={metrics['lift_rate']:.1%} | "
                f"pred_ewa={pred_ewa:.4f} sig_ewa={sig_ewa:.4f} | "
                f"replay={len(replay)} goal={len(goal_buf)} | "
                f"t={elapsed:.0f}s"
            )

    env.close()
    results["total_time_s"] = time.time() - t_start
    results["act_steps"]    = max(0, global_step - observe_steps)
    logger.info(f"[RUN35] Done | best_success={results['best_success']:.1%} "
                f"total_t={results['total_time_s']:.0f}s")
    return results
