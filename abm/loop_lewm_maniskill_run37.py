"""
abm/loop_lewm_maniskill_run37.py — Run 37: HWM-style Hierarchy on FetchPickAndPlace-v4

Two-level world model + hierarchical CEM planning. Addresses Run 35's failure:
flat CEM can't chain "reach → grasp → lift" over 50 steps because world model
error accumulates too far ahead.

Architecture:
  Shared encoder (ViT-Tiny, frozen after OBSERVE) produces z from 64×64 RGB.

  Low-level world model (same as Run 35):
    predictor_lo: (z_t, a_t) → z_{t+1}
    Trained every K=3 steps on primitive transitions.

  High-level world model (new):
    predictor_hi: z_t → z_{t+K}  (unconditional K-step predictor, K=3)
    Trained on waypoint pairs (obs_t, obs_{t+K}) collected during OBSERVE.
    Loss: MSE(predictor_hi(z_t), z_{t+K}) + SIGReg(z_t)

  Hierarchical planner (ACT phase):
    Step 1 — subgoal selection (every K env steps):
      Sample N_CANDIDATES frames from goal_buf (70%) + replay (30%).
      Encode them → z_cands.
      Score each: α·cos(predictor_hi(z_cur), z_cand) + (1-α)·cos(z_cand, z_goal)
      Select z_subgoal = argmax scored candidate.
    Step 2 — low-level CEM (every env step):
      Standard Gaussian CEM, H=3, K=100, targeting z_subgoal.
      After K steps, re-run subgoal selection.

Why this works: instead of planning 50 primitive steps to the goal, we plan 3
steps to an intermediate state that's demonstrably reachable (it came from the
replay) and on the way to the goal (scored by goal proximity). This matches the
HWM paper's 70% vs 0% result on real Franka pick-and-place.

Condition:   lewm_maniskill_hierarchy
Loop module: abm.loop_lewm_maniskill_run37

RunPod:
  MUJOCO_GL=egl python abm_experiment.py --loop-module abm.loop_lewm_maniskill_run37 \\
    --condition lewm_maniskill_hierarchy --device cuda --env maniskill \\
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
IMG_SIZE           = 64
Z_DIM              = 256
REPLAY_CAP         = 50_000
WAYPOINT_CAP       = 20_000   # separate buffer for K-step pairs
GOAL_BUF_CAP       = 2_000
TRAIN_FREQ         = 16
BATCH_SIZE         = 256
SIGREG_LAMBDA      = 0.1
N_PROJ_SIGREG      = 1024
PRED_LR            = 3e-4
OBSERVE_DEFAULT    = 200_000
EP_MAX_STEPS       = 50
EVAL_INTERVAL      = 10_000
EVAL_N_EPS         = 5
TRAIN_WARMUP       = BATCH_SIZE
# Hierarchy
WAYPOINT_STRIDE    = 3        # K: steps between waypoints
LO_CEM_H           = 3        # low-level CEM horizon (to subgoal)
LO_CEM_K           = 100
LO_CEM_ELITES      = 10
LO_CEM_ITERS       = 10
N_CANDIDATES       = 200      # candidate subgoals scored per replan
SUBGOAL_ALPHA      = 0.5      # reachability vs goal-proximity tradeoff
REPLAN_FREQ        = WAYPOINT_STRIDE  # re-select subgoal every K steps
# Goal buffer
REWARD_GOAL_THRESH = -0.1


# ── ViT-Tiny encoder ──────────────────────────────────────────────────────────

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


# ── High-level predictor (unconditional K-step) ───────────────────────────────

class HighLevelPredictor(nn.Module):
    """Predicts z_{t+K} from z_t with no action conditioning."""
    def __init__(self, z_dim: int = Z_DIM, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Replay buffers ────────────────────────────────────────────────────────────

class ContinuousReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE, a_dim: int = 4):
        self.capacity  = capacity
        self.a_dim     = a_dim
        self._obs_t    = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions  = np.zeros((capacity, a_dim), dtype=np.float32)
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
        return to_t(self._obs_t[idx]), torch.from_numpy(self._actions[idx]).float().to(device), to_t(self._obs_next[idx])

    def sample_raw(self, n):
        if self._size == 0:
            return None
        return self._obs_t[np.random.choice(self._size, size=min(n, self._size), replace=False)]

    def __len__(self):
        return self._size


class WaypointReplayBuffer:
    """Stores (obs_t, obs_{t+K}) pixel pairs for high-level world model training."""
    def __init__(self, capacity: int = WAYPOINT_CAP, img_size: int = IMG_SIZE):
        self._obs_t = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._obs_k = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._ptr   = 0
        self._size  = 0
        self.capacity = capacity

    def push(self, obs_t: np.ndarray, obs_k: np.ndarray):
        self._obs_t[self._ptr] = obs_t
        self._obs_k[self._ptr] = obs_k
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n, device):
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        def to_t(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return to_t(self._obs_t[idx]), to_t(self._obs_k[idx])

    def __len__(self):
        return self._size


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


# ── Pixel helpers ─────────────────────────────────────────────────────────────

def _pix_to_tensor(pix_hwc: np.ndarray, device: str) -> torch.Tensor:
    import cv2
    if pix_hwc.shape[0] != IMG_SIZE or pix_hwc.shape[1] != IMG_SIZE:
        pix_hwc = cv2.resize(pix_hwc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(pix_hwc.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


# ── Environment helpers (identical to Run 35) ─────────────────────────────────

def _make_single_env(seed: int = 42):
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
    import gymnasium as gym
    return gym.make(
        "FetchPickAndPlace-v4",
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=EP_MAX_STEPS,
    )


def _get_pixels(env) -> np.ndarray:
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


# ── Training steps ────────────────────────────────────────────────────────────

def _train_lo(encoder, predictor_lo, opt, replay: ContinuousReplayBuffer, device: str):
    """Low-level world model: (z_t, a_t) → z_{t+1}."""
    if len(replay) < TRAIN_WARMUP:
        return None, None
    obs_t, actions, obs_next = replay.sample(BATCH_SIZE, device)
    z_t    = encoder(obs_t)
    z_next = encoder(obs_next).detach()
    z_pred    = predictor_lo(z_t, actions)
    pred_loss = F.mse_loss(z_pred, z_next)
    reg_loss  = sigreg(z_t, n_proj=N_PROJ_SIGREG)
    loss = pred_loss + SIGREG_LAMBDA * reg_loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor_lo.parameters()), max_norm=1.0)
    opt.step()
    return pred_loss.item(), reg_loss.item()


def _train_hi(encoder, predictor_hi, opt_hi, waypoint_buf: WaypointReplayBuffer, device: str):
    """High-level world model: z_t → z_{t+K}."""
    if len(waypoint_buf) < TRAIN_WARMUP:
        return None
    obs_t, obs_k = waypoint_buf.sample(BATCH_SIZE, device)
    with torch.no_grad():
        z_k = encoder(obs_k)
    z_t    = encoder(obs_t)
    z_pred = predictor_hi(z_t)
    loss   = F.mse_loss(z_pred, z_k)
    opt_hi.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(predictor_hi.parameters()), max_norm=1.0)
    opt_hi.step()
    return loss.item()


# ── Subgoal selection ─────────────────────────────────────────────────────────

@torch.no_grad()
def _select_subgoal(
    z_cur: torch.Tensor,
    z_goal: torch.Tensor,
    predictor_hi: HighLevelPredictor,
    goal_buf: GoalPixelBuffer,
    replay: ContinuousReplayBuffer,
    encoder: nn.Module,
    device: str,
) -> torch.Tensor:
    """
    Score N_CANDIDATES frames from goal_buf/replay as candidate subgoals.
    Score = α·cos(hi_pred(z_cur), z_cand) + (1-α)·cos(z_cand, z_goal)
    Returns (1, z_dim) tensor of the best candidate.
    """
    use_goal = len(goal_buf) > 0
    n_goal   = int(N_CANDIDATES * 0.7) if use_goal else 0
    n_replay = N_CANDIDATES - n_goal

    raw_parts = []
    if n_goal > 0:
        g = goal_buf.sample_raw(n_goal)
        if g is not None:
            raw_parts.append(g)
    if n_replay > 0:
        r = replay.sample_raw(n_replay)
        if r is not None:
            raw_parts.append(r)

    if not raw_parts:
        return z_goal  # fallback

    raw = np.concatenate(raw_parts, axis=0)  # (N, H, W, 3)
    # Encode in one batch
    pix_t = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    z_cands = encoder(pix_t)  # (N, z_dim)

    N = z_cands.shape[0]
    z_pred_hi = predictor_hi(z_cur.expand(N, -1))  # (N, z_dim)

    reach_score = F.cosine_similarity(z_pred_hi, z_cands, dim=-1)
    goal_score  = F.cosine_similarity(z_cands, z_goal.expand(N, -1), dim=-1)
    scores      = SUBGOAL_ALPHA * reach_score + (1 - SUBGOAL_ALPHA) * goal_score

    best = scores.argmax()
    return z_cands[best].unsqueeze(0)  # (1, z_dim)


# ── Goal sampling (for goal buffer → z_goal) ──────────────────────────────────

def _sample_goal_z(goal_buf, replay, encoder, device):
    use_goal = len(goal_buf) > 0 and random.random() < 0.7
    raw = goal_buf.sample_raw(1) if use_goal else replay.sample_raw(1)
    if raw is None:
        raw = replay.sample_raw(1)
    if raw is None:
        return None
    with torch.no_grad():
        return encoder(_pix_to_tensor(raw[0], device))


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_hierarchical(
    encoder, predictor_lo, predictor_hi,
    goal_buf, replay, a_dim, device, step,
    n_eps=EVAL_N_EPS, save_video=True,
) -> dict:
    if len(replay) < TRAIN_WARMUP:
        return {"success_rate": 0.0, "near_rate": 0.0}

    encoder.eval()
    predictor_lo.eval()
    predictor_hi.eval()

    lo_planner = ContinuousCEMPlanner(
        predictor_lo, a_dim=a_dim, H=LO_CEM_H, K=LO_CEM_K,
        n_elites=LO_CEM_ELITES, n_iters=LO_CEM_ITERS, device=device,
    )

    successes = 0
    nears     = 0
    video_frames: List[np.ndarray] = []

    for ep in range(n_eps):
        env = _make_single_env(seed=1000 + ep)
        obs, _ = env.reset()
        lo_planner.reset()

        rgb   = _get_pixels(env)
        z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
        if z_goal is None:
            with torch.no_grad():
                z_goal = encoder(_pix_to_tensor(rgb, device))

        ep_success = False
        ep_near    = False
        record     = save_video and ep == 0
        z_subgoal  = z_goal
        step_in_ep = 0

        for _ in range(EP_MAX_STEPS):
            rgb = _get_pixels(env)
            if record:
                video_frames.append(rgb.copy())

            with torch.no_grad():
                z_cur = encoder(_pix_to_tensor(rgb, device))

            # Re-select subgoal every REPLAN_FREQ steps
            if step_in_ep % REPLAN_FREQ == 0:
                z_subgoal = _select_subgoal(
                    z_cur, z_goal, predictor_hi, goal_buf, replay, encoder, device)

            action = lo_planner.plan(z_cur, z_subgoal)
            obs, reward, terminated, truncated, info = env.step(action)
            step_in_ep += 1

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
            Path("media").mkdir(exist_ok=True)
            vid_path = f"media/run37_hierarchy_eval_step{step:07d}.mp4"
            imageio.mimsave(vid_path, video_frames, fps=20)
            logger.info(f"[RUN37] Saved eval video: {vid_path}")
        except Exception as e:
            logger.warning(f"[RUN37] Video save failed: {e}")

    encoder.train()
    predictor_lo.train()
    predictor_hi.train()

    return {"success_rate": successes / n_eps, "near_rate": nears / n_eps}


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_abm_loop(
    condition: str = "lewm_maniskill_hierarchy",
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

    logger.info(f"[RUN37] condition={condition} device={device} "
                f"observe={observe_steps} total={max_steps} n_envs={n_envs}")

    # ── Environments ─────────────────────────────────────────────────────────
    envs = [_make_single_env(seed=seed + i) for i in range(n_envs)]
    obs_list = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]

    a_dim = _get_action_dim(envs[0])
    logger.info(f"[RUN37] FetchPickAndPlace-v4 | a_dim={a_dim} | waypoint_stride={WAYPOINT_STRIDE}")

    # ── Models ───────────────────────────────────────────────────────────────
    encoder      = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor_lo = Predictor(latent_dim=Z_DIM, n_actions=a_dim, hidden=512).to(device)
    predictor_hi = HighLevelPredictor(z_dim=Z_DIM, hidden=512).to(device)

    lo_params = list(encoder.parameters()) + list(predictor_lo.parameters())
    opt_lo    = optim.Adam(lo_params, lr=PRED_LR)
    opt_hi    = optim.Adam(predictor_hi.parameters(), lr=PRED_LR)

    replay      = ContinuousReplayBuffer(capacity=REPLAY_CAP, img_size=IMG_SIZE, a_dim=a_dim)
    waypoint_buf = WaypointReplayBuffer(capacity=WAYPOINT_CAP, img_size=IMG_SIZE)
    goal_buf    = GoalPixelBuffer(capacity=GOAL_BUF_CAP)

    # ── Metrics ──────────────────────────────────────────────────────────────
    results: Dict = {
        "condition":       condition,
        "env_type":        "fetch_hierarchy",
        "env_steps":       [],
        "success_rate":    [],
        "near_rate":       [],
        "pred_loss_ewa":   [],
        "hi_loss_ewa":     [],
        "sigreg_loss_ewa": [],
        "mode":            [],
        "wall_time_s":     [],
        "best_success":    0.0,
        "total_time_s":    0.0,
        "act_steps":       0,
        "observe_steps":   observe_steps,
    }

    pred_ewa  = 0.0
    hi_ewa    = 0.0
    sig_ewa   = 0.0
    ewa_alpha = 0.01
    t_start   = time.time()
    phase     = "OBSERVE"

    lo_planner: Optional[ContinuousCEMPlanner] = None

    global_step  = 0
    train_ticker = 0

    # Per-env step counters for waypoint collection
    ep_step_counters = [0] * n_envs
    prev_rgb         = [None] * n_envs  # obs_t stored for waypoint pair

    rgb_batch = np.stack([_get_pixels(env) for env in envs])

    # Initialise prev_rgb for waypoint stride tracking
    for i in range(n_envs):
        prev_rgb[i] = rgb_batch[i].copy()

    # Per-env subgoal state (ACT phase)
    z_subgoals  = [None] * n_envs
    replan_ctrs = [0] * n_envs

    while global_step < max_steps:

        # ── Phase transition ──────────────────────────────────────────────────
        if phase == "OBSERVE" and global_step >= observe_steps:
            phase = "ACT"
            encoder.eval()
            predictor_lo.eval()
            predictor_hi.eval()
            for p in list(encoder.parameters()) + list(predictor_lo.parameters()) + list(predictor_hi.parameters()):
                p.requires_grad_(False)
            lo_planner = ContinuousCEMPlanner(
                predictor_lo, a_dim=a_dim, H=LO_CEM_H, K=LO_CEM_K,
                n_elites=LO_CEM_ELITES, n_iters=LO_CEM_ITERS, device=device,
            )
            logger.info(f"[RUN37] → ACT phase | pred_ewa={pred_ewa:.4f} hi_ewa={hi_ewa:.4f} "
                        f"sig_ewa={sig_ewa:.4f} waypoints={len(waypoint_buf)} goal={len(goal_buf)}")

        # ── Action selection ──────────────────────────────────────────────────
        if phase == "OBSERVE" or lo_planner is None:
            actions_np = np.stack([envs[0].action_space.sample() for _ in range(n_envs)])
        else:
            actions_np = np.zeros((n_envs, a_dim), dtype=np.float32)
            z_goal_global = _sample_goal_z(goal_buf, replay, encoder, device)
            if z_goal_global is None:
                actions_np = np.stack([envs[0].action_space.sample() for _ in range(n_envs)])
            else:
                for i in range(n_envs):
                    with torch.no_grad():
                        z_cur = encoder(_pix_to_tensor(rgb_batch[i], device))

                    # Re-plan subgoal every REPLAN_FREQ steps
                    if replan_ctrs[i] % REPLAN_FREQ == 0 or z_subgoals[i] is None:
                        z_subgoals[i] = _select_subgoal(
                            z_cur, z_goal_global, predictor_hi,
                            goal_buf, replay, encoder, device,
                        )
                        lo_planner.reset()
                    replan_ctrs[i] += 1

                    actions_np[i] = lo_planner.plan(z_cur, z_subgoals[i])

        # ── Environment step ──────────────────────────────────────────────────
        next_rgb = np.zeros_like(rgb_batch)
        for i in range(n_envs):
            obs, reward, terminated, truncated, info = envs[i].step(actions_np[i])
            pix_next = _get_pixels(envs[i])
            next_rgb[i] = pix_next

            # Low-level replay
            replay.push(rgb_batch[i], actions_np[i], pix_next)

            # Waypoint replay: store (prev_rgb[i], pix_next) every WAYPOINT_STRIDE steps
            ep_step_counters[i] += 1
            if ep_step_counters[i] % WAYPOINT_STRIDE == 0:
                waypoint_buf.push(prev_rgb[i], pix_next)
                prev_rgb[i] = pix_next.copy()

            # Goal buffer
            if reward > REWARD_GOAL_THRESH or info.get("is_success", False):
                goal_buf.add(pix_next)

            if terminated or truncated:
                obs, _ = envs[i].reset()
                next_rgb[i] = _get_pixels(envs[i])
                ep_step_counters[i] = 0
                prev_rgb[i] = next_rgb[i].copy()
                replan_ctrs[i] = 0
                z_subgoals[i] = None
                if lo_planner is not None:
                    lo_planner.reset()

        global_step  += n_envs
        train_ticker += n_envs

        # ── Training (OBSERVE only) ───────────────────────────────────────────
        if phase == "OBSERVE" and train_ticker >= TRAIN_FREQ:
            train_ticker = 0
            pl, rl = _train_lo(encoder, predictor_lo, opt_lo, replay, device)
            hl     = _train_hi(encoder, predictor_hi, opt_hi, waypoint_buf, device)
            if pl is not None:
                pred_ewa = (1 - ewa_alpha) * pred_ewa + ewa_alpha * pl
                sig_ewa  = (1 - ewa_alpha) * sig_ewa  + ewa_alpha * rl
            if hl is not None:
                hi_ewa = (1 - ewa_alpha) * hi_ewa + ewa_alpha * hl

        rgb_batch = next_rgb

        # ── Evaluation & logging ──────────────────────────────────────────────
        if global_step % EVAL_INTERVAL < n_envs:
            metrics = _eval_hierarchical(
                encoder, predictor_lo, predictor_hi,
                goal_buf, replay, a_dim, device,
                step=global_step, save_video=True,
            )
            elapsed = time.time() - t_start
            results["env_steps"].append(global_step)
            results["success_rate"].append(metrics["success_rate"])
            results["near_rate"].append(metrics["near_rate"])
            results["pred_loss_ewa"].append(pred_ewa)
            results["hi_loss_ewa"].append(hi_ewa)
            results["sigreg_loss_ewa"].append(sig_ewa)
            results["mode"].append(phase)
            results["wall_time_s"].append(elapsed)

            if metrics["success_rate"] > results["best_success"]:
                results["best_success"] = metrics["success_rate"]

            logger.info(
                f"[RUN37] step={global_step:>7} | {phase:7} | "
                f"success={metrics['success_rate']:.1%} near={metrics['near_rate']:.1%} | "
                f"pred_ewa={pred_ewa:.4f} hi_ewa={hi_ewa:.4f} sig={sig_ewa:.4f} | "
                f"replay={len(replay)} wp={len(waypoint_buf)} goal={len(goal_buf)} | "
                f"t={elapsed:.0f}s"
            )

    for env in envs:
        env.close()
    results["total_time_s"] = time.time() - t_start
    results["act_steps"]    = max(0, global_step - observe_steps)
    logger.info(f"[RUN37] Done | best_success={results['best_success']:.1%} "
                f"total_t={results['total_time_s']:.0f}s")
    return results
