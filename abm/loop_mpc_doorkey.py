"""
abm/loop_mpc_doorkey.py — DoorKey planner-only training loop (Phase 1).

Scientific purpose: isolate and prove the "world model + planning > RL" claim
on MiniGrid-DoorKey-6x6 before testing System M switching on Crafter.

The design is intentionally minimal:
  - OBSERVE: train LeWM from random-policy data, collect goal images on success
  - ACT: always use CEM to plan toward a sampled goal (no PPO, no switching)
  - System M: fixed schedule only — observe for observe_steps, then always ACT

This is Phase 1 of a two-phase ablation:
  Phase 1 (this file): prove LeWM + CEM > PPO-only on DoorKey (42% baseline)
  Phase 2 (Crafter):   prove autonomous System M > fixed switching once planner works

Conditions:
  "planner_only" — observe observe_steps steps, then always ACT with CEM
  "random"       — always random actions (sanity-check baseline)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
import torch
import torch.optim as optim
from minigrid.wrappers import RGBImgObsWrapper

from .world_model import LeWM, ReplayBuffer, SequenceReplayBuffer
from .cem_planner import CEMPlanner

logger = logging.getLogger(__name__)

# ── DoorKey-specific constants ─────────────────────────────────────────────
N_ENVS            = 16
IMG_H = IMG_W     = 48
N_ACTIONS         = 7
LATENT_DIM        = 256
LEWM_LR           = 1e-4
LEWM_BATCH        = 256
LEWM_WARMUP       = 500
REPLAY_CAPACITY   = 100_000
GOAL_CAPACITY     = 1_024
EVAL_INTERVAL     = 5_000
EVAL_N_EPS        = 30
GOAL_REFRESH_STEPS = 64
N_TRAIN_STEPS     = 4
DEFAULT_OBSERVE   = 80_000      # fixed OBSERVE budget before always-ACT
EP_MAX_STEPS      = 300         # DoorKey episode horizon


# ── Environment helpers ────────────────────────────────────────────────────

def _make_doorkey_env(seed: int = 0):
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    env = gymnasium.wrappers.ResizeObservation(env, (IMG_H, IMG_W))
    return env


def _make_doorkey_vec_env(n_envs: int, seed: int = 0):
    fns = [lambda i=i: _make_doorkey_env(seed=seed + i) for i in range(n_envs)]
    return gymnasium.vector.SyncVectorEnv(fns)


def _obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    imgs = obs_dict["image"]
    if imgs.ndim == 3:
        imgs = imgs[None]
    arr = imgs.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)


# ── Goal image buffer ──────────────────────────────────────────────────────

class GoalImageBuffer:
    """
    Stores raw success-state images so goal encodings stay fresh as the
    encoder updates during OBSERVE. Images are re-encoded at eval time.
    """

    def __init__(self, capacity: int = GOAL_CAPACITY):
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, img: np.ndarray) -> None:
        self._buf.append(np.array(img, copy=True))

    def sample(self, n: int) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(
            len(self._buf),
            size=n,
            replace=len(self._buf) < n,
        )
        return np.stack([self._buf[i] for i in idx], axis=0)

    def __len__(self) -> int:
        return len(self._buf)


# ── Evaluation ─────────────────────────────────────────────────────────────

def _eval_doorkey_mpc(
    mpc: Optional[CEMPlanner],
    encoder_fn,
    goal_buf: GoalImageBuffer,
    device: str,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> float:
    """
    Roll out the CEM planner on DoorKey for n_eps episodes.
    Returns success fraction (ep_ret > 0.5 = reached goal).
    Falls back to random if planner not yet ready.
    """
    if mpc is None or len(goal_buf) == 0:
        return 0.0

    goal_imgs = goal_buf.sample(n_eps)
    successes = 0

    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_steps = 0
        ep_ret = 0.0
        goal_img = goal_imgs[ep] if goal_imgs is not None else None
        goal_age = 0

        with torch.no_grad():
            z_goal = (
                encoder_fn({"image": goal_img[None]})
                if goal_img is not None
                else None
            )

        while not done and ep_steps < EP_MAX_STEPS:
            with torch.no_grad():
                z = encoder_fn(obs)

            if mpc is not None and z_goal is not None:
                action = mpc.plan_single(z, z_goal)
            else:
                action = env.action_space.sample()

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r
            ep_steps += 1
            goal_age += 1

            # Refresh goal if stale — re-sample from buffer
            if goal_age >= GOAL_REFRESH_STEPS and len(goal_buf) > 0:
                new_g = goal_buf.sample(1)
                if new_g is not None:
                    with torch.no_grad():
                        z_goal = encoder_fn({"image": new_g})
                goal_age = 0

        if ep_ret > 0.5:
            successes += 1
        env.close()

    return successes / n_eps


def _eval_doorkey_random(seed_offset: int = 1000, n_eps: int = EVAL_N_EPS) -> float:
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_steps = 0
        ep_ret = 0.0
        while not done and ep_steps < EP_MAX_STEPS:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            ep_ret += r
            ep_steps += 1
        if ep_ret > 0.5:
            successes += 1
        env.close()
    return successes / n_eps


# ── Main loop ──────────────────────────────────────────────────────────────

def run_doorkey_mpc_loop(
    condition: str = "planner_only",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    """
    Phase 1 DoorKey planner-only loop.

    Parameters
    ----------
    condition    : "planner_only" | "random"
    device       : "cuda" | "mps" | "cpu"
    max_steps    : total env steps (200k recommended for scout)
    observe_steps: override default 80k OBSERVE budget
    """
    if condition not in ("planner_only", "random"):
        raise ValueError(f"loop_mpc_doorkey supports: planner_only, random — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey.py only supports env_type='doorkey'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE
    logger.info(
        f"[{condition.upper()}] Phase 1 DoorKey — device={device}, "
        f"max_steps={max_steps}, n_envs={n_envs}, observe={_observe_steps}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Environments
    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    # World model + replay
    lewm = LeWM(
        latent_dim=LATENT_DIM,
        n_actions=N_ACTIONS,
        img_size=IMG_H,
        predictor_type="mlp",    # DoorKey is simpler — MLP predictor is sufficient
    ).to(device)
    buf_lew = ReplayBuffer(capacity=REPLAY_CAPACITY)
    buf_seq = SequenceReplayBuffer(capacity=REPLAY_CAPACITY, seq_len=8)
    opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)
    mpc: Optional[CEMPlanner] = None
    goal_buf = GoalImageBuffer(capacity=GOAL_CAPACITY)

    def encoder_fn(obs_input):
        if isinstance(obs_input, dict):
            return lewm.encode(_obs_to_tensor(obs_input, device))
        # raw (B,H,W,C) uint8 array
        arr = obs_input.astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.permute(0, 3, 1, 2).to(device)
        return lewm.encode(t)

    # Tracking
    metrics: Dict[str, List] = {
        "env_step": [],
        "success_rate": [],
        "ssl_loss_ewa": [],
        "mode": [],
        "wall_time_s": [],
        "per_tier": [],
    }
    ssl_ewa = None
    ssl_loss_val = None
    steps_to_80 = None
    env_step = 0
    act_steps = 0
    total_observe_steps = 0
    ep_ret = np.zeros(n_envs, dtype=np.float32)
    last_done = np.zeros(n_envs, dtype=bool)
    active_goal_imgs: List[Optional[np.ndarray]] = [None] * n_envs
    goal_ages = np.zeros(n_envs, dtype=np.int32)

    while env_step < max_steps:

        # ── Phase determination ────────────────────────────────────────────
        in_observe = (env_step < _observe_steps) or (condition == "random")

        if in_observe:
            # ── OBSERVE: random actions, train LeWM ───────────────────────
            mode_str = "OBSERVE"
            actions = envs.action_space.sample()
            obs_imgs = obs["image"].copy()
            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            next_imgs = next_obs["image"].copy()

            # Gymnasium vector: recover true final obs on episode end
            final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
            if final_mask.any() and "final_observation" in infos:
                for i in range(n_envs):
                    if final_mask[i]:
                        next_imgs[i] = infos["final_observation"]["image"][i]

            dones = terms | truncs
            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions[i]), next_imgs[i])
                buf_seq.push(obs_imgs[i], int(actions[i]), bool(dones[i]))
                # Goal collection: store any success-adjacent state
                if rewards[i] > 0:
                    goal_buf.push(next_imgs[i])

            obs = next_obs
            env_step += n_envs
            total_observe_steps += n_envs

            # Train LeWM once replay is warm
            if len(buf_lew) >= LEWM_WARMUP:
                for _ in range(N_TRAIN_STEPS):
                    obs_t, acts, obs_next = buf_lew.sample(LEWM_BATCH, device)
                    opt_lewm.zero_grad()
                    loss, info = lewm.loss(obs_t, acts, obs_next)
                    loss.backward()
                    opt_lewm.step()
                ssl_loss_val = info["loss_total"]

                if len(buf_seq) > buf_seq.seq_len * 2:
                    seq_data = buf_seq.sample_sequences(
                        batch_size=64,
                        encoder_fn=lewm.encode,
                        n_actions=N_ACTIONS,
                        device=device,
                    )
                    if seq_data is not None:
                        z_seq, a_onehot_seq, z_next_target = seq_data
                        opt_lewm.zero_grad()
                        z_pred = lewm.predictor.forward_sequence(z_seq, a_onehot_seq)
                        seq_loss = (1 - torch.nn.functional.cosine_similarity(
                            z_pred, z_next_target.detach(), dim=-1
                        )).mean()
                        seq_loss.backward()
                        opt_lewm.step()

                ssl_ewa = (
                    ssl_loss_val if ssl_ewa is None
                    else 0.95 * ssl_ewa + 0.05 * ssl_loss_val
                )

                # Init CEM as soon as LeWM is warm — use it the moment OBSERVE ends
                if mpc is None:
                    mpc = CEMPlanner(
                        lewm.predictor,
                        n_actions=N_ACTIONS,
                        horizon=10,
                        n_samples=512,
                        n_elites=64,
                        n_iters=5,
                        device=device,
                        distance="cosine",
                    )
                    logger.info(
                        f"[{condition.upper()}] CEM planner ready "
                        f"(H=10, samples=512, elites=64, iters=5) | "
                        f"goals in buffer: {len(goal_buf)}"
                    )

        else:
            # ── ACT: always plan with CEM ──────────────────────────────────
            mode_str = "ACT"

            # Assign / refresh goal images per env
            refresh_mask = last_done.copy() | (goal_ages >= GOAL_REFRESH_STEPS)
            refresh_mask |= np.array([img is None for img in active_goal_imgs], dtype=bool)
            if len(goal_buf) > 0 and refresh_mask.any():
                n_refresh = int(refresh_mask.sum())
                sampled = goal_buf.sample(n_refresh)
                if sampled is not None:
                    for slot, env_idx in enumerate(np.flatnonzero(refresh_mask)):
                        active_goal_imgs[int(env_idx)] = sampled[slot]

            # Plan or fall back to random
            if mpc is not None and len(goal_buf) > 0:
                with torch.no_grad():
                    z = encoder_fn(obs)
                    valid = [img for img in active_goal_imgs if img is not None]
                    if valid:
                        goal_stack = np.stack(active_goal_imgs if all(
                            img is not None for img in active_goal_imgs
                        ) else [valid[0]] * n_envs, axis=0)
                        z_goal = encoder_fn({"image": goal_stack})
                        actions_np = mpc.plan_batch(z, z_goal)
                    else:
                        actions_np = envs.action_space.sample()
            else:
                actions_np = envs.action_space.sample()

            obs_imgs = obs["image"].copy()
            next_obs, rewards, terms, truncs, infos = envs.step(actions_np)
            next_imgs = next_obs["image"].copy()

            final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
            if final_mask.any() and "final_observation" in infos:
                for i in range(n_envs):
                    if final_mask[i]:
                        next_imgs[i] = infos["final_observation"]["image"][i]

            dones = terms | truncs
            ep_ret += rewards
            env_step += n_envs
            act_steps += n_envs

            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions_np[i]), next_imgs[i])
                buf_seq.push(obs_imgs[i], int(actions_np[i]), bool(dones[i]))
                if rewards[i] > 0:
                    goal_buf.push(next_imgs[i])

                goal_ages[i] += 1
                if dones[i]:
                    ep_ret[i] = 0.0
                    active_goal_imgs[i] = None
                    goal_ages[i] = 0

            last_done = dones
            obs = next_obs

        # ── Periodic evaluation ────────────────────────────────────────────
        if env_step % EVAL_INTERVAL < n_envs:
            if condition == "random":
                sr = _eval_doorkey_random(seed_offset=9000 + env_step, n_eps=EVAL_N_EPS)
            else:
                sr = _eval_doorkey_mpc(
                    mpc=mpc,
                    encoder_fn=encoder_fn,
                    goal_buf=goal_buf,
                    device=device,
                    seed_offset=9000 + env_step,
                    n_eps=EVAL_N_EPS,
                )

            elapsed = time.time() - t0
            logger.info(
                f"[{condition.upper()}] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"goals={len(goal_buf)} | act_steps={act_steps} | {elapsed:.0f}s"
            )

            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({})

            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[{condition.upper()}] *** 80% success at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{condition}.pt"
    torch.save(
        {
            "condition": condition,
            "env_type": "doorkey",
            "env_step": env_step,
            "latent_dim": LATENT_DIM,
            "n_actions": N_ACTIONS,
            "lewm": lewm.state_dict(),
        },
        ckpt_path,
    )
    logger.info(f"[{condition.upper()}] Checkpoint → {ckpt_path}")

    return {
        "condition": condition,
        "env_type": "doorkey",
        "env_steps": metrics["env_step"],
        "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"],
        "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"],
        "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80,
        "n_switches": 0,
        "switch_log": [],
        "total_time_s": elapsed_total,
        "act_steps": act_steps,
        "observe_steps": total_observe_steps,
    }


# Drop-in compatibility with abm_experiment.py dynamic import
run_abm_loop = run_doorkey_mpc_loop
