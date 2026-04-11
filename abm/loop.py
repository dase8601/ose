"""
abm/loop.py — Main A↔B training loop (vectorized environments).

run_abm_loop(condition, device, max_steps, seed) → metrics dict

Three conditions:
  "autonomous" — AutonomousSystemM (plateau-triggered switching)
  "fixed"      — FixedSystemM (switch every K steps)
  "ppo_only"   — Raw-pixel PPO baseline (no LeWM, no mode switching)

Vectorization: N_ENVS parallel environments step simultaneously each outer
loop iteration, giving N_ENVS transitions per call.  This is the primary
speedup — the env.step() bottleneck is parallelized across CPU cores.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from minigrid.wrappers import RGBImgObsWrapper

from .lewm import LeWM, ReplayBuffer
from .ppo import PPO, PPOAgent, RolloutBuffer
from .meta_controller import AutonomousSystemM, FixedSystemM, Mode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LATENT_DIM  = 128
N_ACTIONS   = 7
IMG_H = IMG_W = 48      # MiniGrid-DoorKey-6x6 with tile_size=8

N_ENVS      = 16        # parallel environments — primary speedup lever

LEWM_LR     = 3e-4
LEWM_BATCH  = 256       # larger batch to match higher data throughput
LEWM_WARMUP = 500       # transitions before LeWM training starts

PPO_LR          = 2.5e-4
PPO_ROLLOUT     = 128   # outer loop steps; total transitions = 128 × 16 = 2048
EVAL_INTERVAL   = 5_000
EVAL_EPISODES   = 50

FIXED_SWITCH_EVERY = 10_000


# ---------------------------------------------------------------------------
# Reward shaping wrapper
# ---------------------------------------------------------------------------

class ShapedRewardWrapper(gymnasium.Wrapper):
    """
    Adds small intermediate rewards to DoorKey's sparse signal:
      +0.1 first time agent picks up the key (per episode)
      +0.1 first time agent opens the door (per episode)
    Terminal reward (+1.0 on goal) is unchanged.
    """
    KEY_BONUS  = 0.1
    DOOR_BONUS = 0.1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._has_key     = False
        self._door_opened = False
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        inner = self.env.unwrapped

        if not self._has_key and inner.carrying is not None:
            reward += self.KEY_BONUS
            self._has_key = True

        if not self._door_opened:
            for obj in inner.grid.grid:
                if obj is not None and obj.type == "door" and obj.is_open:
                    reward += self.DOOR_BONUS
                    self._door_opened = True
                    break

        return obs, reward, term, trunc, info


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0):
    """Single env factory — used for eval and as factory for vectorized envs."""
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = ShapedRewardWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=8)
    return env


def make_vec_env(n_envs: int, seed: int = 0, use_async: bool = True):
    """
    Create N parallel environments.
    AsyncVectorEnv: each env runs in its own subprocess (true parallelism).
    SyncVectorEnv:  sequential fallback for systems where multiprocessing is flaky.
    """
    fns = [lambda i=i: make_env(seed=seed + i) for i in range(n_envs)]
    VecCls = AsyncVectorEnv if use_async else SyncVectorEnv
    return VecCls(fns)


def batch_obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    """
    Vectorized obs dict → (N, C, H, W) float32 on device.
    obs_dict["image"]: (N, H, W, C) uint8  (from AsyncVectorEnv)
    """
    imgs = obs_dict["image"]                              # (N, H, W, C) uint8
    x    = torch.from_numpy(imgs.astype(np.float32) / 255.0)  # (N, H, W, C)
    return x.permute(0, 3, 1, 2).to(device)              # (N, C, H, W)


def single_obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    """Single env obs dict → (1, C, H, W) float32 on device."""
    img = obs_dict["image"]                               # (H, W, C) uint8
    x   = torch.from_numpy(img.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)    # (1, C, H, W)


def eval_agent(agent: PPOAgent, encoder, device: str, seed_offset: int = 1000,
               n_eps: int = EVAL_EPISODES) -> float:
    """Evaluate PPO agent with single env — returns success rate."""
    successes = 0
    for ep in range(n_eps):
        env = make_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done     = False
        ep_steps = 0
        ep_ret   = 0.0
        while not done and ep_steps < 300:
            with torch.no_grad():
                z = encoder(obs)
                action, _, _, _ = agent.get_action_and_value(z)
            obs, r, term, trunc, _ = env.step(action.item())
            done    = term or trunc
            ep_ret += r
            ep_steps += 1
        if ep_ret > 0.5:
            successes += 1
        env.close()
    return successes / n_eps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_abm_loop(
    condition: str,
    device:    str = "cuda",
    max_steps: int = 400_000,
    seed:      int = 42,
    n_envs:    int = N_ENVS,
) -> Dict:
    """
    Run a single condition of the A-B-M experiment with vectorized environments.

    Parameters
    ----------
    condition : "autonomous" | "fixed" | "ppo_only"
    device    : "cuda" | "mps" | "cpu"
    max_steps : total environment steps (each outer iter = n_envs steps)
    seed      : random seed
    n_envs    : number of parallel environments

    Returns
    -------
    metrics dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(
        f"[{condition.upper()}] Starting — device={device}, "
        f"max_steps={max_steps}, n_envs={n_envs}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # AsyncVectorEnv uses multiprocessing; fall back to sync if on a platform
    # where subprocess forking is unreliable (e.g., some macOS configurations).
    use_async = (device == "cuda")
    envs = make_vec_env(n_envs, seed=seed, use_async=use_async)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    # ---- Model setup ----
    if condition == "ppo_only":
        flat_dim = IMG_H * IMG_W * 3
        agent    = PPOAgent(latent_dim=flat_dim, n_actions=N_ACTIONS).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(PPO_ROLLOUT, n_envs, flat_dim, device)
        lewm     = None
        opt_lewm = None
        buf_lew  = None

        def encoder(obs_dict):
            imgs = obs_dict["image"]                        # (N, H, W, C)
            x    = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            return x.reshape(len(imgs), -1).to(device)     # (N, flat_dim)

        # single-env encoder for eval
        def encoder_single(obs_dict):
            img = obs_dict["image"]
            x   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return x.flatten().unsqueeze(0).to(device)

    else:
        lewm     = LeWM(latent_dim=LATENT_DIM, n_actions=N_ACTIONS).to(device)
        agent    = PPOAgent(latent_dim=LATENT_DIM, n_actions=N_ACTIONS).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(PPO_ROLLOUT, n_envs, LATENT_DIM, device)
        buf_lew  = ReplayBuffer(capacity=50_000)
        opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)

        def encoder(obs_dict):
            return lewm.encode(batch_obs_to_tensor(obs_dict, device))  # (N, D)

        def encoder_single(obs_dict):
            return lewm.encode(single_obs_to_tensor(obs_dict, device))  # (1, D)

    # ---- System M ----
    if condition == "autonomous":
        sysm = AutonomousSystemM(
            obs_plateau_steps=8_000,
            act_plateau_steps=20_000,
            plateau_threshold=0.01,
        )
    elif condition == "fixed":
        sysm = FixedSystemM(switch_every=FIXED_SWITCH_EVERY)
    else:
        sysm = None

    # ---- Metric tracking ----
    metrics: Dict[str, List] = {
        "env_step":     [],
        "success_rate": [],
        "ssl_loss_ewa": [],
        "mode":         [],
        "wall_time_s":  [],
    }
    ssl_ewa     = None
    mode_str    = "OBSERVE" if condition != "ppo_only" else "ACT"
    steps_to_80 = None
    env_step    = 0

    # Per-env episode return accumulators
    ep_ret    = np.zeros(n_envs, dtype=np.float32)
    last_done = np.zeros(n_envs, dtype=bool)

    # ---- Main loop ----
    while env_step < max_steps:

        # ── Mode determination ──────────────────────────────────────────────
        if condition == "ppo_only":
            current_mode = Mode.ACT
        elif condition == "autonomous":
            current_mode = sysm.mode
        else:
            current_mode = sysm.step(env_step)

        mode_str = current_mode.name

        # ── OBSERVE step ────────────────────────────────────────────────────
        if current_mode == Mode.OBSERVE:
            actions  = envs.action_space.sample()           # (N,) numpy
            obs_imgs = obs["image"].copy()                  # (N, H, W, C)
            next_obs, _, terms, truncs, _ = envs.step(actions)
            next_imgs = next_obs["image"]                   # (N, H, W, C)

            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions[i]), next_imgs[i])

            obs       = next_obs
            env_step += n_envs

            # Train LeWM every 4 outer iterations once buffer is warm
            ssl_loss_val = None
            if len(buf_lew) >= LEWM_WARMUP and (env_step // n_envs) % 4 == 0:
                obs_t, acts, obs_next = buf_lew.sample(LEWM_BATCH, device)
                opt_lewm.zero_grad()
                loss, info = lewm.loss(obs_t, acts, obs_next)
                loss.backward()
                opt_lewm.step()
                ssl_loss_val = info["loss_total"]
                ssl_ewa = (ssl_loss_val if ssl_ewa is None
                           else 0.95 * ssl_ewa + 0.05 * ssl_loss_val)

            if condition == "autonomous" and ssl_loss_val is not None:
                sysm.observe_step(ssl_loss_val, env_step)

        # ── ACT step ────────────────────────────────────────────────────────
        else:
            with torch.no_grad():
                z = encoder(obs)                            # (N, D)
                actions, log_probs, _, values = agent.get_action_and_value(z)

            next_obs, rewards, terms, truncs, _ = envs.step(actions.cpu().numpy())
            dones   = terms | truncs
            ep_ret += rewards
            env_step += n_envs

            buf_ppo.add(
                z.detach(),
                actions,
                log_probs,
                torch.tensor(rewards, dtype=torch.float32, device=device),
                torch.tensor(dones.astype(np.float32), device=device),
                values,
            )

            # Track completed episodes
            for i in range(n_envs):
                if dones[i]:
                    if condition == "autonomous":
                        sysm.act_step(float(ep_ret[i]), None, env_step)
                    ep_ret[i] = 0.0

            last_done = dones
            obs       = next_obs

            if buf_ppo.is_full:
                with torch.no_grad():
                    last_z   = encoder(obs)                 # (N, D)
                    last_val = agent.get_value(last_z)      # (N,)
                last_done_t = torch.tensor(
                    last_done.astype(np.float32), device=device
                )
                ppo.update(buf_ppo, last_val, last_done_t)

        # ── Periodic evaluation ─────────────────────────────────────────────
        if env_step % EVAL_INTERVAL < n_envs:
            if lewm is not None:
                for p in lewm.encoder.parameters():
                    p.requires_grad_(False)

            sr = eval_agent(agent, encoder_single, device,
                            seed_offset=9000 + env_step, n_eps=EVAL_EPISODES)

            if lewm is not None:
                for p in lewm.encoder.parameters():
                    p.requires_grad_(True)

            if condition == "autonomous":
                sysm.act_step(None, sr, env_step)

            n_sw    = sysm.n_switches() if sysm else 0
            elapsed = time.time() - t0
            logger.info(
                f"[{condition.upper()}] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"switches={n_sw} | {elapsed:.0f}s"
            )

            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)

            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[{condition.upper()}] *** 80% success at step {env_step} ***")

        if sysm is not None and sysm.is_solved:
            logger.info(f"[{condition.upper()}] Solved! Stopping at step {env_step}.")
            break

    envs.close()
    elapsed_total = time.time() - t0

    # ---- Save checkpoint ----
    ckpt_dir = Path("results/abm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if condition == "ppo_only":
        ckpt = {
            "condition": condition,
            "agent":     agent.state_dict(),
            "flat_dim":  IMG_H * IMG_W * 3,
            "env_step":  env_step,
        }
    else:
        ckpt = {
            "condition":  condition,
            "lewm":       lewm.state_dict(),
            "agent":      agent.state_dict(),
            "latent_dim": LATENT_DIM,
            "n_actions":  N_ACTIONS,
            "env_step":   env_step,
        }
    ckpt_path = ckpt_dir / f"checkpoint_{condition}.pt"
    torch.save(ckpt, ckpt_path)
    logger.info(f"[{condition.upper()}] Checkpoint → {ckpt_path}")

    return {
        "condition":      condition,
        "env_steps":      metrics["env_step"],
        "success_rate":   metrics["success_rate"],
        "ssl_loss_ewa":   metrics["ssl_loss_ewa"],
        "mode":           metrics["mode"],
        "wall_time_s":    metrics["wall_time_s"],
        "steps_to_80pct": steps_to_80,
        "n_switches":     sysm.n_switches() if sysm else 0,
        "switch_log":     sysm.switch_log if sysm else [],
        "total_time_s":   elapsed_total,
    }
