"""
abm/loop.py — Main A↔B training loop (vectorized environments).

run_abm_loop(condition, device, max_steps, seed, env_type) → metrics dict

Three conditions:
  "autonomous" — AutonomousSystemM (plateau-triggered switching)
  "fixed"      — FixedSystemM (switch every K steps)
  "ppo_only"   — Raw-pixel PPO baseline (no LeWM, no mode switching)

Two environments (env_type):
  "doorkey" — MiniGrid-DoorKey-6x6 (original, 48×48, 7 actions)
  "crafter" — Crafter survival game (64×64, 17 actions, 22 achievements)

Vectorization: N_ENVS parallel environments step simultaneously each outer
loop iteration, giving N_ENVS transitions per call.  This is the primary
speedup — the env.step() bottleneck is parallelized across CPU cores.
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
# Hyperparameters (DoorKey defaults — Crafter overrides set inside run_abm_loop)
# ---------------------------------------------------------------------------

LATENT_DIM  = 256
N_ACTIONS   = 7       # DoorKey; Crafter = 17
IMG_H = IMG_W = 48    # DoorKey; Crafter = 64

N_ENVS      = 16

LEWM_LR     = 3e-4
LEWM_BATCH  = 256
LEWM_WARMUP = 500

PPO_LR          = 2.5e-4
PPO_ROLLOUT     = 128   # total transitions per update = 128 × N_ENVS
EVAL_INTERVAL   = 5_000
EVAL_EPISODES   = 50

FIXED_SWITCH_EVERY = 10_000


# ---------------------------------------------------------------------------
# DoorKey environment helpers
# ---------------------------------------------------------------------------

class ShapedRewardWrapper(gymnasium.Wrapper):
    """
    +0.1 for first key pickup and first door open per episode.
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


def make_doorkey_env(seed: int = 0):
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = ShapedRewardWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=8)
    return env


def make_doorkey_vec_env(n_envs: int, seed: int = 0, use_async: bool = True):
    fns = [lambda i=i: make_doorkey_env(seed=seed + i) for i in range(n_envs)]
    if use_async:
        return AsyncVectorEnv(fns, shared_memory=False)
    return SyncVectorEnv(fns)


# ---------------------------------------------------------------------------
# Shared obs tensor helpers (work for both envs — both use {"image": ...})
# ---------------------------------------------------------------------------

def batch_obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    """(N, H, W, C) uint8 → (N, C, H, W) float32 on device."""
    imgs = obs_dict["image"]
    x    = torch.from_numpy(imgs.astype(np.float32) / 255.0)
    return x.permute(0, 3, 1, 2).to(device)


def single_obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    """(H, W, C) uint8 → (1, C, H, W) float32 on device."""
    img = obs_dict["image"]
    x   = torch.from_numpy(img.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_doorkey(
    agent:          PPOAgent,
    encoder_fn:     Callable,
    device:         str,
    seed_offset:    int = 1000,
    n_eps:          int = EVAL_EPISODES,
) -> float:
    """Returns success rate (fraction of episodes reaching the goal)."""
    successes = 0
    for ep in range(n_eps):
        env  = make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        lstm_state = agent.get_initial_state(1, device)
        done_t   = torch.zeros(1, device=device)
        done     = False
        ep_steps = 0
        ep_ret   = 0.0
        while not done and ep_steps < 300:
            with torch.no_grad():
                z = encoder_fn(obs)
                action, _, _, _, lstm_state = agent.get_action_and_value(
                    z, lstm_state, done_t
                )
            obs, r, term, trunc, _ = env.step(action.item())
            done   = term or trunc
            done_t = torch.tensor([float(done)], device=device)
            ep_ret  += r
            ep_steps += 1
        if ep_ret > 0.5:
            successes += 1
        env.close()
    return successes / n_eps


def eval_crafter(
    agent:       PPOAgent,
    encoder_fn:  Callable,
    device:      str,
    seed_offset: int = 1000,
    n_eps:       int = EVAL_EPISODES,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
      score       — fraction of 22 achievements unlocked ≥1 time across all eps
      per_tier    — {tier_name: fraction_unlocked} for tier-level analysis
    """
    from .crafter_env import make_crafter_env, ACHIEVEMENTS, ACHIEVEMENT_TIERS

    ever_unlocked: Dict[str, int] = {k: 0 for k in ACHIEVEMENTS}

    for ep in range(n_eps):
        env    = make_crafter_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        lstm_state = agent.get_initial_state(1, device)
        done_t = torch.zeros(1, device=device)
        done   = False
        ep_steps = 0

        while not done and ep_steps < 10_000:   # Crafter episodes are up to 10K steps
            with torch.no_grad():
                z = encoder_fn(obs)
                action, _, _, _, lstm_state = agent.get_action_and_value(
                    z, lstm_state, done_t
                )
            obs, _, term, trunc, info = env.step(action.item())
            done   = term or trunc
            done_t = torch.tensor([float(done)], device=device)
            ep_steps += 1

            for k, v in info.get("achievements", {}).items():
                if v and k in ever_unlocked:
                    ever_unlocked[k] = 1   # just need "at least once"
        env.close()

    n_total = len(ACHIEVEMENTS)
    score   = sum(ever_unlocked.values()) / n_total

    per_tier = {}
    for tier, ach_list in ACHIEVEMENT_TIERS.items():
        unlocked_in_tier = sum(ever_unlocked.get(a, 0) for a in ach_list)
        per_tier[tier] = unlocked_in_tier / len(ach_list)

    return score, per_tier


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_abm_loop(
    condition: str,
    device:    str = "cuda",
    max_steps: int = 400_000,
    seed:      int = 42,
    n_envs:    int = N_ENVS,
    env_type:  str = "doorkey",   # "doorkey" | "crafter"
) -> Dict:
    """
    Run a single condition of the A-B-M experiment.

    Parameters
    ----------
    condition : "autonomous" | "fixed" | "ppo_only"
    device    : "cuda" | "mps" | "cpu"
    max_steps : total environment steps
    seed      : random seed
    n_envs    : number of parallel environments
    env_type  : "doorkey" (MiniGrid) or "crafter" (Crafter survival)

    Returns
    -------
    metrics dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Environment-specific config ─────────────────────────────────────────
    if env_type == "crafter":
        from .crafter_env import make_crafter_env, make_crafter_vec_env
        img_h = img_w = 64
        n_actions     = 17
        _make_env     = make_crafter_env
        _make_vec     = make_crafter_vec_env
        # Crafter scores are naturally small — don't trigger ACT→OBSERVE
        # switches unless score is near zero AND flat
        min_sr_to_stay  = 0.03
        solve_threshold = 0.15   # ~3/22 achievements = meaningful progress
    else:  # doorkey
        img_h = img_w = IMG_H
        n_actions     = N_ACTIONS
        _make_env     = make_doorkey_env
        _make_vec     = make_doorkey_vec_env
        min_sr_to_stay  = 0.30   # raised from 0.20 — don't cut off improving LSTM
        solve_threshold = 0.80

    logger.info(
        f"[{condition.upper()}] Starting — env={env_type}, device={device}, "
        f"max_steps={max_steps}, n_envs={n_envs}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    use_async = (device == "cuda")
    envs = _make_vec(n_envs, seed=seed, use_async=use_async)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    # ── Model setup ─────────────────────────────────────────────────────────
    HIDDEN_SIZE = 256

    if condition == "ppo_only":
        flat_dim = img_h * img_w * 3
        agent    = PPOAgent(latent_dim=flat_dim, n_actions=n_actions, hidden=HIDDEN_SIZE).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(PPO_ROLLOUT, n_envs, flat_dim, device, hidden_size=HIDDEN_SIZE)
        lewm     = None
        opt_lewm = None
        buf_lew  = None

        def encoder(obs_dict):
            imgs = obs_dict["image"]
            x    = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            return x.reshape(len(imgs), -1).to(device)

        def encoder_single(obs_dict):
            img = obs_dict["image"]
            x   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return x.flatten().unsqueeze(0).to(device)

    else:
        lewm     = LeWM(latent_dim=LATENT_DIM, n_actions=n_actions).to(device)
        agent    = PPOAgent(latent_dim=LATENT_DIM, n_actions=n_actions, hidden=HIDDEN_SIZE).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(PPO_ROLLOUT, n_envs, LATENT_DIM, device, hidden_size=HIDDEN_SIZE)
        buf_lew  = ReplayBuffer(capacity=50_000)
        opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)

        def encoder(obs_dict):
            return lewm.encode(batch_obs_to_tensor(obs_dict, device))

        def encoder_single(obs_dict):
            return lewm.encode(single_obs_to_tensor(obs_dict, device))

    # ── System M ────────────────────────────────────────────────────────────
    if condition == "autonomous":
        sysm = AutonomousSystemM(
            obs_plateau_steps=8_000,
            act_plateau_steps=20_000,
            plateau_threshold=0.01,
            solve_threshold=solve_threshold,
            min_sr_to_stay=min_sr_to_stay,
        )
    elif condition == "fixed":
        sysm = FixedSystemM(
            switch_every=FIXED_SWITCH_EVERY,
            solve_threshold=solve_threshold,
        )
    else:
        sysm = None

    # ── Metric tracking ─────────────────────────────────────────────────────
    metrics: Dict[str, List] = {
        "env_step":     [],
        "success_rate": [],
        "ssl_loss_ewa": [],
        "mode":         [],
        "wall_time_s":  [],
        "per_tier":     [],    # Crafter only — tier achievement fractions
    }
    ssl_ewa          = None
    mode_str         = "OBSERVE" if condition != "ppo_only" else "ACT"
    steps_to_80      = None
    env_step         = 0
    encoder_frozen   = False
    SSL_FREEZE_THRESHOLD = 0.08

    ep_ret    = np.zeros(n_envs, dtype=np.float32)
    last_done = np.zeros(n_envs, dtype=bool)
    lstm_state = agent.get_initial_state(n_envs, device)

    # ── Main loop ────────────────────────────────────────────────────────────
    while env_step < max_steps:

        # Mode determination
        if condition == "ppo_only":
            current_mode = Mode.ACT
        elif encoder_frozen:
            current_mode = Mode.ACT
        elif condition == "autonomous":
            current_mode = sysm.mode
        else:
            current_mode = sysm.step(env_step)

        mode_str = "ACT(frozen)" if encoder_frozen else current_mode.name

        # ── OBSERVE step ─────────────────────────────────────────────────────
        if current_mode == Mode.OBSERVE:
            actions  = envs.action_space.sample()
            obs_imgs = obs["image"].copy()
            next_obs, _, terms, truncs, infos = envs.step(actions)
            next_imgs = next_obs["image"].copy()

            final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
            if final_mask.any() and "final_observation" in infos:
                for i in range(n_envs):
                    if final_mask[i]:
                        next_imgs[i] = infos["final_observation"]["image"][i]

            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions[i]), next_imgs[i])

            obs       = next_obs
            env_step += n_envs

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

            if (not encoder_frozen and ssl_ewa is not None
                    and ssl_ewa < SSL_FREEZE_THRESHOLD and condition != "ppo_only"):
                encoder_frozen = True
                for p in lewm.encoder.parameters():
                    p.requires_grad_(False)
                logger.info(
                    f"[{condition.upper()}] Encoder frozen at step {env_step} "
                    f"(ssl_ewa={ssl_ewa:.4f})"
                )

        # ── ACT step ─────────────────────────────────────────────────────────
        else:
            done_t = torch.tensor(last_done.astype(np.float32), device=device)

            if buf_ppo._ptr == 0:
                buf_ppo.set_lstm_initial_state(*lstm_state)

            with torch.no_grad():
                z = encoder(obs)
                actions, log_probs, _, values, lstm_state = agent.get_action_and_value(
                    z, lstm_state, done_t
                )

            next_obs, rewards, terms, truncs, _ = envs.step(actions.cpu().numpy())
            dones    = terms | truncs
            ep_ret  += rewards
            env_step += n_envs

            buf_ppo.add(
                z.detach(),
                actions,
                log_probs,
                torch.tensor(rewards, dtype=torch.float32, device=device),
                torch.tensor(dones.astype(np.float32), device=device),
                values,
            )

            for i in range(n_envs):
                if dones[i]:
                    if condition == "autonomous":
                        sysm.act_step(float(ep_ret[i]), None, env_step)
                    ep_ret[i] = 0.0

            last_done = dones
            obs       = next_obs

            if buf_ppo.is_full:
                last_done_t = torch.tensor(last_done.astype(np.float32), device=device)
                with torch.no_grad():
                    last_z   = encoder(obs)
                    last_val = agent.get_value(last_z, lstm_state, last_done_t)
                ppo.update(buf_ppo, last_val, last_done_t)

        # ── Periodic evaluation ──────────────────────────────────────────────
        if env_step % EVAL_INTERVAL < n_envs:
            if lewm is not None and not encoder_frozen:
                for p in lewm.encoder.parameters():
                    p.requires_grad_(False)

            if env_type == "crafter":
                sr, per_tier = eval_crafter(
                    agent, encoder_single, device,
                    seed_offset=9000 + env_step, n_eps=EVAL_EPISODES,
                )
            else:
                sr       = eval_doorkey(agent, encoder_single, device,
                                        seed_offset=9000 + env_step, n_eps=EVAL_EPISODES)
                per_tier = {}

            if lewm is not None and not encoder_frozen:
                for p in lewm.encoder.parameters():
                    p.requires_grad_(True)

            if condition == "autonomous":
                sysm.act_step(None, sr, env_step)

            n_sw    = sysm.n_switches() if sysm else 0
            elapsed = time.time() - t0
            frozen_tag = " [ENC FROZEN]" if encoder_frozen else ""
            tier_str   = (f" | tiers={per_tier}" if per_tier else "")
            logger.info(
                f"[{condition.upper()}] step={env_step:7d} | mode={mode_str:12s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"switches={n_sw}{frozen_tag}{tier_str} | {elapsed:.0f}s"
            )

            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append(per_tier)

            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[{condition.upper()}] *** 80% at step {env_step} ***")

        if sysm is not None and sysm.is_solved:
            logger.info(f"[{condition.upper()}] Solved! Stopping at step {env_step}.")
            break

    envs.close()
    elapsed_total = time.time() - t0

    # ── Save checkpoint ──────────────────────────────────────────────────────
    ckpt_dir = Path(f"results/{env_type}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if condition == "ppo_only":
        ckpt = {
            "condition":  condition,
            "agent":      agent.state_dict(),
            "flat_dim":   img_h * img_w * 3,
            "hidden_size": HIDDEN_SIZE,
            "env_step":   env_step,
            "env_type":   env_type,
        }
    else:
        ckpt = {
            "condition":  condition,
            "lewm":       lewm.state_dict(),
            "agent":      agent.state_dict(),
            "latent_dim": LATENT_DIM,
            "hidden_size": HIDDEN_SIZE,
            "n_actions":  n_actions,
            "env_step":   env_step,
            "env_type":   env_type,
        }
    ckpt_path = ckpt_dir / f"checkpoint_{condition}.pt"
    torch.save(ckpt, ckpt_path)
    logger.info(f"[{condition.upper()}] Checkpoint → {ckpt_path}")

    return {
        "condition":      condition,
        "env_type":       env_type,
        "env_steps":      metrics["env_step"],
        "success_rate":   metrics["success_rate"],
        "ssl_loss_ewa":   metrics["ssl_loss_ewa"],
        "mode":           metrics["mode"],
        "wall_time_s":    metrics["wall_time_s"],
        "per_tier":       metrics["per_tier"],
        "steps_to_80pct": steps_to_80,
        "n_switches":     sysm.n_switches() if sysm else 0,
        "switch_log":     sysm.switch_log if sysm else [],
        "total_time_s":   elapsed_total,
    }
