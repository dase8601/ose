"""
abm/loop.py — Main A↔B training loop (vectorized environments).

run_abm_loop(condition, device, max_steps, seed, env_type) → metrics dict

Three conditions:
  "autonomous" — AutonomousSystemM (plateau-triggered switching)
  "fixed"      — FixedSystemM (switch every K steps)
  "ppo_only"   — Raw-pixel PPO baseline (no LeWM, no mode switching)

Three environments (env_type):
  "doorkey"   — MiniGrid-DoorKey-6x6 (original, 48×48, 7 actions)
  "crafter"   — Crafter survival game (64×64, 17 actions, 22 achievements)
  "miniworld" — MiniWorld 3D maze navigation (160×160, 3 actions)
                Uses V-JEPA 2.1 ViT-B as frozen System A encoder.

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

from .lewm import LeWM, ReplayBuffer, VJEPAPredictor, VJEPAReplayBuffer
from .ppo import PPO, PPOAgent, RolloutBuffer
from .meta_controller import AutonomousSystemM, FixedSystemM, Mode
from .rnd import RND

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

        while not done and ep_steps < 1_000:   # cap at 1K for fast eval (full ep is 10K)
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
# MiniWorld evaluation
# ---------------------------------------------------------------------------

def eval_miniworld(
    agent:       PPOAgent,
    encoder_fn:  Callable,
    device:      str,
    seed_offset: int = 1000,
    n_eps:       int = 20,
) -> float:
    """
    Evaluate on MiniWorld maze navigation.
    Returns: success_rate (fraction of episodes reaching the goal)
    """
    from .miniworld_env import make_miniworld_env

    successes = 0
    for ep in range(n_eps):
        env = make_miniworld_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        lstm_state = agent.get_initial_state(1, device)
        done_t = torch.zeros(1, device=device)
        done   = False
        ep_steps = 0
        ep_ret   = 0.0

        while not done and ep_steps < 500:
            with torch.no_grad():
                z = encoder_fn(obs)
                action, _, _, _, lstm_state = agent.get_action_and_value(
                    z, lstm_state, done_t
                )
            obs, r, term, trunc, info = env.step(action.item())
            done   = term or trunc
            done_t = torch.tensor([float(done)], device=device)
            ep_ret  += r
            ep_steps += 1

        # MiniWorld gives +1 reward on goal reach
        if ep_ret > 0:
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
    env_type:  str = "doorkey",   # "doorkey" | "crafter" | "miniworld"
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
    env_type  : "doorkey" | "crafter" | "miniworld"

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
        n_actions           = 17
        _make_env           = make_crafter_env
        _make_vec           = make_crafter_vec_env
        latent_dim          = 512     # richer visual content needs more capacity
        ppo_rollout         = 256     # 17 actions → noisier gradient, need larger batch
        eval_interval       = 10_000  # Crafter eval is expensive
        eval_n_eps          = 20      # 20 eps × 1K steps each
        ssl_freeze_thr      = 0.06    # tighter threshold for richer visuals
        min_sr_to_stay      = 0.30    # don't switch back to OBSERVE too aggressively
        solve_threshold     = 1.01    # never auto-stop — run full budget
        use_rnd             = True    # intrinsic reward for sparse Crafter signal
        rnd_coef            = 0.1     # scale of intrinsic vs extrinsic reward
        obs_plateau_steps   = 20_000  # longer OBSERVE phases for visual complexity
        act_plateau_steps   = 100_000 # long ACT — let PPO train uninterrupted
        min_initial_observe = 50_000  # deep first OBSERVE (LeCun: observe then act)
        n_train_steps       = 2       # gradient steps per env step during OBSERVE
        use_vjepa           = False
    elif env_type == "miniworld":
        from .miniworld_env import make_miniworld_env, make_miniworld_vec_env
        img_h = img_w       = 160
        n_actions           = 3       # turn_left, turn_right, move_forward
        _make_env           = make_miniworld_env
        _make_vec           = lambda n, seed, use_async: make_miniworld_vec_env(
                                  n, seed=seed, use_async=False, img_size=160)  # sync — OpenGL can't share X across processes
        latent_dim          = 768     # V-JEPA 2.1 ViT-B feature dim
        ppo_rollout         = 128
        eval_interval       = 10_000
        eval_n_eps          = 20
        ssl_freeze_thr      = 0.02    # predictor loss threshold
        min_sr_to_stay      = 0.10    # navigation success is harder
        solve_threshold     = 1.01    # never auto-stop
        use_rnd             = False   # predictor error replaces RND
        rnd_coef            = 0.0
        obs_plateau_steps   = 30_000  # V-JEPA predictor needs time to learn transitions
        act_plateau_steps   = 80_000  # long ACT phases for navigation learning
        min_initial_observe = 40_000  # deep first OBSERVE
        n_train_steps       = 4       # predictor is lightweight, train intensively
        use_vjepa           = True    # use V-JEPA 2.1 encoder instead of LeWM CNN
        intrinsic_coef      = 0.1     # predictor-based intrinsic reward scale
    else:  # doorkey
        img_h = img_w       = IMG_H
        n_actions           = N_ACTIONS
        _make_env           = make_doorkey_env
        _make_vec           = make_doorkey_vec_env
        latent_dim          = LATENT_DIM
        ppo_rollout         = PPO_ROLLOUT
        eval_interval       = EVAL_INTERVAL
        eval_n_eps          = EVAL_EPISODES
        ssl_freeze_thr      = 0.08
        min_sr_to_stay      = 0.30
        solve_threshold     = 0.80
        use_rnd             = False
        rnd_coef            = 0.0
        obs_plateau_steps   = 8_000
        act_plateau_steps   = 20_000
        min_initial_observe = 0
        n_train_steps       = 1       # standard training rate for simple env
        use_vjepa           = False

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

    # V-JEPA encoder (shared across conditions for habitat)
    vjepa_enc    = None
    vjepa_pred   = None
    opt_pred     = None
    buf_vjepa    = None

    if use_vjepa:
        # ── V-JEPA 2.1 path (habitat) ──────────────────────────────────────
        from .vjepa_encoder import VJEPAEncoder

        vjepa_enc = VJEPAEncoder(device=device)
        feat_dim  = vjepa_enc.feature_dim   # 768 for ViT-B
        logger.info(f"[{condition.upper()}] DINOv3 ViT-B/16 loaded — feature_dim={feat_dim}")

        agent   = PPOAgent(latent_dim=feat_dim, n_actions=n_actions, hidden=HIDDEN_SIZE).to(device)
        ppo     = PPO(agent, lr=PPO_LR)
        buf_ppo = RolloutBuffer(ppo_rollout, n_envs, feat_dim, device, hidden_size=HIDDEN_SIZE)

        # V-JEPA encoder handles both "rgb" and "image" keys
        obs_key = "rgb"

        def encoder(obs_dict):
            return vjepa_enc.encode(obs_dict)

        def encoder_single(obs_dict):
            return vjepa_enc.encode_single(obs_dict)

        if condition != "ppo_only":
            # Action-conditioned predictor (trains during OBSERVE)
            vjepa_pred = VJEPAPredictor(
                feature_dim=feat_dim, n_actions=n_actions
            ).to(device)
            opt_pred = optim.Adam(vjepa_pred.parameters(), lr=3e-4)
            buf_vjepa = VJEPAReplayBuffer(capacity=50_000, feature_dim=feat_dim)

        # LeWM not used in V-JEPA path
        lewm     = None
        opt_lewm = None
        buf_lew  = None

    elif condition == "ppo_only":
        # ── Raw pixel baseline ──────────────────────────────────────────────
        obs_key  = "image"
        flat_dim = img_h * img_w * 3
        agent    = PPOAgent(latent_dim=flat_dim, n_actions=n_actions, hidden=HIDDEN_SIZE).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(ppo_rollout, n_envs, flat_dim, device, hidden_size=HIDDEN_SIZE)
        lewm     = None
        opt_lewm = None
        buf_lew  = None

        def encoder(obs_dict):
            imgs = obs_dict[obs_key]
            x    = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            return x.reshape(len(imgs), -1).to(device)

        def encoder_single(obs_dict):
            img = obs_dict[obs_key]
            x   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return x.flatten().unsqueeze(0).to(device)

        rnd_input_dim = flat_dim
    else:
        # ── LeWM CNN path (doorkey / crafter) ───────────────────────────────
        obs_key  = "image"
        lewm     = LeWM(latent_dim=latent_dim, n_actions=n_actions, img_size=img_h).to(device)
        agent    = PPOAgent(latent_dim=latent_dim, n_actions=n_actions, hidden=HIDDEN_SIZE).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(ppo_rollout, n_envs, latent_dim, device, hidden_size=HIDDEN_SIZE)
        buf_lew  = ReplayBuffer(capacity=50_000)
        opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)

        def encoder(obs_dict):
            return lewm.encode(batch_obs_to_tensor(obs_dict, device))

        def encoder_single(obs_dict):
            return lewm.encode(single_obs_to_tensor(obs_dict, device))

        rnd_input_dim = latent_dim

    # ── RND intrinsic reward (Crafter only — habitat uses predictor error) ─
    rnd_module = None
    opt_rnd    = None
    if use_rnd:
        rnd_input_dim = latent_dim if not use_vjepa else vjepa_enc.feature_dim
        rnd_module = RND(input_dim=rnd_input_dim).to(device)
        opt_rnd    = optim.Adam(rnd_module.predictor.parameters(), lr=1e-4)

    # ── System M ────────────────────────────────────────────────────────────
    if condition == "autonomous":
        sysm = AutonomousSystemM(
            obs_plateau_steps=obs_plateau_steps,
            act_plateau_steps=act_plateau_steps,
            plateau_threshold=0.01,
            solve_threshold=solve_threshold,
            min_sr_to_stay=min_sr_to_stay,
            min_initial_observe=min_initial_observe,
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

    ep_ret    = np.zeros(n_envs, dtype=np.float32)
    last_done = np.zeros(n_envs, dtype=bool)
    lstm_state = agent.get_initial_state(n_envs, device)

    # ── Main loop ────────────────────────────────────────────────────────────
    while env_step < max_steps:

        # Mode determination — System M always controls switching.
        # encoder_frozen no longer forces ACT: the encoder stays frozen (no grad)
        # but System M can still switch to OBSERVE to collect new replay data
        # and optionally unfreeze the encoder for retraining.
        if condition == "ppo_only":
            current_mode = Mode.ACT
        elif condition == "autonomous":
            current_mode = sysm.mode
        else:
            current_mode = sysm.step(env_step)

        mode_str = current_mode.name
        if encoder_frozen:
            mode_str += "(enc_frozen)"

        # ── OBSERVE step ─────────────────────────────────────────────────────
        if current_mode == Mode.OBSERVE:

            if use_vjepa:
                # V-JEPA path: collect transitions in feature space,
                # train action-conditioned predictor
                actions_np = envs.action_space.sample()

                with torch.no_grad():
                    z_t = encoder(obs)

                next_obs, _, terms, truncs, infos = envs.step(actions_np)

                with torch.no_grad():
                    z_next = encoder(next_obs)

                # Store in V-JEPA replay buffer
                buf_vjepa.push_batch(
                    z_t.cpu().numpy(),
                    actions_np if isinstance(actions_np, np.ndarray) else np.array(actions_np),
                    z_next.cpu().numpy(),
                )

                obs       = next_obs
                env_step += n_envs

                # Train predictor
                ssl_loss_val = None
                if len(buf_vjepa) >= LEWM_WARMUP and vjepa_pred is not None:
                    for _ in range(n_train_steps):
                        s_zt, s_act, s_zn = buf_vjepa.sample(LEWM_BATCH, device)
                        opt_pred.zero_grad()
                        loss, info = vjepa_pred.loss(s_zt, s_act, s_zn)
                        loss.backward()
                        opt_pred.step()
                    ssl_loss_val = info["predictor_loss"]
                    ssl_ewa = (ssl_loss_val if ssl_ewa is None
                               else 0.95 * ssl_ewa + 0.05 * ssl_loss_val)

                if condition == "autonomous" and ssl_loss_val is not None:
                    sysm.observe_step(ssl_loss_val, env_step)

            else:
                # LeWM CNN path (doorkey / crafter)
                if encoder_frozen and lewm is not None:
                    encoder_frozen = False
                    for p in lewm.encoder.parameters():
                        p.requires_grad_(True)
                    logger.info(
                        f"[{condition.upper()}] Encoder unfrozen at step {env_step} "
                        f"— System M returned to OBSERVE"
                    )

                actions  = envs.action_space.sample()
                obs_imgs = obs[obs_key].copy()
                next_obs, _, terms, truncs, infos = envs.step(actions)
                next_imgs = next_obs[obs_key].copy()

                final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
                if final_mask.any() and "final_observation" in infos:
                    for i in range(n_envs):
                        if final_mask[i]:
                            next_imgs[i] = infos["final_observation"][obs_key][i]

                for i in range(n_envs):
                    buf_lew.push(obs_imgs[i], int(actions[i]), next_imgs[i])

                obs       = next_obs
                env_step += n_envs

                ssl_loss_val = None
                if len(buf_lew) >= LEWM_WARMUP:
                    for _ in range(n_train_steps):
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
                        and ssl_ewa < ssl_freeze_thr and condition != "ppo_only"):
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

            # Intrinsic reward: predictor error (habitat) or RND (crafter)
            combined_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            if vjepa_pred is not None and use_vjepa:
                with torch.no_grad():
                    z_next = encoder(next_obs)
                    intrinsic = vjepa_pred.intrinsic_reward(
                        z.detach(), actions, z_next.detach()
                    )
                combined_rewards = combined_rewards + intrinsic_coef * intrinsic
            elif rnd_module is not None:
                with torch.no_grad():
                    intrinsic = rnd_module.reward(z.detach())
                combined_rewards = combined_rewards + rnd_coef * intrinsic

            buf_ppo.add(
                z.detach(),
                actions,
                log_probs,
                combined_rewards,
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

                # Grab RND training data before PPO update resets the buffer
                if rnd_module is not None and opt_rnd is not None:
                    rnd_z = buf_ppo.latents.reshape(-1, buf_ppo.latent_dim).detach().clone()

                ppo.update(buf_ppo, last_val, last_done_t)

                # Train RND predictor on the rollout's latents
                if rnd_module is not None and opt_rnd is not None:
                    rnd_loss = rnd_module.loss(rnd_z)
                    opt_rnd.zero_grad()
                    rnd_loss.backward()
                    opt_rnd.step()

        # ── Periodic evaluation ──────────────────────────────────────────────
        if env_step % eval_interval < n_envs:
            if lewm is not None and not encoder_frozen:
                for p in lewm.encoder.parameters():
                    p.requires_grad_(False)

            if env_type == "miniworld":
                sr = eval_miniworld(
                    agent, encoder_single, device,
                    seed_offset=9000 + env_step, n_eps=eval_n_eps,
                )
                per_tier = {}
            elif env_type == "crafter":
                sr, per_tier = eval_crafter(
                    agent, encoder_single, device,
                    seed_offset=9000 + env_step, n_eps=eval_n_eps,
                )
            else:
                sr       = eval_doorkey(agent, encoder_single, device,
                                        seed_offset=9000 + env_step, n_eps=eval_n_eps)
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
    ckpt = {
        "condition":   condition,
        "agent":       agent.state_dict(),
        "hidden_size": HIDDEN_SIZE,
        "n_actions":   n_actions,
        "env_step":    env_step,
        "env_type":    env_type,
    }
    if use_vjepa and vjepa_pred is not None:
        ckpt["vjepa_predictor"] = vjepa_pred.state_dict()
        ckpt["feature_dim"]    = vjepa_enc.feature_dim
    elif condition == "ppo_only":
        ckpt["flat_dim"] = img_h * img_w * 3
    elif lewm is not None:
        ckpt["lewm"]       = lewm.state_dict()
        ckpt["latent_dim"] = latent_dim
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
