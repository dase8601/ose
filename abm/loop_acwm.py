"""
abm/loop_acwm.py — Action-Conditioned World Model (ACWM) training loop.

This module is a more LeCun-aligned alternative to loop.py for Crafter:
  - OBSERVE: learn an action-conditioned world model from passive data
  - ACT: plan directly in latent space with CEM/MPC
  - System M: heuristically switch between OBSERVE and ACT

It intentionally removes PPO from the action phase. The file is scoped to
Crafter first because that is the baseline that was just completed and it is
the hardest environment in the current repo to stress-test a world-model loop.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .crafter_env import ACHIEVEMENTS, ACHIEVEMENT_TIERS, make_crafter_env, make_crafter_vec_env
from .lewm import LeWM, ReplayBuffer, SequenceReplayBuffer
from .meta_controller import AutonomousSystemM, FixedSystemM, Mode
from .mpc import CEMPlanner

logger = logging.getLogger(__name__)

# Crafter-first ACWM defaults
N_ENVS = 16
LEWM_LR = 1e-4
LEWM_BATCH = 256
LEWM_WARMUP = 500
FIXED_SWITCH_EVERY = 10_000


def _obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    """Accepts either a single obs dict or a vectorized obs dict."""
    imgs = obs_dict["image"]
    if imgs.ndim == 3:
        imgs = imgs[None]
    arr = imgs.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)


class GoalImageBuffer:
    """
    Stores raw goal images instead of latent vectors so goals stay valid even
    when the encoder keeps changing during OBSERVE.
    """

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, img: np.ndarray) -> None:
        self._buf.append(np.array(img, copy=True))

    def sample(self, batch_size: int) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(
            len(self._buf),
            size=batch_size,
            replace=len(self._buf) < batch_size,
        )
        return np.stack([self._buf[i] for i in idx], axis=0)

    def __len__(self) -> int:
        return len(self._buf)


def _assign_goal_images(
    active_goal_imgs: List[Optional[np.ndarray]],
    refresh_mask: np.ndarray,
    goal_buf: GoalImageBuffer,
) -> None:
    n_refresh = int(refresh_mask.sum())
    if n_refresh == 0:
        return
    sampled = goal_buf.sample(n_refresh)
    if sampled is None:
        return
    for slot, env_idx in enumerate(np.flatnonzero(refresh_mask)):
        active_goal_imgs[int(env_idx)] = sampled[slot]


def _encode_goal_batch(
    active_goal_imgs: List[Optional[np.ndarray]],
    encoder_fn,
    device: str,
) -> Optional[torch.Tensor]:
    if not active_goal_imgs or any(img is None for img in active_goal_imgs):
        return None
    goal_imgs = np.stack(active_goal_imgs, axis=0)
    with torch.no_grad():
        return encoder_fn({"image": goal_imgs})


def _goal_diversity(goal_buf: GoalImageBuffer, encoder_fn, device: str) -> float:
    if len(goal_buf) < 2:
        return 0.0
    sample = goal_buf.sample(min(32, len(goal_buf)))
    if sample is None:
        return 0.0
    with torch.no_grad():
        z = encoder_fn({"image": sample})
    return z.std(dim=0).mean().item()


def _push_if_interesting(goal_buf: GoalImageBuffer, next_img: np.ndarray, reward: float) -> None:
    # Crafter rewards are sparse and tied to first-time achievement unlocks.
    if reward > 0:
        goal_buf.push(next_img)


def eval_crafter_acwm(
    mpc: Optional[CEMPlanner],
    encoder_fn,
    goal_buf: GoalImageBuffer,
    device: str,
    seed_offset: int = 1000,
    n_eps: int = 10,
    goal_refresh_steps: int = 64,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the planner on Crafter using achievement-rich states as latent goals.

    This is intentionally cheaper than the PPO evaluation path because every ACT
    step runs MPC. The metric is still the same Crafter achievement score.
    """

    ever_unlocked: Dict[str, int] = {k: 0 for k in ACHIEVEMENTS}

    for ep in range(n_eps):
        env = make_crafter_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_steps = 0
        goal_age = 0
        goal_img = goal_buf.sample(1) if len(goal_buf) > 0 else None

        while not done and ep_steps < 1_000:
            with torch.no_grad():
                z = encoder_fn(obs)

            if mpc is not None and goal_img is not None:
                with torch.no_grad():
                    z_goal = encoder_fn({"image": goal_img})
                action = mpc.plan_single(z, z_goal)
            else:
                action = env.action_space.sample()

            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_steps += 1

            for k, v in info.get("achievements", {}).items():
                if v and k in ever_unlocked:
                    ever_unlocked[k] = 1

            if reward > 0:
                goal_buf.push(next_obs["image"])
                goal_img = goal_buf.sample(1)
                goal_age = 0
            else:
                goal_age += 1
                if goal_age >= goal_refresh_steps and len(goal_buf) > 0:
                    goal_img = goal_buf.sample(1)
                    goal_age = 0

            obs = next_obs

        env.close()

    score = sum(ever_unlocked.values()) / len(ACHIEVEMENTS)
    per_tier = {}
    for tier, ach_list in ACHIEVEMENT_TIERS.items():
        unlocked_in_tier = sum(ever_unlocked.get(a, 0) for a in ach_list)
        per_tier[tier] = unlocked_in_tier / len(ach_list)
    return score, per_tier


def run_acwm_loop(
    condition: str,
    device: str = "cuda",
    max_steps: int = 500_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "crafter",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    """
    Crafter-first ACWM loop.

    Supported conditions:
      - autonomous: heuristic System M controls switching
      - fixed: fixed-schedule System M controls switching
      - mpc_only: always ACT with MPC when ready, else random
      - random: always ACT with random actions
    """

    if env_type != "crafter":
        raise ValueError("loop_acwm.py currently supports env_type='crafter' only.")
    if condition not in ("autonomous", "fixed", "mpc_only", "random"):
        raise ValueError(f"Unsupported ACWM condition: {condition}")
    if use_rl:
        logger.info("[ACWM] Ignoring use_rl=True — ACT is planner-only in loop_acwm.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Crafter-focused configuration learned from the PPO baseline run.
    img_h = img_w = 64
    n_actions = 17
    latent_dim = 512
    eval_interval = 10_000
    eval_n_eps = 10
    replay_capacity = 300_000
    goal_refresh_steps = 64
    goal_capacity = 2_048
    obs_plateau_steps = 30_000
    act_plateau_steps = 60_000
    min_initial_observe = 150_000
    min_sr_to_stay = 0.30
    solve_threshold = 1.01
    n_train_steps = 4
    predictor_type = "transformer"

    if observe_steps is not None:
        min_initial_observe = observe_steps
        logger.info(f"[{condition.upper()}] observe_steps override: min_initial_observe={observe_steps}")

    logger.info(
        f"[{condition.upper()}] Starting ACWM — env={env_type}, device={device}, "
        f"max_steps={max_steps}, n_envs={n_envs}, replay={replay_capacity}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    use_async = device == "cuda"
    envs = make_crafter_vec_env(n_envs, seed=seed, use_async=use_async)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    lewm = LeWM(
        latent_dim=latent_dim,
        n_actions=n_actions,
        img_size=img_h,
        predictor_type=predictor_type,
    ).to(device)
    buf_lew = ReplayBuffer(capacity=replay_capacity)
    buf_seq = SequenceReplayBuffer(capacity=replay_capacity, seq_len=8)
    opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)
    mpc: Optional[CEMPlanner] = None
    goal_buf = GoalImageBuffer(capacity=goal_capacity)

    def encoder_fn(obs_dict):
        return lewm.encode(_obs_to_tensor(obs_dict, device))

    if condition == "autonomous":
        sysm = AutonomousSystemM(
            obs_plateau_steps=obs_plateau_steps,
            act_plateau_steps=act_plateau_steps,
            plateau_threshold=0.03,
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

    metrics: Dict[str, List] = {
        "env_step": [],
        "success_rate": [],
        "ssl_loss_ewa": [],
        "mode": [],
        "wall_time_s": [],
        "per_tier": [],
    }
    ssl_ewa = None
    steps_to_80 = None
    env_step = 0
    act_steps = 0
    total_observe_steps = 0
    ep_ret = np.zeros(n_envs, dtype=np.float32)
    last_done = np.zeros(n_envs, dtype=bool)
    active_goal_imgs: List[Optional[np.ndarray]] = [None for _ in range(n_envs)]
    goal_ages = np.zeros(n_envs, dtype=np.int32)

    while env_step < max_steps:
        if condition in ("mpc_only", "random"):
            current_mode = Mode.ACT
        elif condition == "autonomous":
            current_mode = sysm.mode
        else:
            current_mode = sysm.step(env_step)

        mode_str = current_mode.name
        ssl_loss_val = None
        executed_mode = current_mode

        if current_mode == Mode.OBSERVE:
            actions = envs.action_space.sample()
            obs_imgs = obs["image"].copy()
            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            next_imgs = next_obs["image"].copy()

            final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
            if final_mask.any() and "final_observation" in infos:
                for i in range(n_envs):
                    if final_mask[i]:
                        next_imgs[i] = infos["final_observation"]["image"][i]

            dones = terms | truncs
            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions[i]), next_imgs[i])
                buf_seq.push(obs_imgs[i], int(actions[i]), bool(dones[i]))
                _push_if_interesting(goal_buf, next_imgs[i], float(rewards[i]))

            obs = next_obs
            env_step += n_envs
            total_observe_steps += n_envs

            if len(buf_lew) >= LEWM_WARMUP:
                for _ in range(n_train_steps):
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
                        n_actions=n_actions,
                        device=device,
                    )
                    if seq_data is not None:
                        z_seq, a_onehot_seq, z_next_target = seq_data
                        opt_lewm.zero_grad()
                        z_pred = lewm.predictor.forward_sequence(z_seq, a_onehot_seq)
                        seq_loss = (1 - torch.nn.functional.cosine_similarity(
                            z_pred,
                            z_next_target.detach(),
                            dim=-1,
                        )).mean()
                        seq_loss.backward()
                        opt_lewm.step()

                ssl_ewa = ssl_loss_val if ssl_ewa is None else 0.95 * ssl_ewa + 0.05 * ssl_loss_val

                if condition == "autonomous":
                    observe_signal = ssl_ewa if ssl_ewa is not None else ssl_loss_val
                    sysm.observe_step(observe_signal, env_step)

                if mpc is None:
                    mpc = CEMPlanner(
                        lewm.predictor,
                        n_actions=n_actions,
                        horizon=15,
                        n_samples=256,
                        n_elites=32,
                        n_iters=4,
                        device=device,
                        distance="cosine",
                    )
                    logger.info(
                        f"[{condition.upper()}] ACWM planner ready for Crafter "
                        f"(horizon=15, samples=256, elites=32, iters=4)"
                    )

                if (
                    condition == "autonomous"
                    and sysm.mode == Mode.OBSERVE
                    and sysm.n_switches() == 0
                    and env_step >= min_initial_observe
                    and mpc is not None
                    and len(goal_buf) > 0
                ):
                    sysm.force_switch(
                        Mode.ACT,
                        env_step,
                        "initial observe budget satisfied; planner ready; goals available",
                    )

        else:
            refresh_mask = last_done.copy() | (goal_ages >= goal_refresh_steps)
            refresh_mask |= np.array([img is None for img in active_goal_imgs], dtype=bool)
            if len(goal_buf) > 0:
                _assign_goal_images(active_goal_imgs, refresh_mask, goal_buf)

            if mpc is not None and len(goal_buf) > 0 and condition != "random":
                with torch.no_grad():
                    z = encoder_fn(obs)
                    z_goal = _encode_goal_batch(active_goal_imgs, encoder_fn, device)
                if z_goal is not None:
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
                _push_if_interesting(goal_buf, next_imgs[i], float(rewards[i]))

                goal_ages[i] += 1
                if rewards[i] > 0 or dones[i]:
                    active_goal_imgs[i] = None
                    goal_ages[i] = 0

                if dones[i]:
                    if condition == "autonomous":
                        sysm.act_step(float(ep_ret[i]), None, env_step)
                    ep_ret[i] = 0.0

            last_done = dones
            obs = next_obs

        if env_step % eval_interval < n_envs:
            sr, per_tier = eval_crafter_acwm(
                mpc=mpc,
                encoder_fn=encoder_fn,
                goal_buf=goal_buf,
                device=device,
                seed_offset=9000 + env_step,
                n_eps=eval_n_eps,
                goal_refresh_steps=goal_refresh_steps,
            )

            if condition == "autonomous" and executed_mode == Mode.ACT:
                sysm.act_step(None, sr, env_step)

            if sysm is not None:
                mode_str = sysm.mode.name

            n_sw = sysm.n_switches() if sysm else 0
            elapsed = time.time() - t0
            logger.info(
                f"[{condition.upper()}] step={env_step:7d} | mode={mode_str:12s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"act_steps={act_steps:7d} | obs_steps={total_observe_steps:7d} | "
                f"switches={n_sw} | tiers={per_tier} | {elapsed:.0f}s"
            )

            detail_parts = [
                f"replay={len(buf_lew)}",
                f"goals={len(goal_buf)}",
                f"active_goals={sum(img is not None for img in active_goal_imgs)}",
            ]
            if mpc is not None:
                detail_parts.append("mpc=ready")
                detail_parts.append(f"cem_cost={getattr(mpc, '_last_best_cost', 0.0):.4f}")
            else:
                detail_parts.append("mpc=NOT_READY")
            if ssl_loss_val is not None:
                detail_parts.append(f"wm_loss={ssl_loss_val:.4f}")
            if len(goal_buf) > 1:
                detail_parts.append(f"goal_div={_goal_diversity(goal_buf, encoder_fn, device):.4f}")
            if condition == "autonomous" and sysm is not None:
                detail_parts.append(
                    f"sysm(ssl_buf={len(sysm._ssl_buf)},sr_buf={len(sysm._sr_buf)},time_in={env_step - sysm._mode_start})"
                )
            logger.info(f"  details: {' | '.join(detail_parts)}")

            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append(per_tier)

            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[{condition.upper()}] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/crafter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{condition}_acwm.pt"
    torch.save(
        {
            "condition": condition,
            "env_type": env_type,
            "env_step": env_step,
            "latent_dim": latent_dim,
            "n_actions": n_actions,
            "replay_capacity": replay_capacity,
            "goal_capacity": goal_capacity,
            "lewm": lewm.state_dict(),
        },
        ckpt_path,
    )
    logger.info(f"[{condition.upper()}] ACWM checkpoint → {ckpt_path}")

    return {
        "condition": condition,
        "env_type": env_type,
        "env_steps": metrics["env_step"],
        "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"],
        "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"],
        "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80,
        "n_switches": sysm.n_switches() if sysm else 0,
        "switch_log": sysm.switch_log if sysm else [],
        "total_time_s": elapsed_total,
        "act_steps": act_steps,
        "observe_steps": total_observe_steps,
    }


# Drop-in compatibility with abm_experiment.py dynamic import.
run_abm_loop = run_acwm_loop
