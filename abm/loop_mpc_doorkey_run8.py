"""
abm/loop_mpc_doorkey_run8.py — Run 8: Short-horizon MPC + subgoals + curiosity + HER.

Diagnosis from Runs 1–7:
  Run 1: H=10 too short
  Run 2: H=30 cosine dist fails
  Run 3: H=30 EBM, only 19 goal images — EBM starved
  Run 4: H=30 + curiosity — curiosity alone insufficient
  Run 5: H=30 + HER — HER alone insufficient (0% at step 115k)
  Run 6: H=30 + subgoals — 416 key images, EBM ON, still 0%

The one constant across all failures: H=30. A 30-step rollout through an MLP
predictor at ~65% per-step accuracy gives 0.65^30 ≈ 0.00002 coherent trajectories
out of 512 CEM samples. CEM is optimizing pure noise regardless of cost quality.

Fix: H=5 with replanning every step (receding-horizon MPC).
  At H=5: 0.65^5 ≈ 11.6% coherent trajectories → ~59 usable CEM samples.
  With subgoals, each CEM call only needs to plan toward the NEAREST subgoal
  (key / door / exit), not the full 30-80 step task.
  7^5 = 16,807 total sequences; 512 samples covers ~3% per CEM call — tractable.

Also includes:
  - Curiosity during OBSERVE (novelty-driven exploration, more key/door images)
  - HER EBM (terminal-state relabeling, thousands of positives vs 19)
  - Subgoal seeding + live detection (key=400+ before OBSERVE even starts)
  - Subgoal-aware planning in eval (stage-conditional goal selection)

Condition: short_horizon
Loop module: abm.loop_mpc_doorkey_run8
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import RGBImgObsWrapper

from .world_model import LeWM, ReplayBuffer, SequenceReplayBuffer
from .cem_planner import CEMPlanner, EBMCostHead

logger = logging.getLogger(__name__)

N_ENVS             = 16
IMG_H = IMG_W      = 48
N_ACTIONS          = 7
LATENT_DIM         = 256
LEWM_LR            = 1e-4
LEWM_BATCH         = 256
LEWM_WARMUP        = 500
REPLAY_CAPACITY    = 100_000
GOAL_CAPACITY      = 1_024
HER_CAPACITY       = 4_096
EVAL_INTERVAL      = 5_000
EVAL_N_EPS         = 10
GOAL_REFRESH_STEPS = 64
N_TRAIN_STEPS      = 4
DEFAULT_OBSERVE    = 80_000
EP_MAX_STEPS       = 300
EBM_MIN_GOALS      = 5
EBM_HER_MIN        = 20
EBM_WARMUP_STEPS   = 500
EBM_LR             = 3e-4
EBM_BATCH          = 32
N_SEED_EPS         = 200

# ── Key change: H=5 instead of H=30 ───────────────────────────────────────
CEM_HORIZON        = 5     # was 30; 0.65^5 ≈ 11.6% coherent vs 0.65^30 ≈ 0.002%
CEM_SAMPLES        = 512
CEM_ELITES         = 64
CEM_ITERS          = 5


def _make_doorkey_env(seed: int = 0):
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    return env


def _make_doorkey_vec_env(n_envs: int, seed: int = 0):
    fns = [lambda i=i: _make_doorkey_env(seed=seed + i) for i in range(n_envs)]
    return gymnasium.vector.SyncVectorEnv(fns)


def _resize_imgs(imgs: np.ndarray) -> np.ndarray:
    if imgs.ndim == 3:
        imgs = imgs[None]
    if imgs.shape[1] == IMG_H and imgs.shape[2] == IMG_W:
        return imgs
    t = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    return (t.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)


def _obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    imgs = obs_dict["image"]
    if imgs.ndim == 3:
        imgs = imgs[None]
    arr = imgs.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
    if t.shape[-2:] != (IMG_H, IMG_W):
        t = F.interpolate(t, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    return t


class GoalImageBuffer:
    def __init__(self, capacity: int = GOAL_CAPACITY):
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, img: np.ndarray) -> None:
        self._buf.append(np.array(img, copy=True))

    def sample(self, n: int) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=n, replace=len(self._buf) < n)
        return np.stack([self._buf[i] for i in idx], axis=0)

    def __len__(self) -> int:
        return len(self._buf)


def _is_door_open(uw) -> bool:
    grid = uw.grid
    for col in range(grid.width):
        for row in range(grid.height):
            cell = grid.get(col, row)
            if cell is not None and cell.type == "door" and cell.is_open:
                return True
    return False


def _curiosity_actions(lewm, obs, running_z_mean: torch.Tensor, device: str) -> np.ndarray:
    """Pick the action with the most novel predicted next-latent per env."""
    n = obs["image"].shape[0]
    with torch.no_grad():
        z_cur = lewm.encode(_obs_to_tensor(obs, device))
        z_rep = z_cur.unsqueeze(1).expand(n, N_ACTIONS, LATENT_DIM).reshape(n * N_ACTIONS, LATENT_DIM)
        a_idx = torch.arange(N_ACTIONS, device=device).unsqueeze(0).expand(n, -1).reshape(-1)
        a_oh  = F.one_hot(a_idx, N_ACTIONS).float()
        z_next = lewm.predictor(z_rep, a_oh).reshape(n, N_ACTIONS, LATENT_DIM)
        novelty = (z_next - running_z_mean.view(1, 1, LATENT_DIM)).pow(2).sum(-1)
        actions = novelty.argmax(dim=-1).cpu().numpy()
    return actions


def _train_ebm(
    ebm: EBMCostHead,
    opt_ebm: optim.Optimizer,
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    her_buf: GoalImageBuffer,
    buf_lew: ReplayBuffer,
    lewm: LeWM,
    device: str,
) -> bool:
    """
    Train EBM with up to two signals per call:
      1. Subgoal contrastive: pick a ready subgoal buffer, (z_pos, z_goal) vs (z_neg, z_goal)
      2. HER: (z_terminal, z_terminal) vs (z_random, z_terminal)
    Returns True if any gradient step was taken.
    """
    if len(buf_lew) < EBM_BATCH * 2:
        return False

    opt_ebm.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    # Subgoal contrastive signal — pick whichever buffer is ready
    ready = [(b, name) for b, name in [(key_buf, "key"), (door_buf, "door"), (goal_buf, "goal")]
             if len(b) >= EBM_MIN_GOALS]
    if ready:
        buf, _ = ready[np.random.randint(len(ready))]
        pos_imgs  = buf.sample(EBM_BATCH)
        goal_imgs = buf.sample(EBM_BATCH)
        neg_obs, _, _ = buf_lew.sample(EBM_BATCH, device)
        with torch.no_grad():
            z_pos  = lewm.encode(torch.from_numpy(pos_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device))
            z_g    = lewm.encode(torch.from_numpy(goal_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device))
            z_neg  = lewm.encode(neg_obs)
        total_loss = total_loss + ebm.contrastive_loss(z_pos, z_neg, z_g)
        n_terms += 1

    # HER signal — every terminal state is its own goal
    if len(her_buf) >= EBM_HER_MIN:
        her_imgs  = her_buf.sample(EBM_BATCH)
        neg_obs2, _, _ = buf_lew.sample(EBM_BATCH, device)
        with torch.no_grad():
            z_her  = lewm.encode(torch.from_numpy(her_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device))
            z_neg2 = lewm.encode(neg_obs2)
        total_loss = total_loss + ebm.contrastive_loss(z_her, z_neg2, z_her.detach())
        n_terms += 1

    if n_terms > 0:
        total_loss.backward()
        opt_ebm.step()
    return n_terms > 0


def _seed_subgoal_buffers(
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    n_eps: int = N_SEED_EPS,
    seed: int = 7777,
) -> None:
    logger.info(f"[RUN8] Seeding subgoal buffers with {n_eps} random episodes…")
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        uw = env.unwrapped
        had_key = False
        door_was_open = False
        done = False
        steps = 0
        while not done and steps < EP_MAX_STEPS:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1
            img = _resize_imgs(obs["image"])
            if not had_key and uw.carrying is not None:
                key_buf.push(img[0])
                had_key = True
            if not door_was_open and _is_door_open(uw):
                door_buf.push(img[0])
                door_was_open = True
            if r > 0:
                goal_buf.push(img[0])
        env.close()
    logger.info(
        f"[RUN8] Seed done: key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
    )


def _eval_run8(
    mpc: Optional[CEMPlanner],
    encoder_fn,
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    device: str,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> float:
    """Subgoal-aware eval: detect current stage, direct CEM toward next subgoal."""
    if mpc is None:
        return 0.0
    successes = 0

    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        uw = env.unwrapped
        done = False
        ep_steps = 0
        ep_ret = 0.0
        current_stage = 0
        goal_age = 0

        def _pick_goal(stage):
            if stage == 0:
                return key_buf.sample(1)
            elif stage == 1:
                return door_buf.sample(1)
            else:
                return goal_buf.sample(1)

        goal_imgs = _pick_goal(0)
        with torch.no_grad():
            z_goal = encoder_fn({"image": goal_imgs}) if goal_imgs is not None else None

        while not done and ep_steps < EP_MAX_STEPS:
            has_key  = uw.carrying is not None
            d_open   = _is_door_open(uw)
            new_stage = 0 if not has_key else (1 if not d_open else 2)
            if new_stage != current_stage or goal_age >= GOAL_REFRESH_STEPS:
                current_stage = new_stage
                goal_imgs = _pick_goal(current_stage)
                with torch.no_grad():
                    z_goal = encoder_fn({"image": goal_imgs}) if goal_imgs is not None else None
                goal_age = 0

            with torch.no_grad():
                z = encoder_fn(obs)
            action = mpc.plan_single(z, z_goal) if (mpc is not None and z_goal is not None) else env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r
            ep_steps += 1
            goal_age += 1

        if ep_ret > 0.5:
            successes += 1
        env.close()

    return successes / n_eps


def run_doorkey_run8_loop(
    condition: str = "short_horizon",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "short_horizon":
        raise ValueError(f"loop_mpc_doorkey_run8 supports: short_horizon — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_run8.py only supports env_type='doorkey'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE
    logger.info(
        f"[RUN8] Short-horizon MPC (H={CEM_HORIZON}) + subgoals + curiosity + HER | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Seed subgoal buffers before any training
    key_buf  = GoalImageBuffer(capacity=GOAL_CAPACITY)
    door_buf = GoalImageBuffer(capacity=GOAL_CAPACITY)
    goal_buf = GoalImageBuffer(capacity=GOAL_CAPACITY)
    her_buf  = GoalImageBuffer(capacity=HER_CAPACITY)
    _seed_subgoal_buffers(key_buf, door_buf, goal_buf, n_eps=N_SEED_EPS, seed=seed + 999)

    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    lewm = LeWM(latent_dim=LATENT_DIM, n_actions=N_ACTIONS, img_size=IMG_H, predictor_type="mlp").to(device)
    buf_lew = ReplayBuffer(capacity=REPLAY_CAPACITY)
    buf_seq = SequenceReplayBuffer(capacity=REPLAY_CAPACITY, seq_len=8)
    opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)
    mpc: Optional[CEMPlanner] = None

    ebm = EBMCostHead(latent_dim=LATENT_DIM).to(device)
    opt_ebm = optim.Adam(ebm.parameters(), lr=EBM_LR)
    ebm_train_count = 0
    ebm_active = False

    running_z_mean = torch.zeros(LATENT_DIM, device=device)

    def encoder_fn(obs_input):
        if isinstance(obs_input, dict):
            return lewm.encode(_obs_to_tensor(obs_input, device))
        arr = obs_input.astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.permute(0, 3, 1, 2).to(device)
        return lewm.encode(t)

    had_key         = np.zeros(n_envs, dtype=bool)
    door_open_flags = np.zeros(n_envs, dtype=bool)
    active_goal_imgs: List[Optional[np.ndarray]] = [None] * n_envs
    active_stage    = np.zeros(n_envs, dtype=np.int32)
    goal_ages       = np.zeros(n_envs, dtype=np.int32)
    last_done       = np.zeros(n_envs, dtype=bool)
    ep_ret          = np.zeros(n_envs, dtype=np.float32)

    metrics: Dict = {"env_step": [], "success_rate": [], "ssl_loss_ewa": [], "mode": [], "wall_time_s": [], "per_tier": []}
    ssl_ewa = None
    ssl_loss_val = None
    steps_to_80 = None
    env_step = 0
    act_steps = 0
    total_observe_steps = 0
    mode_str = "OBSERVE"

    while env_step < max_steps:

        in_observe = (env_step < _observe_steps)

        if in_observe:
            mode_str = "OBSERVE"

            # Curiosity-driven action selection once model is warm
            if len(buf_lew) >= LEWM_WARMUP:
                actions = _curiosity_actions(lewm, obs, running_z_mean, device)
                with torch.no_grad():
                    z_obs = lewm.encode(_obs_to_tensor(obs, device))
                    running_z_mean.mul_(0.99).add_(z_obs.mean(0).detach() * 0.01)
            else:
                actions = envs.action_space.sample()

            obs_imgs = _resize_imgs(obs["image"].copy())
            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            next_imgs = _resize_imgs(next_obs["image"].copy())

            final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
            if final_mask.any() and "final_observation" in infos:
                for i in range(n_envs):
                    if final_mask[i]:
                        next_imgs[i] = _resize_imgs(infos["final_observation"]["image"][i])

            dones = terms | truncs
            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions[i]), next_imgs[i])
                buf_seq.push(obs_imgs[i], int(actions[i]), bool(dones[i]))

                uw = envs.envs[i].unwrapped
                if not had_key[i] and uw.carrying is not None:
                    key_buf.push(next_imgs[i])
                    had_key[i] = True
                if not door_open_flags[i] and _is_door_open(uw):
                    door_buf.push(next_imgs[i])
                    door_open_flags[i] = True
                if rewards[i] > 0:
                    goal_buf.push(next_imgs[i])
                if dones[i]:
                    her_buf.push(next_imgs[i])
                    had_key[i] = False
                    door_open_flags[i] = False

            obs = next_obs
            env_step += n_envs
            total_observe_steps += n_envs

            if len(buf_lew) >= LEWM_WARMUP:
                for _ in range(N_TRAIN_STEPS):
                    obs_t, acts, obs_next = buf_lew.sample(LEWM_BATCH, device)
                    opt_lewm.zero_grad()
                    loss, info = lewm.loss(obs_t, acts, obs_next)
                    loss.backward()
                    opt_lewm.step()
                ssl_loss_val = info["loss_total"]

                if len(buf_seq) > buf_seq.seq_len * 2:
                    seq_data = buf_seq.sample_sequences(batch_size=64, encoder_fn=lewm.encode, n_actions=N_ACTIONS, device=device)
                    if seq_data is not None:
                        z_seq, a_oh_seq, z_next_t = seq_data
                        opt_lewm.zero_grad()
                        z_pred = lewm.predictor.forward_sequence(z_seq, a_oh_seq)
                        seq_loss = (1 - F.cosine_similarity(z_pred[:, -1], z_next_t.detach(), dim=-1)).mean()
                        seq_loss.backward()
                        opt_lewm.step()

                ssl_ewa = ssl_loss_val if ssl_ewa is None else 0.95 * ssl_ewa + 0.05 * ssl_loss_val

                if _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, buf_lew, lewm, device):
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(
                            f"[RUN8] EBM activated | "
                            f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} her={len(her_buf)}"
                        )

                if mpc is None:
                    mpc = CEMPlanner(
                        lewm.predictor,
                        n_actions=N_ACTIONS,
                        horizon=CEM_HORIZON,    # H=5 — the key change
                        n_samples=CEM_SAMPLES,
                        n_elites=CEM_ELITES,
                        n_iters=CEM_ITERS,
                        device=device,
                        distance="cosine",
                    )
                    logger.info(
                        f"[RUN8] CEM ready (H={CEM_HORIZON}) | "
                        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
                    )

        else:
            mode_str = "ACT"

            # Determine current subgoal per env from live env state
            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key  = uw.carrying is not None
                d_open   = _is_door_open(uw)
                new_stage = 0 if not has_key else (1 if not d_open else 2)

                if new_stage != active_stage[i] or active_goal_imgs[i] is None or goal_ages[i] >= GOAL_REFRESH_STEPS:
                    active_stage[i] = new_stage
                    buf = key_buf if new_stage == 0 else (door_buf if new_stage == 1 else goal_buf)
                    sampled = buf.sample(1) if len(buf) > 0 else None
                    active_goal_imgs[i] = sampled[0] if sampled is not None else None
                    goal_ages[i] = 0

            if mpc is not None:
                with torch.no_grad():
                    z = encoder_fn(obs)
                    valid = [img for img in active_goal_imgs if img is not None]
                    if valid:
                        goal_stack = np.stack(
                            active_goal_imgs if all(img is not None for img in active_goal_imgs) else [valid[0]] * n_envs,
                            axis=0,
                        )
                        z_goal = encoder_fn({"image": goal_stack})
                        actions_np = mpc.plan_batch(z, z_goal)
                    else:
                        actions_np = envs.action_space.sample()
            else:
                actions_np = envs.action_space.sample()

            obs_imgs = _resize_imgs(obs["image"].copy())
            next_obs, rewards, terms, truncs, infos = envs.step(actions_np)
            next_imgs = _resize_imgs(next_obs["image"].copy())

            final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
            if final_mask.any() and "final_observation" in infos:
                for i in range(n_envs):
                    if final_mask[i]:
                        next_imgs[i] = _resize_imgs(infos["final_observation"]["image"][i])

            dones = terms | truncs
            ep_ret += rewards
            env_step += n_envs
            act_steps += n_envs

            for i in range(n_envs):
                buf_lew.push(obs_imgs[i], int(actions_np[i]), next_imgs[i])
                buf_seq.push(obs_imgs[i], int(actions_np[i]), bool(dones[i]))

                uw = envs.envs[i].unwrapped
                if not had_key[i] and uw.carrying is not None:
                    key_buf.push(next_imgs[i])
                    had_key[i] = True
                if not door_open_flags[i] and _is_door_open(uw):
                    door_buf.push(next_imgs[i])
                    door_open_flags[i] = True
                if rewards[i] > 0:
                    goal_buf.push(next_imgs[i])
                if dones[i]:
                    her_buf.push(next_imgs[i])
                    had_key[i] = False
                    door_open_flags[i] = False
                    ep_ret[i] = 0.0
                    active_goal_imgs[i] = None
                    goal_ages[i] = 0
                else:
                    goal_ages[i] += 1

            last_done = dones
            obs = next_obs

            if len(buf_lew) >= LEWM_WARMUP:
                for _ in range(N_TRAIN_STEPS):
                    obs_t, acts, obs_next = buf_lew.sample(LEWM_BATCH, device)
                    opt_lewm.zero_grad()
                    loss, info = lewm.loss(obs_t, acts, obs_next)
                    loss.backward()
                    opt_lewm.step()
                ssl_loss_val = info["loss_total"]
                ssl_ewa = 0.95 * ssl_ewa + 0.05 * ssl_loss_val

                if _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, buf_lew, lewm, device):
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(f"[RUN8] EBM activated")

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            logger.info(
                f"[RUN8] heartbeat step={env_step} | buf={len(buf_lew)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} her={len(her_buf)} | "
                f"ssl={ssl_ewa or 0.0:.4f} | ebm={'ON' if ebm_active else f'training({ebm_train_count})'} | "
                f"{time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_run8(
                mpc=mpc, encoder_fn=encoder_fn,
                key_buf=key_buf, door_buf=door_buf, goal_buf=goal_buf,
                device=device, seed_offset=9000 + env_step, n_eps=EVAL_N_EPS,
            )
            elapsed = time.time() - t0
            logger.info(
                f"[RUN8] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"H={CEM_HORIZON} | key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({"key": len(key_buf), "door": len(door_buf), "goal": len(goal_buf)})
            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[RUN8] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0
    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "lewm": lewm.state_dict()}, ckpt_dir / f"checkpoint_{condition}.pt")
    logger.info(f"[RUN8] Checkpoint saved | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"], "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"], "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps, "observe_steps": total_observe_steps,
    }


run_abm_loop = run_doorkey_run8_loop
