"""
abm/loop_mpc_doorkey_subgoals.py — Run 6: Programmatic subgoal sequencing.

DoorKey requires three sequential subgoals:
  Stage 0: pick up the key          (z_key_buf images)
  Stage 1: unlock and open the door (z_door_buf images)
  Stage 2: reach the exit           (z_goal_buf images)

Without subgoals, CEM plans H=30 steps toward the exit goal from the start.
With a 30-step horizon, the agent can't simultaneously pick up the key, navigate
to the door, unlock it, and reach the exit — they're 30–80 steps apart.

Subgoal sequencing breaks the problem into three separately-tractable CEM tasks.
Each CEM invocation only needs to plan to the NEXT subgoal, not the full sequence.

Subgoal buffers are seeded by:
  1. N_SEED_EPS random episodes run BEFORE OBSERVE — look for key pickups, door
     opens, and successful exits, store images in the appropriate buffer.
  2. During OBSERVE and ACT, any step that triggers a subgoal updates the buffer.

Stage detection (per-env, live from env internals):
  env.envs[i].unwrapped.carrying is not None  → agent has key
  grid scan for cell.type=='door' and is_open  → door is open
  reward > 0                                   → exit reached

Scientific question: does breaking the task into stages let CEM succeed at each
sub-problem individually, ultimately solving the full task?
Saves to: results/doorkey_mpc/checkpoint_subgoals.pt
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
EVAL_INTERVAL      = 5_000
EVAL_N_EPS         = 10
GOAL_REFRESH_STEPS = 64
N_TRAIN_STEPS      = 4
DEFAULT_OBSERVE    = 80_000
EP_MAX_STEPS       = 300
EBM_MIN_GOALS      = 5
EBM_WARMUP_STEPS   = 500
EBM_LR             = 3e-4
EBM_BATCH          = 32
N_SEED_EPS         = 200            # random episodes for subgoal buffer seeding


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
    """Return True if any door on this env's grid is open."""
    grid = uw.grid
    for col in range(grid.width):
        for row in range(grid.height):
            cell = grid.get(col, row)
            if cell is not None and cell.type == "door" and cell.is_open:
                return True
    return False


def _seed_subgoal_buffers(
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    n_eps: int = N_SEED_EPS,
    seed: int = 7777,
) -> None:
    """
    Run N_SEED_EPS random episodes on a single env. Record the frame at each
    subgoal transition: key pickup, door open, episode success.

    Uses random actions — we're just trying to collect incidental examples of
    each subgoal state to seed the buffers before the main loop starts.
    """
    logger.info(f"[SUBGOALS] Seeding subgoal buffers with {n_eps} random episodes…")
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

            img = _resize_imgs(obs["image"])  # (1, H, W, 3)

            # Key pickup
            if not had_key and uw.carrying is not None:
                key_buf.push(img[0])
                had_key = True

            # Door open
            if not door_was_open and _is_door_open(uw):
                door_buf.push(img[0])
                door_was_open = True

            # Success
            if r > 0:
                goal_buf.push(img[0])

        env.close()

    logger.info(
        f"[SUBGOALS] Seed complete: key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
    )


def _eval_doorkey_subgoals(
    mpc: Optional[CEMPlanner],
    encoder_fn,
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    device: str,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> float:
    """
    Eval with subgoal-aware planning: at each step, detect the current stage
    and direct CEM toward the appropriate next subgoal image.
    Falls back to random if the required buffer is empty.
    """
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
        goal_age = 0
        current_stage = 0

        # Pre-sample one goal image per stage
        def _pick_goal(stage):
            if stage == 0:
                return key_buf.sample(1)
            elif stage == 1:
                return door_buf.sample(1)
            else:
                return goal_buf.sample(1)

        goal_imgs = _pick_goal(current_stage)
        with torch.no_grad():
            z_goal = encoder_fn({"image": goal_imgs}) if goal_imgs is not None else None

        while not done and ep_steps < EP_MAX_STEPS:
            # Detect stage transitions
            has_key = uw.carrying is not None
            door_open = _is_door_open(uw)
            new_stage = 0 if not has_key else (1 if not door_open else 2)

            if new_stage != current_stage:
                current_stage = new_stage
                goal_imgs = _pick_goal(current_stage)
                with torch.no_grad():
                    z_goal = encoder_fn({"image": goal_imgs}) if goal_imgs is not None else None
                goal_age = 0

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

            if goal_age >= GOAL_REFRESH_STEPS:
                goal_imgs = _pick_goal(current_stage)
                if goal_imgs is not None:
                    with torch.no_grad():
                        z_goal = encoder_fn({"image": goal_imgs})
                goal_age = 0

        if ep_ret > 0.5:
            successes += 1
        env.close()

    return successes / n_eps


def run_doorkey_subgoals_loop(
    condition: str = "subgoals",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "subgoals":
        raise ValueError(f"loop_mpc_doorkey_subgoals supports: subgoals — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_subgoals.py only supports env_type='doorkey'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE
    logger.info(
        f"[{condition.upper()}] Run 6: Subgoal sequencing — device={device}, "
        f"max_steps={max_steps}, n_envs={n_envs}, observe={_observe_steps}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Subgoal buffers — seeded before any training begins
    key_buf  = GoalImageBuffer(capacity=GOAL_CAPACITY)
    door_buf = GoalImageBuffer(capacity=GOAL_CAPACITY)
    goal_buf = GoalImageBuffer(capacity=GOAL_CAPACITY)
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

    def encoder_fn(obs_input):
        if isinstance(obs_input, dict):
            return lewm.encode(_obs_to_tensor(obs_input, device))
        arr = obs_input.astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.permute(0, 3, 1, 2).to(device)
        return lewm.encode(t)

    # Per-env tracking
    had_key = np.zeros(n_envs, dtype=bool)
    door_open_flags = np.zeros(n_envs, dtype=bool)
    active_goal_imgs: List[Optional[np.ndarray]] = [None] * n_envs
    active_stage = np.zeros(n_envs, dtype=np.int32)   # 0=key, 1=door, 2=exit
    goal_ages = np.zeros(n_envs, dtype=np.int32)
    last_done = np.zeros(n_envs, dtype=bool)

    metrics: Dict = {"env_step": [], "success_rate": [], "ssl_loss_ewa": [], "mode": [], "wall_time_s": [], "per_tier": []}
    ssl_ewa = None
    ssl_loss_val = None
    steps_to_80 = None
    env_step = 0
    act_steps = 0
    total_observe_steps = 0
    ep_ret = np.zeros(n_envs, dtype=np.float32)
    mode_str = "OBSERVE"

    while env_step < max_steps:

        in_observe = (env_step < _observe_steps)

        # ── Subgoal detection: check env internals after each step ─────────
        # Done after envs.step() in both phases.

        if in_observe:
            mode_str = "OBSERVE"
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

                # Collect subgoal examples online
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

                # EBM — train on whichever subgoal buffers are ready (not just goal_buf)
                any_buf_ready = any(len(b) >= EBM_MIN_GOALS for b in [key_buf, door_buf, goal_buf])
                if any_buf_ready and len(buf_lew) >= EBM_BATCH * 2:
                    _train_subgoal_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, buf_lew, lewm, device)
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(f"[{condition.upper()}] EBM activated")

                if mpc is None:
                    mpc = CEMPlanner(lewm.predictor, n_actions=N_ACTIONS, horizon=30, n_samples=512, n_elites=64, n_iters=5, device=device, distance="cosine")
                    logger.info(
                        f"[{condition.upper()}] CEM ready (H=30) | "
                        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
                    )

        else:
            mode_str = "ACT"

            # Determine current stage per env and assign subgoal image
            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key = uw.carrying is not None
                d_open  = _is_door_open(uw)
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

                any_buf_ready = any(len(b) >= EBM_MIN_GOALS for b in [key_buf, door_buf, goal_buf])
                if any_buf_ready and len(buf_lew) >= EBM_BATCH * 2:
                    _train_subgoal_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, buf_lew, lewm, device)
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(f"[{condition.upper()}] EBM activated")

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            logger.info(
                f"[{condition.upper()}] heartbeat step={env_step} | buf={len(buf_lew)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} | "
                f"ssl={ssl_ewa or 0.0:.4f} | ebm={'ON' if ebm_active else f'training({ebm_train_count})'} | "
                f"{time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_doorkey_subgoals(
                mpc=mpc, encoder_fn=encoder_fn,
                key_buf=key_buf, door_buf=door_buf, goal_buf=goal_buf,
                device=device, seed_offset=9000 + env_step, n_eps=EVAL_N_EPS,
            )
            elapsed = time.time() - t0
            logger.info(
                f"[{condition.upper()}] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({"key": len(key_buf), "door": len(door_buf), "goal": len(goal_buf)})
            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[{condition.upper()}] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0
    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "lewm": lewm.state_dict()}, ckpt_dir / f"checkpoint_{condition}.pt")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"], "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"], "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps, "observe_steps": total_observe_steps,
    }


def _train_subgoal_ebm(
    ebm: EBMCostHead,
    opt_ebm: optim.Optimizer,
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    buf_lew: ReplayBuffer,
    lewm: LeWM,
    device: str,
) -> None:
    """Train EBM with positive samples from whichever subgoal buffers are populated."""
    # Pool all available subgoal images and sample from the combined set
    all_bufs = [(b, name) for b, name in [(key_buf, "key"), (door_buf, "door"), (goal_buf, "goal")] if len(b) >= EBM_MIN_GOALS]
    if not all_bufs or len(buf_lew) < EBM_BATCH * 2:
        return

    buf, _ = all_bufs[np.random.randint(len(all_bufs))]
    pos_imgs  = buf.sample(EBM_BATCH)
    goal_imgs = buf.sample(EBM_BATCH)
    neg_obs, _, _ = buf_lew.sample(EBM_BATCH, device)

    with torch.no_grad():
        z_pos  = lewm.encode(torch.from_numpy(pos_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device))
        z_g    = lewm.encode(torch.from_numpy(goal_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device))
        z_neg  = lewm.encode(neg_obs)

    opt_ebm.zero_grad()
    ebm.contrastive_loss(z_pos, z_neg, z_g).backward()
    opt_ebm.step()


run_abm_loop = run_doorkey_subgoals_loop
