"""
abm/loop_mpc_doorkey_curiosity.py — Run 4: Curiosity-driven OBSERVE.

Identical to loop_mpc_doorkey.py (Run 3) except:
  OBSERVE uses curiosity-driven action selection instead of random walk.

For each step, the world model predicts the next latent for every possible
action (batched). The action whose predicted next-latent is most novel
(furthest from the running mean of all seen latents) is chosen. This
drives the agent toward key pickups, door interactions, and other rare
transitions that random walk almost never hits.

Scientific question: does better OBSERVE data collection fix the 0% result?
Saves to: results/doorkey/metrics_curiosity_observe.json
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

# ── Constants (identical to base) ─────────────────────────────────────────
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
EVAL_N_EPS        = 10
GOAL_REFRESH_STEPS = 64
N_TRAIN_STEPS     = 4
DEFAULT_OBSERVE   = 80_000
EP_MAX_STEPS      = 300
EBM_MIN_GOALS     = 5
EBM_WARMUP_STEPS  = 500
EBM_LR            = 3e-4
EBM_BATCH         = 32


# ── Environment helpers (identical to base) ────────────────────────────────

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


# ── Curiosity helpers ──────────────────────────────────────────────────────

def _curiosity_actions(lewm, obs, running_z_mean: torch.Tensor, device: str) -> np.ndarray:
    """
    For each env, predict next latent for all 7 actions. Pick the action
    whose predicted next latent is furthest from the running mean of all
    seen latents (novelty-maximizing exploration).
    """
    n = obs["image"].shape[0]
    with torch.no_grad():
        z_cur = lewm.encode(_obs_to_tensor(obs, device))                       # (N, D)
        z_rep = z_cur.unsqueeze(1).expand(n, N_ACTIONS, LATENT_DIM)
        z_rep = z_rep.reshape(n * N_ACTIONS, LATENT_DIM)
        a_idx = torch.arange(N_ACTIONS, device=device).unsqueeze(0).expand(n, -1).reshape(-1)
        a_oh  = F.one_hot(a_idx, N_ACTIONS).float()
        z_next = lewm.predictor(z_rep, a_oh).reshape(n, N_ACTIONS, LATENT_DIM) # (N, A, D)
        novelty = (z_next - running_z_mean.view(1, 1, LATENT_DIM)).pow(2).sum(-1)  # (N, A)
        actions = novelty.argmax(dim=-1).cpu().numpy()
    return actions


# ── Evaluation (identical to base) ────────────────────────────────────────

def _eval_doorkey_mpc(mpc, encoder_fn, goal_buf, device, seed_offset=1000, n_eps=EVAL_N_EPS):
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
            z_goal = encoder_fn({"image": goal_img[None]}) if goal_img is not None else None
        while not done and ep_steps < EP_MAX_STEPS:
            with torch.no_grad():
                z = encoder_fn(obs)
            action = mpc.plan_single(z, z_goal) if (mpc is not None and z_goal is not None) else env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r
            ep_steps += 1
            goal_age += 1
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


# ── Main loop ──────────────────────────────────────────────────────────────

def run_doorkey_curiosity_loop(
    condition: str = "curiosity_observe",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "curiosity_observe":
        raise ValueError(f"loop_mpc_doorkey_curiosity supports: curiosity_observe — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_curiosity.py only supports env_type='doorkey'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE
    logger.info(
        f"[{condition.upper()}] Run 4: Curiosity-driven OBSERVE — device={device}, "
        f"max_steps={max_steps}, n_envs={n_envs}, observe={_observe_steps}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    lewm = LeWM(latent_dim=LATENT_DIM, n_actions=N_ACTIONS, img_size=IMG_H, predictor_type="mlp").to(device)
    buf_lew = ReplayBuffer(capacity=REPLAY_CAPACITY)
    buf_seq = SequenceReplayBuffer(capacity=REPLAY_CAPACITY, seq_len=8)
    opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)
    mpc: Optional[CEMPlanner] = None
    goal_buf = GoalImageBuffer(capacity=GOAL_CAPACITY)

    ebm = EBMCostHead(latent_dim=LATENT_DIM).to(device)
    opt_ebm = optim.Adam(ebm.parameters(), lr=EBM_LR)
    ebm_train_count = 0
    ebm_active = False

    # Curiosity: running mean of seen latents (EWA)
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

    metrics: Dict = {"env_step": [], "success_rate": [], "ssl_loss_ewa": [], "mode": [], "wall_time_s": [], "per_tier": []}
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
    mode_str = "OBSERVE"

    while env_step < max_steps:

        in_observe = (env_step < _observe_steps)

        if in_observe:
            mode_str = "OBSERVE"

            # ── Curiosity-driven action selection ──────────────────────────
            if len(buf_lew) >= LEWM_WARMUP:
                actions = _curiosity_actions(lewm, obs, running_z_mean, device)
                # Update running mean with current observations
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
                if rewards[i] > 0:
                    goal_buf.push(next_imgs[i])

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

                if len(goal_buf) >= EBM_MIN_GOALS and len(buf_lew) >= EBM_BATCH * 2:
                    pos_imgs  = goal_buf.sample(EBM_BATCH)
                    goal_imgs = goal_buf.sample(EBM_BATCH)
                    neg_obs, _, _ = buf_lew.sample(EBM_BATCH, device)
                    with torch.no_grad():
                        z_pos = lewm.encode(torch.from_numpy(pos_imgs.astype(np.float32)/255.0).permute(0,3,1,2).to(device))
                        z_goal_ebm = lewm.encode(torch.from_numpy(goal_imgs.astype(np.float32)/255.0).permute(0,3,1,2).to(device))
                        z_neg = lewm.encode(neg_obs)
                    opt_ebm.zero_grad()
                    ebm.contrastive_loss(z_pos, z_neg, z_goal_ebm).backward()
                    opt_ebm.step()
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(f"[{condition.upper()}] EBM activated")

                if mpc is None:
                    mpc = CEMPlanner(lewm.predictor, n_actions=N_ACTIONS, horizon=30, n_samples=512, n_elites=64, n_iters=5, device=device, distance="cosine")
                    logger.info(f"[{condition.upper()}] CEM ready (H=30) | goals={len(goal_buf)}")

        else:
            mode_str = "ACT"

            refresh_mask = last_done.copy() | (goal_ages >= GOAL_REFRESH_STEPS)
            refresh_mask |= np.array([img is None for img in active_goal_imgs], dtype=bool)
            if len(goal_buf) > 0 and refresh_mask.any():
                sampled = goal_buf.sample(int(refresh_mask.sum()))
                if sampled is not None:
                    for slot, env_idx in enumerate(np.flatnonzero(refresh_mask)):
                        active_goal_imgs[int(env_idx)] = sampled[slot]

            if mpc is not None and len(goal_buf) > 0:
                with torch.no_grad():
                    z = encoder_fn(obs)
                    valid = [img for img in active_goal_imgs if img is not None]
                    if valid:
                        goal_stack = np.stack(active_goal_imgs if all(img is not None for img in active_goal_imgs) else [valid[0]] * n_envs, axis=0)
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
                if rewards[i] > 0:
                    goal_buf.push(next_imgs[i])
                goal_ages[i] += 1
                if dones[i]:
                    ep_ret[i] = 0.0
                    active_goal_imgs[i] = None
                    goal_ages[i] = 0

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

                if len(goal_buf) >= EBM_MIN_GOALS and len(buf_lew) >= EBM_BATCH * 2:
                    pos_imgs  = goal_buf.sample(EBM_BATCH)
                    goal_imgs = goal_buf.sample(EBM_BATCH)
                    neg_obs, _, _ = buf_lew.sample(EBM_BATCH, device)
                    with torch.no_grad():
                        z_pos = lewm.encode(torch.from_numpy(pos_imgs.astype(np.float32)/255.0).permute(0,3,1,2).to(device))
                        z_goal_ebm = lewm.encode(torch.from_numpy(goal_imgs.astype(np.float32)/255.0).permute(0,3,1,2).to(device))
                        z_neg = lewm.encode(neg_obs)
                    opt_ebm.zero_grad()
                    ebm.contrastive_loss(z_pos, z_neg, z_goal_ebm).backward()
                    opt_ebm.step()
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(f"[{condition.upper()}] EBM activated")

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            logger.info(
                f"[{condition.upper()}] heartbeat step={env_step} | buf={len(buf_lew)} | "
                f"goals={len(goal_buf)} | ssl={ssl_ewa or 0.0:.4f} | "
                f"ebm={'ON' if ebm_active else f'training({ebm_train_count})'} | {time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_doorkey_mpc(mpc=mpc, encoder_fn=encoder_fn, goal_buf=goal_buf, device=device, seed_offset=9000 + env_step, n_eps=EVAL_N_EPS)
            elapsed = time.time() - t0
            logger.info(
                f"[{condition.upper()}] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | goals={len(goal_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({})
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


run_abm_loop = run_doorkey_curiosity_loop
