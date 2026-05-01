"""
abm/loop_mpc_crafter_run34.py — Run 34: Two-level CEM, no RL (pure planning hierarchy)

Run 32 conclusion: REINFORCE on sparse achievement reward cannot learn prerequisite
ordering. Drops the manager policy and REINFORCE entirely.

Replaces with a two-level CEM planner:
  High-level CEM: every H_HIGH=450 steps, plans a sequence of S=3 subgoals by
    sampling N_HIGH=256 random codebook triplets and scoring each by:
    cost = w1 * cos_dist(cb[k1], z_cur)          ← k1 is reachable from now
         + w2 * cos_dist(cb[k2], cb[k1])          ← k2 is reachable from k1
         + w3 * cos_dist(cb[k3], z_final_goal)    ← k3 lands near an achievement
    The triplet with minimum cost is selected.

  Low-level CEM: executes each subgoal for H_LOW=150 primitive steps using
    standard cosine-distance CEM, identical to Runs 29-32.

No REINFORCE. No manager policy weights. Ordering emerges from the high-level
CEM discovering that path cost is minimized by intermediate stepping stones —
without any gradient signal.

Codebook still built from replay at OBSERVE end and refreshed every 100k ACT
steps (same as Run 32), so tier3-adjacent states accumulate over ACT phase.

Key differences from Run 32:
  - SubgoalManager + REINFORCE removed
  - High-level CEM selects triplets of subgoal codes every H_HIGH=450 steps
  - Low-level CEM executes each subgoal for H_LOW=150 steps
  - z_final_goal drawn from goal_buf (achievement-positive) at each H_HIGH step
  - Codebook refresh every 100k ACT steps (unchanged)

Condition:   lewm_crafter_twolevel
Loop module: abm.loop_mpc_crafter_run34
RunPod:
  pip install timm crafter scikit-learn wandb moviepy
  python abm_experiment.py --loop-module abm.loop_mpc_crafter_run34 \\
    --condition lewm_crafter_twolevel --device cuda --env crafter \\
    --steps 600000 --n-envs 8
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

from .cem_planner import CEMPlanner
from .crafter_env import ACHIEVEMENTS, ACHIEVEMENT_TIERS, make_crafter_env, make_crafter_vec_env
from .world_model import Predictor, sigreg

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE        = 64
Z_DIM           = 256
N_ACTIONS       = 17
N_ENVS          = 8
REPLAY_CAP      = 30_000
GOAL_BUF_CAP    = 2_000
TRAIN_FREQ      = 16
BATCH_SIZE      = 256
SIGREG_LAMBDA   = 0.05
PRED_LR         = 3e-4
OBSERVE_DEFAULT = 300_000
EP_MAX_STEPS    = 1_000
EVAL_INTERVAL   = 10_000
EVAL_N_EPS      = 10
CEM_SAMPLES     = 512
CEM_ELITES      = 50
CEM_ITERS       = 10
TRAIN_WARMUP    = BATCH_SIZE
# Two-level CEM
N_CODES                  = 64
H_LOW                    = 150     # primitive steps per subgoal
H_HIGH                   = 450     # primitive steps per subgoal sequence (S=3 × H_LOW)
SUBGOAL_SEQ_LEN          = 3       # number of subgoals per high-level plan
N_HIGH_SAMPLES           = 256     # triplets evaluated per high-level CEM call
W1, W2, W3              = 0.3, 0.3, 0.4   # cost weights: reachability, path, goal
CODEBOOK_REFRESH_INTERVAL = 100_000
VIDEO_LOG_INTERVAL        = 50_000


# ── ViT-Tiny encoder ───────────────────────────────────────────────────────────

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


# ── Pixel replay buffer ────────────────────────────────────────────────────────

class PixelReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE):
        self.capacity  = capacity
        self._obs_t    = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions  = np.zeros(capacity, dtype=np.int64)
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
        return (to_t(self._obs_t[idx]),
                torch.from_numpy(self._actions[idx]).long().to(device),
                to_t(self._obs_next[idx]))

    def sample_raw(self, n):
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return self._obs_t[idx]

    def __len__(self):
        return self._size


# ── Goal buffer ────────────────────────────────────────────────────────────────

class GoalPixelBuffer:
    def __init__(self, capacity: int = GOAL_BUF_CAP):
        self._buf: deque = deque(maxlen=capacity)

    def add(self, pix_hwc):
        self._buf.append(pix_hwc.copy())

    def sample_raw(self, n=1):
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self):
        return len(self._buf)


# ── Pixel helpers ──────────────────────────────────────────────────────────────

def _extract_pix(obs):
    if isinstance(obs, dict) and "image" in obs:
        return obs["image"]
    return obs

def _pix_batch_to_tensor(pix_np, device):
    return torch.from_numpy(pix_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)

def _pix_to_tensor(pix, device):
    return torch.from_numpy(pix.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


# ── Codebook builder ───────────────────────────────────────────────────────────

def _build_codebook(encoder, replay, device, n_codes=N_CODES, encode_batch=512, tag=""):
    from sklearn.cluster import MiniBatchKMeans
    encoder.eval()
    all_z = []
    for start in range(0, len(replay), encode_batch):
        raw = replay._obs_t[start:min(start + encode_batch, len(replay))]
        pix = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            all_z.append(encoder(pix).cpu().numpy())
    all_z_np = np.concatenate(all_z, axis=0)
    label = f"[RUN34{tag}]"
    logger.info(f"{label} k-means: {len(all_z_np)} replay z → K={n_codes} codes")
    km = MiniBatchKMeans(n_clusters=n_codes, random_state=42, n_init=10, batch_size=2048)
    km.fit(all_z_np)
    centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    logger.info(f"{label} Codebook built | inertia={km.inertia_:.2f}")
    return centers


# ── High-level CEM: plan a subgoal triplet ─────────────────────────────────────

def _highlevel_cem(
    z_cur: torch.Tensor,          # (1, Z_DIM)
    codebook: torch.Tensor,       # (N_CODES, Z_DIM) on device
    z_final: torch.Tensor,        # (1, Z_DIM) — final achievement goal
    n_samples: int = N_HIGH_SAMPLES,
    seq_len: int = SUBGOAL_SEQ_LEN,
) -> List[int]:
    """
    Sample N_SAMPLES random codebook index triplets and score each by path cost.
    Cost = W1 * cos_dist(cb[k1], z_cur)     — k1 reachable from current state
         + W2 * cos_dist(cb[k2], cb[k1])    — k2 reachable from k1
         + W3 * cos_dist(cb[k3], z_final)   — k3 lands near achievement goal

    Returns the best triplet as a list of codebook indices.
    """
    device = codebook.device
    n_codes = codebook.shape[0]

    # Sample random triplets: (n_samples, seq_len)
    indices = torch.randint(0, n_codes, (n_samples, seq_len), device=device)

    # Gather codebook embeddings: (n_samples, seq_len, Z_DIM)
    cb_seqs = codebook[indices]   # (n_samples, seq_len, Z_DIM)

    # Normalise for cosine ops
    def cos_dist(a, b):
        # a, b: (..., Z_DIM) — element-wise cosine distance
        return 1.0 - F.cosine_similarity(a, b, dim=-1)

    # W1: distance from z_cur to first subgoal
    z_cur_exp = z_cur.expand(n_samples, -1)          # (n_samples, Z_DIM)
    cost  = W1 * cos_dist(z_cur_exp, cb_seqs[:, 0])

    # W2: sum of step distances along the path
    for s in range(seq_len - 1):
        cost += (W2 / (seq_len - 1)) * cos_dist(cb_seqs[:, s], cb_seqs[:, s + 1])

    # W3: distance from last subgoal to final goal
    z_fin_exp = z_final.expand(n_samples, -1)        # (n_samples, Z_DIM)
    cost += W3 * cos_dist(cb_seqs[:, -1], z_fin_exp)

    best = cost.argmin().item()
    return indices[best].tolist()   # list of seq_len ints


def _sample_final_goal(
    goal_buf: GoalPixelBuffer,
    replay: PixelReplayBuffer,
    encoder: nn.Module,
    device: str,
) -> Optional[torch.Tensor]:
    """Sample a final goal z from goal_buf (70%) or replay (30%)."""
    use_goal = len(goal_buf) > 0 and random.random() < 0.7
    raw = goal_buf.sample_raw(1) if use_goal else replay.sample_raw(1)
    if raw is None:
        raw = replay.sample_raw(1)
    if raw is None:
        return None
    with torch.no_grad():
        return encoder(_pix_to_tensor(raw[0], device))


# ── World model training ───────────────────────────────────────────────────────

def _train_step(encoder, predictor, opt, replay, device):
    if len(replay) < TRAIN_WARMUP:
        return None, None
    obs_t, actions, obs_next = replay.sample(BATCH_SIZE, device)
    z_t    = encoder(obs_t)
    z_next = encoder(obs_next).detach()
    a_oh   = F.one_hot(actions, N_ACTIONS).float()
    z_pred = predictor(z_t, a_oh)
    pred_loss = F.mse_loss(z_pred, z_next)
    reg_loss  = sigreg(z_t)
    loss = pred_loss + SIGREG_LAMBDA * reg_loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
    opt.step()
    return pred_loss.item(), reg_loss.item()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_crafter(
    encoder, predictor, goal_buf, replay, device,
    codebook=None, replay_size=0, seed_offset=1000, n_eps=EVAL_N_EPS,
):
    if replay_size < TRAIN_WARMUP:
        return 0.0, {}
    was_enc = encoder.training
    encoder.eval(); predictor.eval()

    mpc = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=5,
                     n_samples=256, n_elites=32, n_iters=5,
                     device=device, distance="cosine")
    ever_unlocked = {k: 0 for k in ACHIEVEMENTS}

    for ep in range(n_eps):
        env = make_crafter_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_steps = False, 0
        high_steps = H_HIGH   # force plan on first step
        low_steps  = H_LOW
        subgoal_seq: List[int] = []
        seq_pos    = SUBGOAL_SEQ_LEN   # force replan
        z_goal     = None

        while not done and ep_steps < EP_MAX_STEPS:
            pix = _extract_pix(obs)
            with torch.no_grad():
                z_cur = encoder(_pix_to_tensor(pix, device))

            # High-level re-plan
            if codebook is not None and (seq_pos >= SUBGOAL_SEQ_LEN or high_steps >= H_HIGH):
                z_final = _sample_final_goal(goal_buf, replay, encoder, device)
                if z_final is not None:
                    cb = codebook.to(device)
                    subgoal_seq = _highlevel_cem(z_cur, cb, z_final)
                    seq_pos = 0; high_steps = 0
                    z_goal = codebook[subgoal_seq[seq_pos]].unsqueeze(0).to(device)
                    low_steps = 0

            # Low-level subgoal advance
            if codebook is not None and low_steps >= H_LOW and seq_pos < SUBGOAL_SEQ_LEN:
                seq_pos += 1
                if seq_pos < SUBGOAL_SEQ_LEN:
                    z_goal = codebook[subgoal_seq[seq_pos]].unsqueeze(0).to(device)
                low_steps = 0

            action = (mpc.plan_single(z_cur, z_goal)
                      if z_goal is not None else env.action_space.sample())
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc; ep_steps += 1
            high_steps += 1; low_steps += 1

            for k, v in info.get("achievements", {}).items():
                if v and k in ever_unlocked:
                    ever_unlocked[k] = 1
            if reward > 0:
                goal_buf.add(_extract_pix(obs))
        env.close()

    score    = sum(ever_unlocked.values()) / len(ACHIEVEMENTS)
    per_tier = {
        tier: sum(ever_unlocked.get(a, 0) for a in ach_list) / len(ach_list)
        for tier, ach_list in ACHIEVEMENT_TIERS.items()
    }
    if was_enc: encoder.train(); predictor.train()
    return score, per_tier


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_crafter_run34_loop(
    condition: str = "lewm_crafter_twolevel",
    device: str = "cuda",
    max_steps: int = 600_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "crafter",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "lewm_crafter_twolevel":
        raise ValueError(f"loop_mpc_crafter_run34 supports: lewm_crafter_twolevel — got: {condition}")
    if env_type != "crafter":
        raise ValueError("loop_mpc_crafter_run34 only supports env_type='crafter'.")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else OBSERVE_DEFAULT

    logger.info(
        f"[RUN34] Two-level CEM Crafter | K={N_CODES} codes | "
        f"H_low={H_LOW} H_high={H_HIGH} S={SUBGOAL_SEQ_LEN} N_high={N_HIGH_SAMPLES} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs}"
    )
    t0 = time.time()

    if _wandb is not None:
        _wandb.init(
            project="lewm-crafter",
            name=f"run34-{condition}",
            config={
                "z_dim": Z_DIM, "n_codes": N_CODES,
                "h_low": H_LOW, "h_high": H_HIGH, "seq_len": SUBGOAL_SEQ_LEN,
                "n_high_samples": N_HIGH_SAMPLES, "w1": W1, "w2": W2, "w3": W3,
                "sigreg_lambda": SIGREG_LAMBDA, "cem_distance": "cosine",
                "observe_steps": _observe_steps, "max_steps": max_steps,
                "codebook_refresh_interval": CODEBOOK_REFRESH_INTERVAL,
            },
        )
        logger.info("[RUN34] wandb initialized")

    replay   = PixelReplayBuffer(REPLAY_CAP)
    goal_buf = GoalPixelBuffer(GOAL_BUF_CAP)

    encoder   = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor = Predictor(latent_dim=Z_DIM, n_actions=N_ACTIONS, hidden=512).to(device)
    opt = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=PRED_LR
    )

    codebook:  Optional[torch.Tensor] = None   # (N_CODES, Z_DIM) CPU
    mpc:       Optional[CEMPlanner]   = None

    # Per-env two-level state
    subgoal_seqs: List[List[int]]              = [[] for _ in range(n_envs)]
    seq_pos:      np.ndarray                   = np.full(n_envs, SUBGOAL_SEQ_LEN, dtype=np.int32)
    active_goal_z: List[Optional[torch.Tensor]] = [None] * n_envs
    high_steps:   np.ndarray                   = np.full(n_envs, H_HIGH, dtype=np.int32)
    low_steps:    np.ndarray                   = np.full(n_envs, H_LOW, dtype=np.int32)

    envs = make_crafter_vec_env(n_envs, seed=seed, use_async=False)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    metrics: Dict = {
        "env_step": [], "crafter_score": [], "per_tier": [],
        "pred_loss_ewa": [], "sigreg_loss_ewa": [],
        "mode": [], "wall_time_s": [],
    }
    pred_ewa = sigreg_ewa = None
    env_step = act_steps = total_observe_steps = 0
    frozen = False
    best_score = 0.0
    last_refresh = 0

    while env_step < max_steps:
        in_observe = env_step < _observe_steps

        # ── OBSERVE → ACT transition ──
        if not in_observe and not frozen:
            for p in encoder.parameters():  p.requires_grad_(False)
            for p in predictor.parameters(): p.requires_grad_(False)
            encoder.eval(); predictor.eval()

            codebook = _build_codebook(encoder, replay, device, n_codes=N_CODES)
            mpc = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=5,
                             n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                             device=device, distance="cosine")
            frozen = True; last_refresh = 0
            logger.info(
                f"[RUN34] OBSERVE→ACT at step={env_step} | frozen | "
                f"K={N_CODES} codes | replay={len(replay)} goal_buf={len(goal_buf)}"
            )

        # ── Codebook refresh ──
        if (not in_observe and frozen and codebook is not None and
                act_steps - last_refresh >= CODEBOOK_REFRESH_INTERVAL and act_steps > 0):
            codebook = _build_codebook(encoder, replay, device, n_codes=N_CODES,
                                       tag=f" refresh@{act_steps}")
            last_refresh = act_steps
            logger.info(f"[RUN34] Codebook refreshed at act_step={act_steps}")

        pix_cur_np = _extract_pix(obs)

        # ── Actions ──
        if in_observe:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                z_cur_t = encoder(_pix_batch_to_tensor(pix_cur_np, device))

            cb_dev = codebook.to(device)

            for i in range(n_envs):
                z_i = z_cur_t[i:i+1]

                # High-level replan: new subgoal sequence
                if seq_pos[i] >= SUBGOAL_SEQ_LEN or high_steps[i] >= H_HIGH:
                    z_final = _sample_final_goal(goal_buf, replay, encoder, device)
                    if z_final is not None:
                        subgoal_seqs[i] = _highlevel_cem(z_i, cb_dev, z_final)
                    else:
                        subgoal_seqs[i] = list(np.random.randint(0, N_CODES, SUBGOAL_SEQ_LEN))
                    seq_pos[i]     = 0
                    high_steps[i]  = 0
                    low_steps[i]   = 0
                    active_goal_z[i] = cb_dev[subgoal_seqs[i][0]].unsqueeze(0)

                # Low-level advance: move to next subgoal in sequence
                elif low_steps[i] >= H_LOW and seq_pos[i] < SUBGOAL_SEQ_LEN - 1:
                    seq_pos[i] += 1
                    low_steps[i] = 0
                    active_goal_z[i] = cb_dev[subgoal_seqs[i][seq_pos[i]]].unsqueeze(0)

            valid   = [i for i in range(n_envs) if active_goal_z[i] is not None]
            actions = np.random.randint(N_ACTIONS, size=n_envs)
            if valid and mpc is not None:
                z_batch   = z_cur_t[valid]
                z_g_batch = torch.cat([active_goal_z[i] for i in valid], dim=0)
                actions[valid] = mpc.plan_batch(z_batch, z_g_batch)

        # ── Step ──
        obs_next, rewards, terms, truncs, infos = envs.step(actions)
        dones    = terms | truncs
        env_step += n_envs
        if in_observe:
            total_observe_steps += n_envs
        else:
            act_steps += n_envs

        pix_next_np = _extract_pix(obs_next)

        for i in range(n_envs):
            pix_next_i = pix_cur_np[i] if dones[i] else pix_next_np[i]
            replay.push(pix_cur_np[i], int(actions[i]), pix_next_i)
            if rewards[i] > 0:
                goal_buf.add(pix_next_i)
            if not in_observe:
                high_steps[i] += 1
                low_steps[i]  += 1
                if dones[i]:
                    seq_pos[i]       = SUBGOAL_SEQ_LEN   # force replan
                    active_goal_z[i] = None
                    high_steps[i]    = H_HIGH
                    low_steps[i]     = H_LOW

        obs = obs_next

        # ── Train world model (OBSERVE only) ──
        if in_observe and (env_step // n_envs) % TRAIN_FREQ == 0:
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa   = pl if pred_ewa   is None else 0.95 * pred_ewa   + 0.05 * pl
                sigreg_ewa = rl if sigreg_ewa is None else 0.95 * sigreg_ewa + 0.05 * rl

        # ── Heartbeat ──
        if (env_step // n_envs) % 2000 == 0 and env_step > 0:
            mode_str = "OBSERVE" if in_observe else "ACT"
            logger.info(
                f"[RUN34] step={env_step:7d} | {mode_str:7s} | "
                f"replay={len(replay)} goal_buf={len(goal_buf)} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | {time.time()-t0:.0f}s"
            )

        # ── Eval ──
        if env_step % EVAL_INTERVAL < n_envs:
            mode_str = "OBSERVE" if in_observe else "ACT"
            score, per_tier = _eval_crafter(
                encoder, predictor, goal_buf, replay, device,
                codebook=codebook if not in_observe else None,
                replay_size=len(replay), seed_offset=9000 + env_step,
            )
            elapsed  = time.time() - t0
            tier_str = " ".join(f"{k.replace('tier','t')}={v:.0%}" for k, v in per_tier.items())
            logger.info(
                f"[RUN34] step={env_step:7d} | mode={mode_str:7s} | "
                f"score={score:.1%} | {tier_str} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["crafter_score"].append(score)
            metrics["per_tier"].append(per_tier)
            metrics["pred_loss_ewa"].append(pred_ewa or 0.0)
            metrics["sigreg_loss_ewa"].append(sigreg_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            if score > best_score:
                best_score = score
                logger.info(f"[RUN34] *** new best score={score:.1%} at step {env_step} ***")

            if _wandb is not None:
                wb = {
                    "crafter_score":   score,
                    "pred_loss_ewa":   pred_ewa or 0.0,
                    "sigreg_loss_ewa": sigreg_ewa or 0.0,
                    "replay_size":     len(replay),
                    "goal_buf_size":   len(goal_buf),
                    "mode":            0 if in_observe else 1,
                }
                wb.update({f"tier/{k}": v for k, v in per_tier.items()})
                _wandb.log(wb, step=env_step)

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/crafter_lewm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(), "codebook": codebook},
               ckpt_dir / f"checkpoint_{condition}.pt")
    logger.info(f"[RUN34] Done | best_score={best_score:.1%} | total_time={elapsed_total:.0f}s")
    if _wandb is not None:
        _wandb.finish()

    return {
        "condition":       condition,
        "env_type":        "crafter",
        "env_steps":       metrics["env_step"],
        "crafter_score":   metrics["crafter_score"],
        "per_tier":        metrics["per_tier"],
        "pred_loss_ewa":   metrics["pred_loss_ewa"],
        "sigreg_loss_ewa": metrics["sigreg_loss_ewa"],
        "mgr_loss_ewa":    [0.0] * len(metrics["env_step"]),
        "mode":            metrics["mode"],
        "wall_time_s":     metrics["wall_time_s"],
        "best_score":      best_score,
        "n_switches":      0,
        "switch_log":      [],
        "total_time_s":    elapsed_total,
        "act_steps":       act_steps,
        "observe_steps":   total_observe_steps,
    }


run_abm_loop = run_crafter_run34_loop
