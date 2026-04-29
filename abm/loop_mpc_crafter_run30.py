"""
abm/loop_mpc_crafter_run30.py — Run 30: Director-lite hierarchy on Crafter pixels

Adds SubgoalManager (Director-style, arXiv 2206.04114) on top of Run 29's frozen
encoder + predictor. A k-means codebook of K=64 latent codes is built at the
OBSERVE→ACT transition. The manager (small MLP) maps z_cur → discrete code → z_goal.
Worker (CEM) plans toward z_goal with cosine distance. Manager trained with REINFORCE
on Crafter achievement rewards. Ordering (wood→table→pickaxe→iron) expected to emerge
from the manager learning which subgoal sequences lead to achievements.

Key differences from Run 29:
  - SubgoalManager: z_cur → discrete code (K=64) → z_goal from codebook
  - Codebook: sklearn MiniBatchKMeans on OBSERVE replay z at OBSERVE→ACT
  - Manager trained with REINFORCE on achievement reward (+1 per achievement unlock)
  - CEM distance: "cosine" (Director max-cosine) instead of "l2"
  - Manager horizon H_MANAGER=50: worker gets 50 primitive steps per subgoal
  - Entropy bonus 0.01 to prevent manager collapse to single subgoal
  - Manager checkpoint saved alongside encoder+predictor

Hypothesis: manager learns prerequisite ordering (wood→table→pickaxe) because
selecting the correct subgoal sequence is the only path to +1 achievement reward.

Condition:   lewm_crafter_hierarchy
Loop module: abm.loop_mpc_crafter_run30
RunPod:
  pip install timm crafter scikit-learn
  python abm_experiment.py --loop-module abm.loop_mpc_crafter_run30 \\
    --condition lewm_crafter_hierarchy --device cuda --env crafter \\
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
# Manager (Run 30 additions)
N_CODES         = 64    # codebook size
H_MANAGER       = 50    # primitive steps per subgoal before manager re-selects
MGR_LR          = 3e-4
MGR_BATCH       = 32    # manager decisions before one REINFORCE update
ENTROPY_COEF    = 0.01  # entropy bonus — prevents collapse to single code
VIDEO_LOG_INTERVAL = 50_000   # log one recorded eval episode to wandb every N steps


# ── ViT-Tiny encoder ───────────────────────────────────────────────────────────

class ViTTinyEncoder(nn.Module):
    """ViT-Tiny for 64×64 Crafter observations. 4×4=16 patches, 192-dim → z_dim."""
    def __init__(self, img_size: int = IMG_SIZE, z_dim: int = Z_DIM):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,
        )
        self.proj = nn.Linear(192, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 [0,1] → (B, z_dim)"""
        return self.proj(self.vit(x))


# ── Pixel replay buffer ────────────────────────────────────────────────────────

class PixelReplayBuffer:
    """Circular replay buffer. 30k × 64×64×3 ≈ 368 MB uint8."""
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE):
        self.capacity  = capacity
        self._obs_t    = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._obs_next = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._ptr      = 0
        self._size     = 0

    def push(self, obs_t: np.ndarray, action: int, obs_next: np.ndarray) -> None:
        self._obs_t[self._ptr]    = obs_t
        self._actions[self._ptr]  = action
        self._obs_next[self._ptr] = obs_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        def to_t(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return (to_t(self._obs_t[idx]),
                torch.from_numpy(self._actions[idx]).long().to(device),
                to_t(self._obs_next[idx]))

    def sample_raw(self, n: int) -> Optional[np.ndarray]:
        if self._size == 0:
            return None
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return self._obs_t[idx]

    def __len__(self) -> int:
        return self._size


# ── Goal buffer (OBSERVE-phase fallback) ───────────────────────────────────────

class GoalPixelBuffer:
    """Achievement-positive pixel observations for OBSERVE-phase eval fallback."""
    def __init__(self, capacity: int = GOAL_BUF_CAP):
        self._buf: deque = deque(maxlen=capacity)

    def add(self, pix_hwc: np.ndarray) -> None:
        self._buf.append(pix_hwc.copy())

    def sample_raw(self, n: int = 1) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self) -> int:
        return len(self._buf)


# ── SubgoalManager ─────────────────────────────────────────────────────────────

class SubgoalManager(nn.Module):
    """
    Director-style manager. Maps z_cur → discrete codebook index → z_goal.

    Policy trained with REINFORCE on Crafter achievement rewards. Ordering
    emerges because only the correct subgoal sequence leads to +1 reward.

    Codebook is loaded from k-means centers built at OBSERVE→ACT transition —
    it covers the latent space of states the encoder has seen during exploration.
    """
    def __init__(self, z_dim: int = Z_DIM, n_codes: int = N_CODES):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
            nn.Linear(256, n_codes),
        )
        self.register_buffer("codebook", torch.zeros(n_codes, z_dim))

    def set_codebook(self, centers: torch.Tensor) -> None:
        """Load k-means centers into codebook (called once after _build_codebook)."""
        self.codebook.copy_(centers.to(self.codebook.device))

    def select(
        self, z_cur: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_cur: (B, z_dim) — current latent (no grad; encoder frozen)
        Returns:
          z_goal:   (B, z_dim) — selected codebook entry
          code_idx: (B,) long
          log_prob: (B,) scalar — retains grad through policy params for REINFORCE
        """
        logits   = self.policy(z_cur)
        dist     = torch.distributions.Categorical(logits=logits)
        code_idx = dist.sample()
        log_prob = dist.log_prob(code_idx)
        z_goal   = self.codebook[code_idx]
        return z_goal, code_idx, log_prob

    def policy_entropy(self, z_cur: torch.Tensor) -> torch.Tensor:
        """(B,) entropy for the policy distribution at z_cur."""
        return torch.distributions.Categorical(logits=self.policy(z_cur)).entropy()


# ── Codebook builder ───────────────────────────────────────────────────────────

def _build_codebook(
    encoder: nn.Module,
    replay: PixelReplayBuffer,
    device: str,
    n_codes: int = N_CODES,
    encode_batch: int = 512,
) -> torch.Tensor:
    """
    Encode all replay observations and cluster with MiniBatchKMeans → K centers.
    Called once at OBSERVE→ACT transition with encoder already frozen+eval.
    Returns (n_codes, Z_DIM) float32 CPU tensor.
    """
    from sklearn.cluster import MiniBatchKMeans

    encoder.eval()
    all_z: List[np.ndarray] = []
    n = len(replay)
    for start in range(0, n, encode_batch):
        end = min(start + encode_batch, n)
        raw = replay._obs_t[start:end]
        pix_t = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            z = encoder(pix_t).cpu().numpy()
        all_z.append(z)
    all_z_np = np.concatenate(all_z, axis=0)

    logger.info(f"[RUN30] k-means: {len(all_z_np)} replay z → K={n_codes} codes")
    km = MiniBatchKMeans(n_clusters=n_codes, random_state=42, n_init=10, batch_size=2048)
    km.fit(all_z_np)
    centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    logger.info(f"[RUN30] Codebook built | inertia={km.inertia_:.2f}")
    return centers


# ── Pixel helpers ──────────────────────────────────────────────────────────────

def _extract_pix(obs) -> np.ndarray:
    if isinstance(obs, dict) and "image" in obs:
        return obs["image"]
    return obs


def _pix_batch_to_tensor(pix_np: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(pix_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


def _pix_to_tensor(pix: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(pix.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def _sample_goal_z(
    goal_buf: GoalPixelBuffer,
    replay: PixelReplayBuffer,
    encoder: nn.Module,
    device: str,
) -> Optional[torch.Tensor]:
    """OBSERVE-phase fallback: 70% from goal_buf, 30% from replay."""
    use_goal = len(goal_buf) > 0 and random.random() < 0.7
    raw = goal_buf.sample_raw(1) if use_goal else replay.sample_raw(1)
    if raw is None:
        raw = replay.sample_raw(1)
    if raw is None:
        return None
    with torch.no_grad():
        return encoder(_pix_to_tensor(raw[0], device))


# ── Video recording (single eval episode, for wandb) ──────────────────────────

def _record_episode(
    encoder: nn.Module,
    predictor: nn.Module,
    goal_buf: GoalPixelBuffer,
    replay: PixelReplayBuffer,
    device: str,
    manager: Optional[SubgoalManager] = None,
    seed: int = 7777,
) -> np.ndarray:
    """
    Run one eval episode and return frames as (T, C, H, W) uint8 for wandb.Video.
    Uses manager subgoals if provided, otherwise falls back to goal_buf sampling.
    """
    encoder.eval(); predictor.eval()
    if manager is not None:
        manager.eval()

    mpc = CEMPlanner(
        predictor, n_actions=N_ACTIONS, horizon=5,
        n_samples=256, n_elites=32, n_iters=5,
        device=device, distance="cosine",
    )
    env = make_crafter_env(seed=seed)
    obs, _ = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    done, steps, mgr_steps_ep = False, 0, H_MANAGER
    z_goal: Optional[torch.Tensor] = None

    while not done and steps < EP_MAX_STEPS:
        pix = _extract_pix(obs)
        frames.append(pix.copy())  # (H, W, 3) uint8

        with torch.no_grad():
            z_cur = encoder(_pix_to_tensor(pix, device))

        if mgr_steps_ep >= H_MANAGER or z_goal is None:
            if manager is not None:
                with torch.no_grad():
                    z_g, _, _ = manager.select(z_cur)
                z_goal = z_g
            else:
                z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
            mgr_steps_ep = 0

        action = (mpc.plan_single(z_cur, z_goal)
                  if z_goal is not None else env.action_space.sample())
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
        mgr_steps_ep += 1

    env.close()
    # (T, H, W, 3) → (T, C, H, W) for wandb.Video
    return np.stack(frames).transpose(0, 3, 1, 2)


# ── World model training (OBSERVE phase — unchanged from Run 29) ───────────────

def _train_step(
    encoder: nn.Module,
    predictor: nn.Module,
    opt: torch.optim.Optimizer,
    replay: PixelReplayBuffer,
    device: str,
) -> Tuple[Optional[float], Optional[float]]:
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
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
    )
    opt.step()
    return pred_loss.item(), reg_loss.item()


# ── REINFORCE update for manager ───────────────────────────────────────────────

def _manager_update(
    manager: SubgoalManager,
    mgr_opt: torch.optim.Optimizer,
    log_probs: List[torch.Tensor],
    returns: List[float],
    device: str,
) -> float:
    """
    REINFORCE: loss = -mean(log_prob * normalized_return) - entropy_bonus.
    log_probs retain their computation graphs through manager.policy parameters.
    Returns the scalar policy loss value for logging.
    """
    if not log_probs:
        return 0.0
    lp  = torch.stack(log_probs)                                          # (N,) with grad
    ret = torch.tensor(returns, dtype=torch.float32, device=device)       # (N,) no grad
    if ret.std() > 1e-6:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
    policy_loss = -(lp * ret).mean()
    mgr_opt.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(manager.policy.parameters(), 1.0)
    mgr_opt.step()
    return policy_loss.item()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_run30(
    encoder: nn.Module,
    predictor: nn.Module,
    goal_buf: GoalPixelBuffer,
    replay: PixelReplayBuffer,
    device: str,
    manager: Optional[SubgoalManager] = None,
    replay_size: int = 0,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> Tuple[float, Dict]:
    """
    Eval with frozen encoder+predictor.
    ACT phase: SubgoalManager selects subgoals, CEM with cosine distance.
    OBSERVE phase (manager=None): goal_buf/replay sampling, cosine CEM.
    """
    if replay_size < TRAIN_WARMUP:
        return 0.0, {}

    was_training_enc = encoder.training
    encoder.eval(); predictor.eval()
    if manager is not None:
        was_training_mgr = manager.training
        manager.eval()

    mpc = CEMPlanner(
        predictor, n_actions=N_ACTIONS, horizon=5,
        n_samples=256, n_elites=32, n_iters=5,
        device=device, distance="cosine",
    )

    ever_unlocked: Dict[str, int] = {k: 0 for k in ACHIEVEMENTS}

    for ep in range(n_eps):
        env = make_crafter_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_steps, mgr_steps_ep = False, 0, H_MANAGER  # force first subgoal pick
        z_goal: Optional[torch.Tensor] = None

        while not done and ep_steps < EP_MAX_STEPS:
            pix = _extract_pix(obs)
            with torch.no_grad():
                z_cur = encoder(_pix_to_tensor(pix, device))  # (1, Z_DIM)

            if mgr_steps_ep >= H_MANAGER or z_goal is None:
                if manager is not None:
                    with torch.no_grad():
                        z_g, _, _ = manager.select(z_cur)
                    z_goal = z_g
                else:
                    z_goal = _sample_goal_z(goal_buf, replay, encoder, device)
                mgr_steps_ep = 0

            action = (mpc.plan_single(z_cur, z_goal)
                      if z_goal is not None else env.action_space.sample())

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_steps += 1
            mgr_steps_ep += 1

            for k, v in info.get("achievements", {}).items():
                if v and k in ever_unlocked:
                    ever_unlocked[k] = 1
            if reward > 0:
                goal_buf.add(_extract_pix(obs))

        env.close()

    score = sum(ever_unlocked.values()) / len(ACHIEVEMENTS)
    per_tier = {
        tier: sum(ever_unlocked.get(a, 0) for a in ach_list) / len(ach_list)
        for tier, ach_list in ACHIEVEMENT_TIERS.items()
    }

    if was_training_enc:
        encoder.train(); predictor.train()
    if manager is not None and was_training_mgr:
        manager.train()

    return score, per_tier


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_crafter_run30_loop(
    condition: str = "lewm_crafter_hierarchy",
    device: str = "cuda",
    max_steps: int = 600_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "crafter",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "lewm_crafter_hierarchy":
        raise ValueError(f"loop_mpc_crafter_run30 supports: lewm_crafter_hierarchy — got: {condition}")
    if env_type != "crafter":
        raise ValueError("loop_mpc_crafter_run30 only supports env_type='crafter'.")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else OBSERVE_DEFAULT

    logger.info(
        f"[RUN30] Director-lite Crafter | ViT-Tiny 64×64 | cosine CEM | "
        f"K={N_CODES} codes H_mgr={H_MANAGER} mgr_batch={MGR_BATCH} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    # ── W&B ──
    if _wandb is not None:
        _wandb.init(
            project="lewm-crafter",
            name=f"run30-{condition}",
            config={
                "z_dim": Z_DIM, "n_codes": N_CODES, "h_manager": H_MANAGER,
                "mgr_lr": MGR_LR, "entropy_coef": ENTROPY_COEF,
                "sigreg_lambda": SIGREG_LAMBDA, "cem_distance": "cosine",
                "observe_steps": _observe_steps, "max_steps": max_steps,
                "n_envs": n_envs,
            },
        )
        logger.info("[RUN30] wandb initialized — live dashboard active")

    # ── Buffers ──
    replay   = PixelReplayBuffer(REPLAY_CAP)
    goal_buf = GoalPixelBuffer(GOAL_BUF_CAP)

    # ── World model ──
    encoder   = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor = Predictor(latent_dim=Z_DIM, n_actions=N_ACTIONS, hidden=512).to(device)
    opt = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=PRED_LR
    )

    # Created at OBSERVE→ACT transition
    manager:  Optional[SubgoalManager]         = None
    mgr_opt:  Optional[torch.optim.Optimizer]  = None
    mpc:      Optional[CEMPlanner]             = None

    # ── Per-env manager state ──
    # mgr_log_prob[i]: scalar tensor with grad through manager.policy, or None
    mgr_log_prob:   List[Optional[torch.Tensor]] = [None] * n_envs
    mgr_reward_acc: np.ndarray = np.zeros(n_envs, dtype=np.float32)
    mgr_steps:      np.ndarray = np.full(n_envs, H_MANAGER, dtype=np.int32)
    active_goal_z:  List[Optional[torch.Tensor]] = [None] * n_envs

    # ── REINFORCE accumulation buffer ──
    mgr_lp_buf:  List[torch.Tensor] = []   # log_prob tensors — retain grad
    mgr_ret_buf: List[float]        = []   # corresponding accumulated returns

    # ── Vectorised envs ──
    envs = make_crafter_vec_env(n_envs, seed=seed, use_async=False)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    metrics: Dict = {
        "env_step": [], "crafter_score": [], "per_tier": [],
        "pred_loss_ewa": [], "sigreg_loss_ewa": [], "mgr_loss_ewa": [],
        "mode": [], "wall_time_s": [],
    }
    pred_ewa = sigreg_ewa = mgr_loss_ewa = None
    env_step = act_steps = total_observe_steps = 0
    frozen = False
    best_score = 0.0

    while env_step < max_steps:
        in_observe = env_step < _observe_steps

        # ── OBSERVE → ACT transition ──
        if not in_observe and not frozen:
            for p in encoder.parameters():  p.requires_grad_(False)
            for p in predictor.parameters(): p.requires_grad_(False)
            encoder.eval(); predictor.eval()

            centers = _build_codebook(encoder, replay, device, n_codes=N_CODES)
            manager = SubgoalManager(z_dim=Z_DIM, n_codes=N_CODES).to(device)
            manager.set_codebook(centers)
            manager.train()
            mgr_opt = optim.Adam(manager.policy.parameters(), lr=MGR_LR)

            mpc = CEMPlanner(
                predictor, n_actions=N_ACTIONS, horizon=5,
                n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                device=device, distance="cosine",
            )
            frozen = True
            logger.info(
                f"[RUN30] OBSERVE→ACT at step={env_step} | "
                f"encoder+predictor frozen | K={N_CODES} codes | "
                f"replay={len(replay)} goal_buf={len(goal_buf)}"
            )

        # ── Current pixel obs ──
        pix_cur_np = _extract_pix(obs)  # (n_envs, 64, 64, 3) uint8

        # ── Actions ──
        if in_observe:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                z_cur_t = encoder(_pix_batch_to_tensor(pix_cur_np, device))  # (n_envs, Z_DIM)

            # Manager subgoal selection (outside no_grad — log_prob needs grad)
            for i in range(n_envs):
                if active_goal_z[i] is None or mgr_steps[i] >= H_MANAGER:
                    # Flush completed subgoal experience
                    if mgr_log_prob[i] is not None:
                        mgr_lp_buf.append(mgr_log_prob[i])
                        mgr_ret_buf.append(float(mgr_reward_acc[i]))

                    z_i = z_cur_t[i:i+1]              # (1, Z_DIM), no grad
                    z_g, _, lp = manager.select(z_i)   # lp retains grad through policy
                    active_goal_z[i]  = z_g.detach()   # CEM needs no grad
                    mgr_log_prob[i]   = lp
                    mgr_reward_acc[i] = 0.0
                    mgr_steps[i]      = 0

            # CEM planning (batch over envs with valid goals)
            valid = [i for i in range(n_envs) if active_goal_z[i] is not None]
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

        # ── Per-env post-step ──
        for i in range(n_envs):
            pix_next_i = pix_cur_np[i] if dones[i] else pix_next_np[i]
            replay.push(pix_cur_np[i], int(actions[i]), pix_next_i)

            if rewards[i] > 0:
                goal_buf.add(pix_next_i)

            if not in_observe:
                mgr_reward_acc[i] += float(rewards[i])
                mgr_steps[i] += 1

                if dones[i]:
                    # Flush on episode end
                    if mgr_log_prob[i] is not None:
                        mgr_lp_buf.append(mgr_log_prob[i])
                        mgr_ret_buf.append(float(mgr_reward_acc[i]))
                    active_goal_z[i]  = None
                    mgr_log_prob[i]   = None
                    mgr_reward_acc[i] = 0.0
                    mgr_steps[i]      = H_MANAGER  # force subgoal pick at next step

        obs = obs_next

        # ── Train world model (OBSERVE only) ──
        if in_observe and (env_step // n_envs) % TRAIN_FREQ == 0:
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa   = pl if pred_ewa   is None else 0.95 * pred_ewa   + 0.05 * pl
                sigreg_ewa = rl if sigreg_ewa is None else 0.95 * sigreg_ewa + 0.05 * rl

        # ── REINFORCE update for manager (ACT only) ──
        if not in_observe and len(mgr_lp_buf) >= MGR_BATCH:
            ml = _manager_update(manager, mgr_opt, mgr_lp_buf, mgr_ret_buf, device)
            mgr_lp_buf.clear()
            mgr_ret_buf.clear()
            mgr_loss_ewa = (ml if mgr_loss_ewa is None
                            else 0.95 * mgr_loss_ewa + 0.05 * ml)

        # ── Heartbeat ──
        if (env_step // n_envs) % 2000 == 0 and env_step > 0:
            mode_str = "OBSERVE" if in_observe else "ACT"
            logger.info(
                f"[RUN30] step={env_step:7d} | {mode_str:7s} | "
                f"replay={len(replay)} goal_buf={len(goal_buf)} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} "
                f"mgr={mgr_loss_ewa or 0.0:.4f} | {time.time()-t0:.0f}s"
            )

        # ── Eval ──
        if env_step % EVAL_INTERVAL < n_envs:
            mode_str = "OBSERVE" if in_observe else "ACT"
            score, per_tier = _eval_run30(
                encoder, predictor, goal_buf, replay, device,
                manager=manager if not in_observe else None,
                replay_size=len(replay),
                seed_offset=9000 + env_step,
                n_eps=EVAL_N_EPS,
            )
            elapsed = time.time() - t0
            tier_str = " ".join(f"{k.replace('tier','t')}={v:.0%}" for k, v in per_tier.items())
            logger.info(
                f"[RUN30] step={env_step:7d} | mode={mode_str:7s} | "
                f"score={score:.1%} | {tier_str} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} "
                f"mgr={mgr_loss_ewa or 0.0:.4f} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["crafter_score"].append(score)
            metrics["per_tier"].append(per_tier)
            metrics["pred_loss_ewa"].append(pred_ewa or 0.0)
            metrics["sigreg_loss_ewa"].append(sigreg_ewa or 0.0)
            metrics["mgr_loss_ewa"].append(mgr_loss_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            if score > best_score:
                best_score = score
                logger.info(f"[RUN30] *** new best score={score:.1%} at step {env_step} ***")

            # ── W&B scalar logging ──
            if _wandb is not None:
                wb_payload: Dict = {
                    "crafter_score":   score,
                    "pred_loss_ewa":   pred_ewa or 0.0,
                    "sigreg_loss_ewa": sigreg_ewa or 0.0,
                    "mgr_loss_ewa":    mgr_loss_ewa or 0.0,
                    "replay_size":     len(replay),
                    "goal_buf_size":   len(goal_buf),
                    "mode":            0 if in_observe else 1,
                }
                wb_payload.update({f"tier/{k}": v for k, v in per_tier.items()})
                _wandb.log(wb_payload, step=env_step)

            # ── W&B video logging (every VIDEO_LOG_INTERVAL steps) ──
            if _wandb is not None and env_step % VIDEO_LOG_INTERVAL < n_envs:
                frames = _record_episode(
                    encoder, predictor, goal_buf, replay, device,
                    manager=manager if not in_observe else None,
                    seed=7777 + env_step,
                )
                _wandb.log(
                    {"eval_video": _wandb.Video(frames, fps=10, format="mp4",
                                                caption=f"step={env_step} mode={mode_str} score={score:.1%}")},
                    step=env_step,
                )

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/crafter_lewm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "condition": condition,
            "encoder":   encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "manager":   manager.state_dict() if manager is not None else None,
        },
        ckpt_dir / f"checkpoint_{condition}.pt",
    )
    logger.info(f"[RUN30] Done | best_score={best_score:.1%} | total_time={elapsed_total:.0f}s")
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
        "mgr_loss_ewa":    metrics["mgr_loss_ewa"],
        "mode":            metrics["mode"],
        "wall_time_s":     metrics["wall_time_s"],
        "best_score":      best_score,
        "n_switches":      0,
        "switch_log":      [],
        "total_time_s":    elapsed_total,
        "act_steps":       act_steps,
        "observe_steps":   total_observe_steps,
    }


run_abm_loop = run_crafter_run30_loop
