"""
abm/loop_mpc_doorkey_run28.py — Run 28: LeWM on DoorKey pixels

Motivation: Runs 26/27 hit the hard ceiling of 5-dim symbolic state.
  CEM cannot navigate reliably when [x/5, y/5] differences between
  adjacent grid cells are indistinguishable after H predictor steps.
  Run 28 trains a ViT-Tiny encoder online with SIGReg (arXiv 2603.19312)
  and uses L2 CEM planning in the resulting metrically-meaningful latent space.

Architecture (scalable to Crafter and beyond):
  - ViT-Tiny encoder (timm, 5M params, img_size=48, z_dim=256)
  - MLP predictor (Predictor from world_model.py, hidden=512)
  - SIGReg regularization (sigreg() from world_model.py, lambda=0.05)
  - L2 CEM planning (CEMPlanner(distance="l2") from cem_planner.py)
  - OBSERVE+ACT structure (proven Run 24+)
  - 3 manual stages: key -> door -> exit
  - PixelGoalBuffer: pixel obs bucketed by goal-cell position
    (no episode-mixing bug: stage-2 goal cell read from live env)

Condition:   lewm_doorkey_pixels
Loop module: abm.loop_mpc_doorkey_run28
RunPod:
  pip install timm
  python abm_experiment.py --loop-module abm.loop_mpc_doorkey_run28 \\
    --condition lewm_doorkey_pixels --device cuda --env doorkey \\
    --steps 300000 --n-envs 8
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from minigrid.wrappers import RGBImgObsWrapper

from .cem_planner import CEMPlanner
from .world_model import Predictor, sigreg

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE         = 48
Z_DIM            = 256
N_ACTIONS        = 7
N_ENVS           = 8
REPLAY_CAP       = 50_000
GOAL_BUF_CAP     = 256      # per bucket
TRAIN_FREQ       = 16       # train every N vectorised steps
BATCH_SIZE       = 256
SIGREG_LAMBDA    = 0.05
PRED_LR          = 3e-4
OBSERVE_DEFAULT  = 150_000
EP_MAX_STEPS     = 300
EVAL_INTERVAL    = 5_000
EVAL_N_EPS       = 20
N_SEED_EPS       = 200
GOAL_REFRESH     = 64       # re-sample goal z every N steps per env
CEM_H            = 5
CEM_K            = 512
CEM_ITERS        = 10
CEM_ELITE        = 50
TRAIN_WARMUP     = BATCH_SIZE


# ── ViT-Tiny encoder ───────────────────────────────────────────────────────────

class ViTTinyEncoder(nn.Module):
    """
    ViT-Tiny (timm) encoder for pixel observations.
    img_size=48, patch_size=16 → 3×3=9 spatial patches, 192-dim hidden → 256-dim z.
    Same architecture scales to larger img_size (Crafter: 64×64) for Run 29+.
    Requires: pip install timm
    """
    def __init__(self, img_size: int = IMG_SIZE, z_dim: int = Z_DIM):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,   # strip classification head → (B, 192)
        )
        self.proj = nn.Linear(192, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 [0,1] → (B, z_dim)"""
        return self.proj(self.vit(x))


# ── Pixel goal buffer ──────────────────────────────────────────────────────────

class PixelGoalBuffer:
    """
    Stores raw pixel obs (H×W×3 uint8) bucketed by (col, row) goal-cell position.
    Encoding happens on-the-fly at plan time using the current (frozen) encoder,
    so goal embeddings are always consistent with the predictor latent space.
    Bucketing prevents the run-26 episode-mixing bug: stage-2 goal cell is read
    from the live env, ensuring we look up the correct exit-cell bucket.
    """
    def __init__(self, capacity: int = GOAL_BUF_CAP):
        self.capacity = capacity
        self.buckets: Dict[Tuple[int, int], List[np.ndarray]] = {}

    def add(self, pix_hwc: np.ndarray, cell: Tuple[int, int]) -> None:
        buf = self.buckets.setdefault(cell, [])
        if len(buf) >= self.capacity:
            buf.pop(0)
        buf.append(pix_hwc.copy())

    def sample_z(
        self,
        cell: Tuple[int, int],
        encoder: nn.Module,
        device: str,
    ) -> Optional[torch.Tensor]:
        """Encode a random stored pixel obs for the given cell. Returns (1, Z_DIM) or None."""
        bucket = self.buckets.get(cell, [])
        if not bucket:
            return None
        pix = random.choice(bucket)
        obs_t = _pix_to_tensor(pix).unsqueeze(0).to(device)
        with torch.no_grad():
            return encoder(obs_t)   # (1, Z_DIM)

    def __len__(self) -> int:
        return sum(len(v) for v in self.buckets.values())


# ── Pixel replay buffer ────────────────────────────────────────────────────────

class PixelReplayBuffer:
    """
    Circular replay buffer storing (H×W×3 uint8, action, H×W×3 uint8) transitions.
    Pre-allocates numpy arrays for cache-friendly random sampling.
    50k × 48 × 48 × 3 ≈ 346 MB — manageable on A100.
    """
    def __init__(self, capacity: int = REPLAY_CAP, img_size: int = IMG_SIZE):
        self.capacity   = capacity
        self._obs_t     = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._actions   = np.zeros(capacity, dtype=np.int64)
        self._obs_next  = np.zeros((capacity, img_size, img_size, 3), dtype=np.uint8)
        self._ptr       = 0
        self._size      = 0

    def push(self, obs_t: np.ndarray, action: int, obs_next: np.ndarray) -> None:
        self._obs_t[self._ptr]    = obs_t
        self._actions[self._ptr]  = action
        self._obs_next[self._ptr] = obs_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        def to_t(arr: np.ndarray) -> torch.Tensor:
            # (B, H, W, 3) uint8 → (B, 3, H, W) float32 [0,1]
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return (to_t(self._obs_t[idx]),
                torch.from_numpy(self._actions[idx]).long().to(device),
                to_t(self._obs_next[idx]))

    def __len__(self) -> int:
        return self._size


# ── Pixel helpers ──────────────────────────────────────────────────────────────

def _resize_pix(pix: np.ndarray) -> np.ndarray:
    """Resize (H, W, 3) uint8 to (IMG_SIZE, IMG_SIZE, 3) uint8 using PIL."""
    if pix.shape[0] == IMG_SIZE and pix.shape[1] == IMG_SIZE:
        return pix.astype(np.uint8)
    img = Image.fromarray(pix.astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img)


def _pix_to_tensor(pix: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 [0,1]"""
    return torch.from_numpy(pix.astype(np.float32) / 255.0).permute(2, 0, 1)


def _extract_pix_obs(obs) -> np.ndarray:
    """Extract pixel array from obs — handles RGBImgObsWrapper dict or direct array."""
    if isinstance(obs, dict) and 'image' in obs:
        return obs['image']
    return obs


def _get_pix_batch(raw_obs, n_envs: int) -> np.ndarray:
    """(n_envs, H, W, 3) or dict → (n_envs, IMG_SIZE, IMG_SIZE, 3) uint8."""
    pix = _extract_pix_obs(raw_obs)
    if pix.shape[1] != IMG_SIZE or pix.shape[2] != IMG_SIZE:
        pix = np.stack([_resize_pix(pix[i]) for i in range(n_envs)])
    return pix.astype(np.uint8)


def _pix_batch_to_tensor(pix_np: np.ndarray, device: str) -> torch.Tensor:
    """(n_envs, H, W, 3) uint8 → (n_envs, 3, H, W) float32 [0,1] on device."""
    return torch.from_numpy(pix_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


# ── Env helpers ────────────────────────────────────────────────────────────────

def _make_doorkey_env(seed: int = 0):
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    return RGBImgObsWrapper(env)


def _make_doorkey_vec_env(n_envs: int, seed: int = 0):
    fns = [lambda i=i: _make_doorkey_env(seed + i) for i in range(n_envs)]
    return gymnasium.vector.SyncVectorEnv(fns)


def _is_door_open(uw) -> bool:
    for col in range(uw.grid.width):
        for row in range(uw.grid.height):
            cell = uw.grid.get(col, row)
            if cell is not None and cell.type == "door" and cell.is_open:
                return True
    return False


def _find_cell(uw, cell_type) -> Optional[Tuple[int, int]]:
    for row in range(uw.height):
        for col in range(uw.width):
            cell = uw.grid.get(col, row)
            if cell is not None and cell.type == cell_type:
                return (col, row)
    return None


# ── BFS scripted policy (seeder only) ─────────────────────────────────────────

_DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
_VEC_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


def _bfs(uw, start, goal, allow_open_door: bool = False):
    if start == goal:
        return [start]
    def passable(c, r):
        if not (0 <= c < uw.width and 0 <= r < uw.height): return False
        cell = uw.grid.get(c, r)
        if cell is None: return True
        if cell.type == "wall": return False
        if cell.type == "door": return allow_open_door and cell.is_open
        return True
    queue, visited = deque([[start]]), {start}
    while queue:
        path = queue.popleft(); c, r = path[-1]
        for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (c+dc, r+dr)
            if nxt == goal: return path + [nxt]
            if nxt not in visited and passable(c+dc, r+dr):
                visited.add(nxt); queue.append(path + [nxt])
    return []


def _turn_toward(cur, tgt):
    return 1 if (tgt - cur) % 4 in (1, 2) else 0


def _step_toward(pos, d, nxt):
    tgt = _VEC_TO_DIR[(nxt[0]-pos[0], nxt[1]-pos[1])]
    return 2 if d == tgt else _turn_toward(d, tgt)


def _approach_and_interact(uw, pos, d, target, interact):
    fwd = (pos[0]+_DIR_TO_VEC[d][0], pos[1]+_DIR_TO_VEC[d][1])
    if fwd == target: return interact
    for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        adj = (target[0]+dc, target[1]+dr)
        if pos == adj:
            tgt_d = _VEC_TO_DIR[(target[0]-pos[0], target[1]-pos[1])]
            return interact if d == tgt_d else _turn_toward(d, tgt_d)
    best = None
    for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
        adj = (target[0]+dc, target[1]+dr)
        if 0 <= adj[0] < uw.width and 0 <= adj[1] < uw.height:
            cell = uw.grid.get(adj[0], adj[1])
            if cell is None or cell.type == "key":
                path = _bfs(uw, pos, adj)
                if path and (best is None or len(path) < len(best)): best = path
    return _step_toward(pos, d, best[1]) if best and len(best) >= 2 else -1


def _scripted_action(uw, fallback_fn):
    pos, d = tuple(map(int, uw.agent_pos)), int(uw.agent_dir)
    if uw.carrying is None:
        kp = _find_cell(uw, "key")
        if kp is None: return fallback_fn()
        a = _approach_and_interact(uw, pos, d, kp, 3)
        return a if a >= 0 else fallback_fn()
    dp = _find_cell(uw, "door")
    if dp is not None:
        dc = uw.grid.get(dp[0], dp[1])
        if not dc.is_open:
            a = _approach_and_interact(uw, pos, d, dp, 5)
            return a if a >= 0 else fallback_fn()
    gp = _find_cell(uw, "goal")
    if gp is None: return fallback_fn()
    fwd = (pos[0]+_DIR_TO_VEC[d][0], pos[1]+_DIR_TO_VEC[d][1])
    if fwd == gp: return 2
    path = _bfs(uw, pos, gp, allow_open_door=True)
    return _step_toward(pos, d, path[1]) if path and len(path) >= 2 else fallback_fn()


# ── Scripted pixel seeder ──────────────────────────────────────────────────────

def _seed_scripted_pixels(
    key_buf: PixelGoalBuffer,
    door_buf: PixelGoalBuffer,
    goal_buf: PixelGoalBuffer,
    replay_buf: PixelReplayBuffer,
    n_eps: int = N_SEED_EPS,
    seed: int = 7777,
) -> None:
    """
    Run scripted episodes to warm-start:
      - replay_buf with pixel transitions (for encoder training)
      - key_buf / door_buf / goal_buf with bucketed goal pixel observations
    Single env (no vectorisation) so we have full control over obs at each transition.
    """
    logger.info(f"[RUN28] Scripted pixel seeding: {n_eps} episodes…")
    successes = 0

    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        uw = env.unwrapped
        done, steps = False, 0
        had_key = door_was_open = False

        pix_prev = _resize_pix(_extract_pix_obs(obs))

        while not done and steps < EP_MAX_STEPS:
            action = _scripted_action(uw, fallback_fn=env.action_space.sample)

            # Save pre-step cell positions (needed after transition is detected)
            key_cell  = _find_cell(uw, "key")
            door_cell = _find_cell(uw, "door")
            goal_cell = _find_cell(uw, "goal")

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; steps += 1
            uw = env.unwrapped

            pix_cur = _resize_pix(_extract_pix_obs(obs))
            replay_buf.push(pix_prev, action, pix_cur)

            if not had_key and uw.carrying is not None and key_cell:
                key_buf.add(pix_cur, key_cell)   # post-pickup obs in key's bucket
                had_key = True

            if not door_was_open and _is_door_open(uw) and door_cell:
                door_buf.add(pix_cur, door_cell)
                door_was_open = True

            if r > 0 and goal_cell:
                goal_buf.add(pix_prev, goal_cell)  # state just before stepping onto exit
                successes += 1

            pix_prev = pix_cur

        env.close()

    logger.info(
        f"[RUN28] Seed done: {successes}/{n_eps} | replay={len(replay_buf)} | "
        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
    )


# ── Training ───────────────────────────────────────────────────────────────────

def _train_step(
    encoder: nn.Module,
    predictor: nn.Module,
    opt: torch.optim.Optimizer,
    replay_buf: PixelReplayBuffer,
    device: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    One gradient step: MSE(predictor(z_t, a), sg(z_next)) + lambda*SIGReg(z_t)
    Returns (pred_loss, sigreg_loss) or (None, None) if buffer too small.
    """
    if len(replay_buf) < TRAIN_WARMUP:
        return None, None

    obs_t, actions, obs_next = replay_buf.sample(BATCH_SIZE, device)

    z_t    = encoder(obs_t)                          # (B, Z_DIM)  — trains encoder
    z_next = encoder(obs_next).detach()              # (B, Z_DIM)  — stop-gradient target

    a_oh   = F.one_hot(actions, N_ACTIONS).float()
    z_pred = predictor(z_t, a_oh)                   # (B, Z_DIM)

    pred_loss = F.mse_loss(z_pred, z_next)
    reg_loss  = sigreg(z_t)
    loss      = pred_loss + SIGREG_LAMBDA * reg_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
    )
    opt.step()

    return pred_loss.item(), reg_loss.item()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_run28(
    encoder: nn.Module,
    predictor: nn.Module,
    key_buf: PixelGoalBuffer,
    door_buf: PixelGoalBuffer,
    goal_buf: PixelGoalBuffer,
    device: str,
    replay_size: int = 0,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> float:
    """Eval with frozen encoder+predictor. L2 CEM planning per stage."""
    if replay_size < TRAIN_WARMUP:
        return 0.0

    was_training = encoder.training
    encoder.eval(); predictor.eval()

    mpc = CEMPlanner(
        predictor, n_actions=N_ACTIONS, horizon=CEM_H,
        n_samples=CEM_K, n_elites=CEM_ELITE, n_iters=CEM_ITERS,
        device=device, distance="l2",
    )

    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        uw = env.unwrapped
        done, ep_steps, ep_ret = False, 0, 0.0
        current_stage = -1
        goal_age = GOAL_REFRESH   # force refresh on first step
        z_goal: Optional[torch.Tensor] = None

        while not done and ep_steps < EP_MAX_STEPS:
            has_key   = uw.carrying is not None
            door_open = _is_door_open(uw)
            new_stage = 0 if not has_key else (1 if not door_open else 2)

            if new_stage != current_stage or goal_age >= GOAL_REFRESH:
                current_stage = new_stage
                goal_age = 0

                uw = env.unwrapped
                if current_stage == 0:
                    cell = _find_cell(uw, "key")
                    z_goal = key_buf.sample_z(cell, encoder, device) if cell else None
                elif current_stage == 1:
                    cell = _find_cell(uw, "door")
                    z_goal = door_buf.sample_z(cell, encoder, device) if cell else None
                else:
                    cell = _find_cell(uw, "goal")
                    z_goal = goal_buf.sample_z(cell, encoder, device) if cell else None

            pix_hwc = _resize_pix(_extract_pix_obs(obs))
            z_cur   = encoder(_pix_to_tensor(pix_hwc).unsqueeze(0).to(device))

            if z_goal is not None:
                action = mpc.plan_single(z_cur, z_goal)
            else:
                action = env.action_space.sample()

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; ep_ret += r; ep_steps += 1; goal_age += 1
            uw = env.unwrapped

        if ep_ret > 0.5:
            successes += 1
        env.close()

    if was_training:
        encoder.train(); predictor.train()

    return successes / n_eps




# ── Main loop ──────────────────────────────────────────────────────────────────

def run_doorkey_run28_loop(
    condition: str = "lewm_doorkey_pixels",
    device: str = "cuda",
    max_steps: int = 300_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "lewm_doorkey_pixels":
        raise ValueError(f"loop_mpc_doorkey_run28 supports: lewm_doorkey_pixels — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_run28 only supports env_type='doorkey'.")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else OBSERVE_DEFAULT

    logger.info(
        f"[RUN28] LeWM pixel | ViT-Tiny encoder | L2 CEM H={CEM_H} K={CEM_K} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    # ── Buffers ──
    key_buf  = PixelGoalBuffer(GOAL_BUF_CAP)
    door_buf = PixelGoalBuffer(GOAL_BUF_CAP)
    goal_buf = PixelGoalBuffer(GOAL_BUF_CAP)
    replay   = PixelReplayBuffer(REPLAY_CAP)

    # ── Scripted seeder ──
    _seed_scripted_pixels(key_buf, door_buf, goal_buf, replay,
                          n_eps=N_SEED_EPS, seed=seed + 999)

    # ── Model ──
    encoder   = ViTTinyEncoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(device)
    predictor = Predictor(latent_dim=Z_DIM, n_actions=N_ACTIONS, hidden=512).to(device)
    opt = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=PRED_LR
    )

    mpc: Optional[CEMPlanner] = None

    # ── Vectorised envs ──
    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    # Per-env tracking
    had_key          = np.zeros(n_envs, dtype=bool)
    door_open_flags  = np.zeros(n_envs, dtype=bool)
    active_stage     = np.zeros(n_envs, dtype=np.int32)
    goal_ages        = np.full(n_envs, GOAL_REFRESH, dtype=np.int32)  # force refresh
    active_goal_z: List[Optional[torch.Tensor]] = [None] * n_envs
    ep_ret           = np.zeros(n_envs, dtype=np.float32)

    # Per-env cell tracking (updated before each step, valid for done-env handling)
    goal_cells: List[Optional[Tuple]] = [None] * n_envs
    key_cells:  List[Optional[Tuple]] = [None] * n_envs
    door_cells: List[Optional[Tuple]] = [None] * n_envs

    metrics: Dict = {
        "env_step": [], "success_rate": [], "pred_loss_ewa": [],
        "sigreg_loss_ewa": [], "mode": [], "wall_time_s": [],
    }
    pred_ewa = sigreg_ewa = None
    env_step = act_steps = total_observe_steps = 0
    steps_to_80 = None
    frozen = False

    while env_step < max_steps:
        in_observe = env_step < _observe_steps

        # ── OBSERVE → ACT transition ──
        if not in_observe and not frozen:
            for p in encoder.parameters():  p.requires_grad_(False)
            for p in predictor.parameters(): p.requires_grad_(False)
            encoder.eval(); predictor.eval()
            mpc = CEMPlanner(
                predictor, n_actions=N_ACTIONS, horizon=CEM_H,
                n_samples=CEM_K, n_elites=CEM_ELITE, n_iters=CEM_ITERS,
                device=device, distance="l2",
            )
            frozen = True
            logger.info(
                f"[RUN28] OBSERVE→ACT at step={env_step} | encoder+predictor frozen | "
                f"CEM H={CEM_H} K={CEM_K} iters={CEM_ITERS} distance=l2 | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
            )

        # ── Get current pixel obs ──
        pix_cur_np = _get_pix_batch(obs, n_envs)  # (n_envs, IMG_SIZE, IMG_SIZE, 3)

        # ── Update cell tracking before step ──
        for i in range(n_envs):
            uw = envs.envs[i].unwrapped
            if goal_cells[i] is None:
                goal_cells[i] = _find_cell(uw, "goal")
            if key_cells[i] is None:
                key_cells[i] = _find_cell(uw, "key")
            if door_cells[i] is None:
                door_cells[i] = _find_cell(uw, "door")

        # ── Choose actions ──
        if in_observe:
            actions = envs.action_space.sample()
        else:
            # Encode current obs for all envs
            with torch.no_grad():
                pix_t = _pix_batch_to_tensor(pix_cur_np, device)
                z_cur_t = encoder(pix_t)  # (n_envs, Z_DIM)

            actions = np.random.randint(N_ACTIONS, size=n_envs)

            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key   = uw.carrying is not None
                door_open = _is_door_open(uw)
                new_stage = 0 if not has_key else (1 if not door_open else 2)

                if new_stage != active_stage[i]:
                    active_stage[i] = new_stage
                    active_goal_z[i] = None
                    goal_ages[i] = GOAL_REFRESH  # force refresh

                if active_goal_z[i] is None or goal_ages[i] >= GOAL_REFRESH:
                    stage = active_stage[i]
                    if stage == 0:
                        cell = _find_cell(uw, "key")
                        z_g  = key_buf.sample_z(cell, encoder, device) if cell else None
                    elif stage == 1:
                        cell = _find_cell(uw, "door")
                        z_g  = door_buf.sample_z(cell, encoder, device) if cell else None
                    else:
                        # Read exit cell from live env — prevents episode-mixing bug
                        cell = _find_cell(uw, "goal")
                        z_g  = goal_buf.sample_z(cell, encoder, device) if cell else None
                    active_goal_z[i] = z_g
                    goal_ages[i] = 0

            # Batch-plan for envs with a goal
            has_goal = [i for i in range(n_envs) if active_goal_z[i] is not None]
            if has_goal and mpc is not None:
                z_batch  = z_cur_t[has_goal]
                z_g_batch = torch.cat([active_goal_z[i] for i in has_goal], dim=0)
                actions[has_goal] = mpc.plan_batch(z_batch, z_g_batch)

        # ── Step ──
        obs_next, rewards, terms, truncs, infos = envs.step(actions)
        dones = terms | truncs
        if not in_observe:
            ep_ret += rewards
        env_step += n_envs
        if in_observe:
            total_observe_steps += n_envs
        else:
            act_steps += n_envs

        pix_next_np = _get_pix_batch(obs_next, n_envs)

        # ── Per-env post-step ──
        for i in range(n_envs):
            # Replay buffer: for done envs, next obs is reset → use pix_cur as terminal target
            pix_next_i = pix_cur_np[i] if dones[i] else pix_next_np[i]
            replay.push(pix_cur_np[i], int(actions[i]), pix_next_i)

            if not dones[i]:
                uw = envs.envs[i].unwrapped
                # Key pickup
                if not had_key[i] and uw.carrying is not None:
                    if key_cells[i]:
                        key_buf.add(pix_next_np[i], key_cells[i])
                    had_key[i] = True
                # Door open
                if not door_open_flags[i] and _is_door_open(uw):
                    if door_cells[i]:
                        door_buf.add(pix_next_np[i], door_cells[i])
                    door_open_flags[i] = True

            # Success
            if rewards[i] > 0 and goal_cells[i]:
                goal_buf.add(pix_cur_np[i], goal_cells[i])

            if dones[i]:
                had_key[i] = door_open_flags[i] = False
                goal_cells[i] = key_cells[i] = door_cells[i] = None
                if not in_observe:
                    ep_ret[i] = 0.0
                    active_goal_z[i] = None
                    goal_ages[i] = GOAL_REFRESH
            elif not in_observe:
                goal_ages[i] += 1

        obs = obs_next

        # ── Train (OBSERVE only) ──
        if in_observe and (env_step // n_envs) % TRAIN_FREQ == 0:
            pl, rl = _train_step(encoder, predictor, opt, replay, device)
            if pl is not None:
                pred_ewa    = pl if pred_ewa    is None else 0.95 * pred_ewa    + 0.05 * pl
                sigreg_ewa  = rl if sigreg_ewa  is None else 0.95 * sigreg_ewa  + 0.05 * rl

        # ── Heartbeat ──
        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            mode_str = "OBSERVE" if in_observe else "ACT"
            s2_count = int(np.sum(active_stage == 2)) if not in_observe else 0
            logger.info(
                f"[RUN28] step={env_step:7d} | {mode_str:7s} | replay={len(replay)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} | "
                f"pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | "
                f"s2_envs={s2_count} | {time.time()-t0:.0f}s"
            )

        # ── Eval ──
        if env_step % EVAL_INTERVAL < n_envs:
            mode_str = "OBSERVE" if in_observe else "ACT"
            sr = _eval_run28(encoder, predictor, key_buf, door_buf, goal_buf,
                             device, replay_size=len(replay),
                             seed_offset=9000 + env_step, n_eps=EVAL_N_EPS)
            elapsed = time.time() - t0
            logger.info(
                f"[RUN28] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | pred={pred_ewa or 0.0:.4f} sig={sigreg_ewa or 0.0:.4f} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["pred_loss_ewa"].append(pred_ewa or 0.0)
            metrics["sigreg_loss_ewa"].append(sigreg_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[RUN28] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0

    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"condition": condition, "encoder": encoder.state_dict(),
         "predictor": predictor.state_dict()},
        ckpt_dir / f"checkpoint_{condition}.pt",
    )
    logger.info(f"[RUN28] Done | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "pred_loss_ewa": metrics["pred_loss_ewa"],
        "sigreg_loss_ewa": metrics["sigreg_loss_ewa"],
        "mode": metrics["mode"], "wall_time_s": metrics["wall_time_s"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps,
        "observe_steps": total_observe_steps,
    }


run_abm_loop = run_doorkey_run28_loop
