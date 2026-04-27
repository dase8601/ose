"""
abm/loop_mpc_doorkey_run15a.py — Run 15a: V-JEPA 2.1 + adapter + delayed EBM

Fix over Run 14a:
  Run 14a activated EBM training at step ~10k when the adapter was randomly
  initialized. EBM gradients trained the energy function on random adapter
  projections. As the adapter evolved over 80k steps, the EBM became
  misaligned with the current adapter geometry — goal energy pointed in
  random directions, CEM planning found nothing consistent.

Fix (delayed EBM):
  Adapter trains freely during all of OBSERVE (80k steps) with no EBM
  interference. EBM training starts only when OBSERVE ends (env_step >= 80k),
  at which point the adapter has stabilized and the 128-dim space has
  meaningful geometry. EBM trains on stable adapter representations from
  the start, learning a correct energy landscape from day one.

Fix (learned adapter):
  A small trainable MLP maps the frozen 768-dim V-JEPA features into a
  128-dim space and is trained jointly with the predictor. In this
  bottleneck, the adapter is forced to preserve only variance that actually
  predicts the next state — it cannot cheat with identity because the
  predictor's cos-sim loss penalizes wrong transitions in 128-dim space.

  All replay buffers store raw 768-dim V-JEPA features.
  The adapter projects them to 128-dim on the fly during training.
  CEM plans in 128-dim adapter space.

  Architecture:
    Frozen V-JEPA 2.1 → 768-dim raw features (no grad, stored in buffers)
    FeatureAdapter: 768 → 256 → 128 (trained, L2-norm output)
    FeaturePredictor: (128 + 7) → 512 → 128 (trained)
    EBMCostHead(128) (trained)
    CEMPlanner in 128-dim adapter space

Condition: vjepa2_adapter_late_ebm
Loop module: abm.loop_mpc_doorkey_run15a
"""

from __future__ import annotations

import logging
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
from minigrid.wrappers import RGBImgObsWrapper

from .cem_planner import CEMPlanner, EBMCostHead

logger = logging.getLogger(__name__)

N_ENVS                 = 16
N_ACTIONS              = 7
RAW_DIM                = 768    # V-JEPA feature dim — stored in all buffers
ADAPTER_DIM            = 128    # adapter output dim — used for planning
VJEPA2_IMG_SIZE        = 384
PRED_HIDDEN            = 512    # smaller since adapter_dim=128
ADAPTER_LR             = 3e-4   # separate lr for adapter (faster than predictor)
PRED_LR                = 1e-4
PRED_BATCH             = 256
PRED_WARMUP            = 500
PRED_TRAIN_STEPS       = 4
REPLAY_CAPACITY        = 100_000
SEED_BUF_CAPACITY      = 20_000
POST_DOOR_NEG_CAPACITY = 5_000
GOAL_BUF_CAPACITY      = 1_024
HER_CAPACITY           = 4_096
EVAL_INTERVAL          = 5_000
EVAL_N_EPS             = 10
GOAL_REFRESH_STEPS     = 64
DEFAULT_OBSERVE        = 80_000
EP_MAX_STEPS           = 300
EBM_MIN_GOALS          = 5
EBM_HER_MIN            = 20
EBM_POST_DOOR_MIN      = 20
EBM_WARMUP_STEPS       = 200        # lower — adapter is stable before EBM starts
EBM_LR                 = 3e-4
EBM_BATCH              = 32
N_SEED_EPS             = 200

CEM_HORIZON            = 8
CEM_SAMPLES            = 512
CEM_ELITES             = 64
CEM_ITERS              = 5


# ── V-JEPA 2.1 encoder ────────────────────────────────────────────────────

_VJEPA2_CKPT_URL  = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
_VJEPA2_CKPT_NAME = "vjepa2_1_vitb_dist_vitG_384.pt"


def _load_vjepa2(device: str) -> nn.Module:
    import urllib.request
    logger.info("[RUN15A] Loading V-JEPA 2.1 ViT-Base-384 (frozen)...")
    encoder, _ = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_1_vit_base_384",
        pretrained=False, trust_repo=True,
    )
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / _VJEPA2_CKPT_NAME
    if not ckpt_path.exists():
        logger.info(f"[RUN15A] Downloading checkpoint...")
        urllib.request.urlretrieve(_VJEPA2_CKPT_URL, str(ckpt_path))
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    enc_sd = state_dict["ema_encoder"]
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_sd.items()}
    encoder.load_state_dict(cleaned)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"[RUN15A] V-JEPA 2.1 loaded — {n_params:.1f}M params (frozen) | adapter_dim={ADAPTER_DIM}")
    return encoder


_VJEPA2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_VJEPA2_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


@torch.no_grad()
def _encode(model: nn.Module, imgs_np: np.ndarray, device: str) -> torch.Tensor:
    """imgs_np: (B, H, W, 3) uint8 → (B, 768) L2-norm V-JEPA raw features"""
    if imgs_np.ndim == 3:
        imgs_np = imgs_np[None]
    x = torch.from_numpy(imgs_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    if x.shape[-2:] != (VJEPA2_IMG_SIZE, VJEPA2_IMG_SIZE):
        x = F.interpolate(x, size=(VJEPA2_IMG_SIZE, VJEPA2_IMG_SIZE), mode="bilinear", align_corners=False)
    x = x.unsqueeze(2)
    x = (x - _VJEPA2_MEAN.to(device)) / _VJEPA2_STD.to(device)
    tokens = model(x)
    return F.normalize(tokens.mean(dim=1), p=2, dim=-1)


def _doorkey_discrimination_test(model: nn.Module, device: str) -> float:
    frames = []
    for s in range(20):
        env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
        env = RGBImgObsWrapper(env)
        obs, _ = env.reset(seed=s)
        frames.append(obs["image"])
        env.close()
    imgs = np.stack(frames)
    z = _encode(model, imgs, device)
    sims = [F.cosine_similarity(z[i:i+1], z[i+1:i+2]).item() for i in range(0, 20, 2)]
    mean_sim = float(np.mean(sims))
    logger.info(f"[RUN15A] V-JEPA raw discrimination: cross-seed cos_sim={mean_sim:.4f} (adapter will fix this)")
    return mean_sim


# ── Adapter module ─────────────────────────────────────────────────────────

class FeatureAdapter(nn.Module):
    """
    Trainable bottleneck: frozen 768-dim V-JEPA → learned 128-dim.
    Trained jointly with the predictor. Forces the 128-dim space to
    capture action-relevant variance — cannot collapse to identity
    because the bottleneck can't replicate the full 768-dim signal.
    """

    def __init__(self, in_dim: int = RAW_DIM, out_dim: int = ADAPTER_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), p=2, dim=-1)


# ── Feature-based replay buffers — store RAW_DIM (768) ────────────────────

class FeatureReplayBuffer:
    """Stores raw 768-dim V-JEPA features. Adapter projects on sampling."""

    def __init__(self, capacity: int, feature_dim: int = RAW_DIM):
        self.capacity    = capacity
        self.feature_dim = feature_dim
        self._z      = np.zeros((capacity, feature_dim), dtype=np.float32)
        self._a      = np.zeros(capacity, dtype=np.int64)
        self._z_next = np.zeros((capacity, feature_dim), dtype=np.float32)
        self._ptr    = 0
        self._size   = 0

    def push(self, z: np.ndarray, a: int, z_next: np.ndarray) -> None:
        self._z[self._ptr]      = z
        self._a[self._ptr]      = a
        self._z_next[self._ptr] = z_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_raw(self, n: int, device: str):
        """Returns raw 768-dim tensors — caller adapts if needed."""
        idx    = np.random.choice(self._size, size=n, replace=self._size < n)
        z      = torch.from_numpy(self._z[idx]).to(device)
        a      = torch.from_numpy(self._a[idx]).long().to(device)
        z_next = torch.from_numpy(self._z_next[idx]).to(device)
        return z, a, z_next

    def __len__(self) -> int:
        return self._size


class GoalFeatureBuffer:
    """Stores raw 768-dim V-JEPA features. Adapter projects on sampling."""

    def __init__(self, capacity: int, feature_dim: int = RAW_DIM):
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, z: np.ndarray) -> None:
        self._buf.append(np.array(z, copy=True))

    def sample_raw(self, n: int) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=n, replace=len(self._buf) < n)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self) -> int:
        return len(self._buf)


# ── Predictor (operates in 128-dim adapter space) ─────────────────────────

class FeaturePredictor(nn.Module):
    """(z_adapted: 128, a_oh: 7) → z_next_adapted: 128"""

    def __init__(self, feature_dim: int = ADAPTER_DIM, n_actions: int = N_ACTIONS, hidden: int = PRED_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, z: torch.Tensor, a_oh: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat([z, a_oh], dim=-1)), p=2, dim=-1)

    def forward_sequence(self, z_seq: torch.Tensor, a_oh_seq: torch.Tensor) -> torch.Tensor:
        z = z_seq[:, 0]
        preds = []
        for t in range(z_seq.shape[1]):
            z = self.forward(z, a_oh_seq[:, t])
            preds.append(z)
        return torch.stack(preds, dim=1)


# ── Environment helpers ───────────────────────────────────────────────────

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


# ── BFS scripted policy ───────────────────────────────────────────────────

_DIR_TO_VEC = [(1,0),(0,1),(-1,0),(0,-1)]
_VEC_TO_DIR = {(1,0):0,(0,1):1,(-1,0):2,(0,-1):3}


def _find_cell(uw, cell_type):
    for row in range(uw.height):
        for col in range(uw.width):
            cell = uw.grid.get(col, row)
            if cell is not None and cell.type == cell_type:
                return (col, row)
    return None


def _bfs(uw, start, goal, allow_open_door=False):
    if start == goal: return [start]
    def passable(c, r):
        if not (0 <= c < uw.width and 0 <= r < uw.height): return False
        cell = uw.grid.get(c, r)
        if cell is None: return True
        if cell.type == "wall": return False
        if cell.type == "door": return allow_open_door and cell.is_open
        return True
    queue, visited = deque([[start]]), {start}
    while queue:
        path = queue.popleft()
        c, r = path[-1]
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
                if path and (best is None or len(path) < len(best)):
                    best = path
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


# ── Scripted seeder — stores RAW 768-dim features ─────────────────────────

def _seed_scripted(
    vjepa2, key_buf, door_buf, goal_buf, seed_buf, post_door_neg_buf,
    device, n_eps=N_SEED_EPS, seed=7777,
):
    logger.info(f"[RUN15A] Scripted seeding: {n_eps} episodes (storing raw 768-dim)…")
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        uw = env.unwrapped
        done, steps = False, 0
        had_key = door_was_open = collecting_post_door = False

        while not done and steps < EP_MAX_STEPS:
            action = _scripted_action(uw, fallback_fn=env.action_space.sample)
            z_prev = _encode(vjepa2, obs["image"], device).cpu().numpy()[0]   # (768,)

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; steps += 1
            uw = env.unwrapped

            z_cur = _encode(vjepa2, obs["image"], device).cpu().numpy()[0]    # (768,)
            seed_buf.push(z_prev, action, z_cur)

            if not had_key and uw.carrying is not None:
                key_buf.push(z_cur); had_key = True
            if not door_was_open and _is_door_open(uw):
                door_buf.push(z_cur); door_was_open = True; collecting_post_door = True
            if r > 0:
                goal_buf.push(z_cur); collecting_post_door = False; successes += 1
            elif collecting_post_door:
                post_door_neg_buf.push(z_cur)
        env.close()

    logger.info(
        f"[RUN15A] Seed done: {successes}/{n_eps} | seed_buf={len(seed_buf)} | "
        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} post_neg={len(post_door_neg_buf)}"
    )


# ── Predictor + adapter training ──────────────────────────────────────────

def _train_predictor(adapter, predictor, opt, buf, seed_buf, device, n_steps=PRED_TRAIN_STEPS):
    if len(buf) < PRED_WARMUP:
        return None
    n_seed, n_online = PRED_BATCH // 2, PRED_BATCH - PRED_BATCH // 2
    use_mix = len(seed_buf) >= n_seed
    last_loss = None
    for _ in range(n_steps):
        if use_mix:
            z_s, a_s, zn_s = seed_buf.sample_raw(n_seed, device)
            z_o, a_o, zn_o = buf.sample_raw(n_online, device)
            z_raw, a_t, zn_raw = torch.cat([z_s,z_o]), torch.cat([a_s,a_o]), torch.cat([zn_s,zn_o])
        else:
            z_raw, a_t, zn_raw = buf.sample_raw(PRED_BATCH, device)

        # Project raw features through adapter (gradients flow into adapter)
        z_adapted  = adapter(z_raw)
        zn_adapted = adapter(zn_raw)

        a_oh   = F.one_hot(a_t, N_ACTIONS).float()
        z_pred = predictor(z_adapted, a_oh)
        loss   = (1 - F.cosine_similarity(z_pred, zn_adapted.detach(), dim=-1)).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = loss.item()
    return last_loss


# ── Curiosity ─────────────────────────────────────────────────────────────

def _curiosity_actions(adapter, predictor, z_raw, running_z_mean, device):
    n = z_raw.shape[0]
    with torch.no_grad():
        z_adapted = adapter(z_raw)   # (n, 128)
        z_rep  = z_adapted.unsqueeze(1).expand(n, N_ACTIONS, ADAPTER_DIM).reshape(n*N_ACTIONS, ADAPTER_DIM)
        a_idx  = torch.arange(N_ACTIONS, device=device).unsqueeze(0).expand(n,-1).reshape(-1)
        a_oh   = F.one_hot(a_idx, N_ACTIONS).float()
        z_next = predictor(z_rep, a_oh).reshape(n, N_ACTIONS, ADAPTER_DIM)
        novelty = (z_next - running_z_mean.view(1,1,ADAPTER_DIM)).pow(2).sum(-1)
        return novelty.argmax(dim=-1).cpu().numpy()


# ── EBM training ──────────────────────────────────────────────────────────

def _adapt_raw(buf_or_arr, n, device, adapter):
    """Sample raw features and project through adapter."""
    if hasattr(buf_or_arr, 'sample_raw'):
        raw = buf_or_arr.sample_raw(n)
    else:
        raw = buf_or_arr
    if raw is None:
        return None
    t = torch.from_numpy(raw).to(device)
    return adapter(t)


def _train_ebm(adapter, ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, post_door_neg_buf, buf, device):
    if len(buf) < EBM_BATCH * 2:
        return False
    opt_ebm.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    ready = [(b, n) for b, n in [(key_buf,"key"),(door_buf,"door"),(goal_buf,"goal")] if len(b) >= EBM_MIN_GOALS]
    if ready:
        chosen, _ = ready[np.random.randint(len(ready))]
        z_pos = _adapt_raw(chosen, EBM_BATCH, device, adapter)
        z_g   = _adapt_raw(chosen, EBM_BATCH, device, adapter)
        z_raw_n, _, _ = buf.sample_raw(EBM_BATCH, device)
        z_n = adapter(z_raw_n)
        total_loss = total_loss + ebm.contrastive_loss(z_pos, z_n, z_g)
        n_terms += 1

    if len(her_buf) >= EBM_HER_MIN:
        z_her = _adapt_raw(her_buf, EBM_BATCH, device, adapter)
        z_raw_n2, _, _ = buf.sample_raw(EBM_BATCH, device)
        z_n2 = adapter(z_raw_n2)
        total_loss = total_loss + ebm.contrastive_loss(z_her, z_n2, z_her.detach())
        n_terms += 1

    if len(goal_buf) >= EBM_MIN_GOALS and len(post_door_neg_buf) >= EBM_POST_DOOR_MIN:
        z_exit   = _adapt_raw(goal_buf, EBM_BATCH, device, adapter)
        z_g_exit = _adapt_raw(goal_buf, EBM_BATCH, device, adapter)
        z_rh_neg = _adapt_raw(post_door_neg_buf, EBM_BATCH, device, adapter)
        total_loss = total_loss + ebm.contrastive_loss(z_exit, z_rh_neg, z_g_exit)
        n_terms += 1

    if n_terms > 0:
        total_loss.backward(); opt_ebm.step()
    return n_terms > 0


# ── Eval ──────────────────────────────────────────────────────────────────

def _eval_run15a(mpc, vjepa2, adapter, key_buf, door_buf, goal_buf, device, seed_offset=1000, n_eps=EVAL_N_EPS):
    if mpc is None:
        return 0.0
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        uw = env.unwrapped
        done, ep_steps, ep_ret = False, 0, 0.0
        current_stage, goal_age = 0, 0

        def _pick_adapted(stage):
            b = key_buf if stage == 0 else door_buf if stage == 1 else goal_buf
            raw = b.sample_raw(1)
            if raw is None: return None
            with torch.no_grad():
                return adapter(torch.from_numpy(raw).to(device)).cpu().numpy()

        z_goal_np = _pick_adapted(0)
        z_goal = torch.from_numpy(z_goal_np).to(device) if z_goal_np is not None else None

        while not done and ep_steps < EP_MAX_STEPS:
            has_key   = uw.carrying is not None
            d_open    = _is_door_open(uw)
            new_stage = 0 if not has_key else (1 if not d_open else 2)
            if new_stage != current_stage or goal_age >= GOAL_REFRESH_STEPS:
                current_stage = new_stage
                z_goal_np = _pick_adapted(current_stage)
                z_goal = torch.from_numpy(z_goal_np).to(device) if z_goal_np is not None else None
                goal_age = 0

            with torch.no_grad():
                z_raw = _encode(vjepa2, obs["image"], device)
                z_t   = adapter(z_raw)   # (1, 128)

            action = (mpc.plan_single(z_t, z_goal)
                      if mpc is not None and z_goal is not None
                      else env.action_space.sample())

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; ep_ret += r; ep_steps += 1; goal_age += 1
            uw = env.unwrapped

        if ep_ret > 0.5: successes += 1
        env.close()
    return successes / n_eps


# ── Main loop ──────────────────────────────────────────────────────────────

def run_doorkey_run15a_loop(
    condition: str = "vjepa2_adapter_late_ebm",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "vjepa2_adapter_late_ebm":
        raise ValueError(f"loop_mpc_doorkey_run15a supports: vjepa2_adapter_late_ebm — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_run15a only supports env_type='doorkey'.")

    torch.manual_seed(seed); np.random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE

    logger.info(
        f"[RUN15A] V-JEPA 2.1 (frozen 768-dim) + adapter ({RAW_DIM}→{ADAPTER_DIM}) + H={CEM_HORIZON} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    vjepa2 = _load_vjepa2(device)
    _doorkey_discrimination_test(vjepa2, device)

    # All buffers store RAW_DIM (768) features
    key_buf           = GoalFeatureBuffer(capacity=GOAL_BUF_CAPACITY)
    door_buf          = GoalFeatureBuffer(capacity=GOAL_BUF_CAPACITY)
    goal_buf          = GoalFeatureBuffer(capacity=GOAL_BUF_CAPACITY)
    her_buf           = GoalFeatureBuffer(capacity=HER_CAPACITY)
    post_door_neg_buf = GoalFeatureBuffer(capacity=POST_DOOR_NEG_CAPACITY)
    seed_buf          = FeatureReplayBuffer(capacity=SEED_BUF_CAPACITY)
    buf               = FeatureReplayBuffer(capacity=REPLAY_CAPACITY)

    _seed_scripted(vjepa2, key_buf, door_buf, goal_buf, seed_buf, post_door_neg_buf,
                   device=device, n_eps=N_SEED_EPS, seed=seed + 999)

    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    # Adapter + predictor share one optimizer (adapter needs faster LR)
    adapter   = FeatureAdapter(RAW_DIM, ADAPTER_DIM).to(device)
    predictor = FeaturePredictor(ADAPTER_DIM, N_ACTIONS, PRED_HIDDEN).to(device)
    opt_ap    = optim.Adam([
        {"params": adapter.parameters(),   "lr": ADAPTER_LR},
        {"params": predictor.parameters(), "lr": PRED_LR},
    ])

    ebm     = EBMCostHead(latent_dim=ADAPTER_DIM).to(device)
    opt_ebm = optim.Adam(ebm.parameters(), lr=EBM_LR)
    ebm_train_count = 0
    ebm_active      = False

    mpc: Optional[CEMPlanner] = None
    running_z_mean = torch.zeros(ADAPTER_DIM, device=device)

    had_key         = np.zeros(n_envs, dtype=bool)
    door_open_flags = np.zeros(n_envs, dtype=bool)
    active_goal_z: List[Optional[np.ndarray]] = [None] * n_envs   # 128-dim adapted
    active_stage    = np.zeros(n_envs, dtype=np.int32)
    goal_ages       = np.zeros(n_envs, dtype=np.int32)
    ep_ret          = np.zeros(n_envs, dtype=np.float32)

    metrics: Dict = {"env_step":[], "success_rate":[], "ssl_loss_ewa":[], "mode":[], "wall_time_s":[], "per_tier":[]}
    pred_ewa = None
    steps_to_80 = None
    env_step = act_steps = total_observe_steps = 0

    while env_step < max_steps:
        in_observe = env_step < _observe_steps
        mode_str   = "OBSERVE" if in_observe else "ACT"

        # Encode raw visual features (no grad, not through adapter yet)
        z_raw_t  = _encode(vjepa2, obs["image"], device)   # (n_envs, 768)
        z_raw_np = z_raw_t.cpu().numpy()

        if in_observe:
            if len(buf) >= PRED_WARMUP:
                actions = _curiosity_actions(adapter, predictor, z_raw_t, running_z_mean, device)
                with torch.no_grad():
                    z_adapted_cur = adapter(z_raw_t)
                running_z_mean.mul_(0.99).add_(z_adapted_cur.mean(0).detach() * 0.01)
            else:
                actions = envs.action_space.sample()
        else:
            # ACT: adapt raw features, then plan with CEM
            with torch.no_grad():
                z_adapted_t = adapter(z_raw_t)   # (n_envs, 128)

            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key   = uw.carrying is not None
                d_open    = _is_door_open(uw)
                new_stage = 0 if not has_key else (1 if not d_open else 2)
                if new_stage != active_stage[i] or active_goal_z[i] is None or goal_ages[i] >= GOAL_REFRESH_STEPS:
                    active_stage[i] = new_stage
                    g_buf = key_buf if new_stage == 0 else (door_buf if new_stage == 1 else goal_buf)
                    raw = g_buf.sample_raw(1)
                    if raw is not None:
                        with torch.no_grad():
                            adapted = adapter(torch.from_numpy(raw).to(device)).cpu().numpy()
                        active_goal_z[i] = adapted[0]
                    else:
                        active_goal_z[i] = None
                    goal_ages[i] = 0

            if mpc is not None:
                valid = [z for z in active_goal_z if z is not None]
                if valid:
                    goal_stack = np.stack(
                        active_goal_z if all(z is not None for z in active_goal_z)
                        else [valid[0]] * n_envs
                    )
                    z_goal_t = torch.from_numpy(goal_stack).to(device)
                    actions = mpc.plan_batch(z_adapted_t, z_goal_t)
                else:
                    actions = envs.action_space.sample()
            else:
                actions = envs.action_space.sample()

        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        dones = terms | truncs
        if not in_observe: ep_ret += rewards
        env_step += n_envs
        if in_observe: total_observe_steps += n_envs
        else:          act_steps += n_envs

        z_next_raw_t  = _encode(vjepa2, next_obs["image"], device)
        z_next_raw_np = z_next_raw_t.cpu().numpy()

        final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
        if final_mask.any() and "final_observation" in infos:
            fin_imgs = infos["final_observation"]["image"]
            for i in range(n_envs):
                if final_mask[i]:
                    z_next_raw_np[i] = _encode(vjepa2, fin_imgs[i], device).cpu().numpy()[0]

        # Store raw 768-dim in all buffers
        for i in range(n_envs):
            buf.push(z_raw_np[i], int(actions[i]), z_next_raw_np[i])
            uw = envs.envs[i].unwrapped
            if not had_key[i] and uw.carrying is not None:
                key_buf.push(z_next_raw_np[i]); had_key[i] = True
            if not door_open_flags[i] and _is_door_open(uw):
                door_buf.push(z_next_raw_np[i]); door_open_flags[i] = True
            if rewards[i] > 0:
                goal_buf.push(z_next_raw_np[i])
            elif _is_door_open(uw):
                post_door_neg_buf.push(z_next_raw_np[i])
            if dones[i]:
                her_buf.push(z_next_raw_np[i])
                had_key[i] = door_open_flags[i] = False
                if not in_observe:
                    ep_ret[i] = 0.0; active_goal_z[i] = None; goal_ages[i] = 0
            elif not in_observe:
                goal_ages[i] += 1

        obs = next_obs

        pred_loss = _train_predictor(adapter, predictor, opt_ap, buf, seed_buf, device)
        if pred_loss is not None:
            pred_ewa = pred_loss if pred_ewa is None else 0.95 * pred_ewa + 0.05 * pred_loss

            # Only train EBM after OBSERVE ends — adapter must be stable first
            if not in_observe:
                if _train_ebm(adapter, ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf,
                              post_door_neg_buf, buf, device):
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                        mpc.set_ebm(ebm); ebm_active = True
                        logger.info(f"[RUN15A] EBM activated (step={env_step}) | "
                                    f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}")

            if mpc is None and len(buf) >= PRED_WARMUP:
                mpc = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=CEM_HORIZON,
                                 n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                                 device=device, distance="cosine")
                logger.info(f"[RUN15A] CEM ready (H={CEM_HORIZON}, adapter_dim={ADAPTER_DIM}) | "
                            f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}")

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            logger.info(
                f"[RUN15A] heartbeat step={env_step} | buf={len(buf)} seed_buf={len(seed_buf)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
                f"post_neg={len(post_door_neg_buf)} her={len(her_buf)} | "
                f"pred_ewa={pred_ewa or 0.0:.4f} | H={CEM_HORIZON} | "
                f"ebm={'ON' if ebm_active else f'training({ebm_train_count})'} | {time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_run15a(mpc, vjepa2, adapter, key_buf, door_buf, goal_buf,
                              device, seed_offset=9000 + env_step, n_eps=EVAL_N_EPS)
            elapsed = time.time() - t0
            logger.info(
                f"[RUN15A] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | pred_ewa={pred_ewa or 0.0:.4f} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
                f"post_neg={len(post_door_neg_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(pred_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({"key":len(key_buf),"door":len(door_buf),"goal":len(goal_buf),"post_neg":len(post_door_neg_buf)})
            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[RUN15A] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0
    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"condition": condition, "adapter": adapter.state_dict(), "predictor": predictor.state_dict()},
        ckpt_dir / f"checkpoint_{condition}.pt",
    )
    logger.info(f"[RUN15A] Done | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"], "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"], "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps, "observe_steps": total_observe_steps,
    }


run_abm_loop = run_doorkey_run15a_loop
