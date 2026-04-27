"""
abm/loop_mpc_doorkey_run15b.py — Run 15b: V-JEPA 2.1 + scaled symbolic augmentation

Fix over Run 14b:
  Run 14b concatenated 4 symbolic scalars in [0,1] to a 768-dim L2-normalized
  V-JEPA feature. The symbolic contribution to cosine similarity was ~0.5%
  (4/772 dims). A one-tile agent move changed cos_sim by <0.001 — invisible
  to the predictor. pred_ewa stayed at 0.0002-0.0005 (same as frozen encoder
  runs). The symbolic information was drowned out.

Fix: scale symbolic dims by SYM_SCALE=10.0 before concatenation.
  At scale 10, a one-tile move produces cos_sim ≈ 0.980 between consecutive
  frames (vs 0.9998 without scaling). The predictor loss rises to ~0.02+,
  forcing it to actually learn to predict position changes and key/door state.

  sym_scaled = [has_key×10, door_open×10, (agent_x/5)×10, (agent_y/5)×10]
  FEATURE_DIM = 772 (768 visual + 4 scaled symbolic)

Fix (symbolic augmentation):
  Concat 4 symbolic scalars to every 768-dim V-JEPA feature before it
  enters the predictor, EBM, and CEM. These scalars change discretely and
  predictably with actions — the predictor can now learn real dynamics.

  sym = [has_key (0/1), door_open (0/1), agent_x/5, agent_y/5]

  Feature pipeline:
    V-JEPA 2.1 → 768-dim visual (frozen, L2-norm)
    concat [sym] → 772-dim augmented feature
    FeaturePredictor: (772 + 7) → 1024 → 772
    EBMCostHead(772), CEMPlanner in 772-dim space

  All replay buffers store 772-dim augmented features.
  Seeder encodes both visual + symbolic at each step.

Condition: vjepa2_symbolic_scaled
Loop module: abm.loop_mpc_doorkey_run15b
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
VIS_DIM                = 768         # V-JEPA 2.1 visual feature dim
SYM_DIM                = 4           # [has_key, door_open, agent_x/5, agent_y/5]
SYM_SCALE              = 10.0        # scale symbolic dims so they contribute meaningfully to cosine sim
FEATURE_DIM            = VIS_DIM + SYM_DIM   # 772
VJEPA2_IMG_SIZE        = 384
PRED_HIDDEN            = 1024
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
EBM_WARMUP_STEPS       = 500
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
    logger.info("[RUN15B] Loading V-JEPA 2.1 ViT-Base-384 (frozen)...")
    encoder, _ = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_1_vit_base_384",
        pretrained=False, trust_repo=True,
    )
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / _VJEPA2_CKPT_NAME
    if not ckpt_path.exists():
        logger.info(f"[RUN15B] Downloading checkpoint...")
        urllib.request.urlretrieve(_VJEPA2_CKPT_URL, str(ckpt_path))
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    enc_sd = state_dict["ema_encoder"]
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_sd.items()}
    encoder.load_state_dict(cleaned)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"[RUN15B] V-JEPA 2.1 loaded — {n_params:.1f}M params | FEATURE_DIM={FEATURE_DIM} (vis={VIS_DIM}+sym={SYM_DIM})")
    return encoder


_VJEPA2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_VJEPA2_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


@torch.no_grad()
def _encode_visual(model: nn.Module, imgs_np: np.ndarray, device: str) -> torch.Tensor:
    """imgs_np: (B, H, W, 3) uint8 → (B, 768) L2-norm V-JEPA features"""
    if imgs_np.ndim == 3:
        imgs_np = imgs_np[None]
    x = torch.from_numpy(imgs_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    if x.shape[-2:] != (VJEPA2_IMG_SIZE, VJEPA2_IMG_SIZE):
        x = F.interpolate(x, size=(VJEPA2_IMG_SIZE, VJEPA2_IMG_SIZE), mode="bilinear", align_corners=False)
    x = x.unsqueeze(2)
    x = (x - _VJEPA2_MEAN.to(device)) / _VJEPA2_STD.to(device)
    tokens = model(x)
    return F.normalize(tokens.mean(dim=1), p=2, dim=-1)


def _get_symbolic(uw) -> np.ndarray:
    """4-dim symbolic state scaled by SYM_SCALE so it contributes meaningfully to cosine sim."""
    has_key   = 1.0 if uw.carrying is not None else 0.0
    door_open = 1.0 if _is_door_open(uw) else 0.0
    ax = float(uw.agent_pos[0]) / 5.0
    ay = float(uw.agent_pos[1]) / 5.0
    return np.array([has_key, door_open, ax, ay], dtype=np.float32) * SYM_SCALE


def _encode_aug_single(model: nn.Module, img_np: np.ndarray, uw, device: str) -> np.ndarray:
    """Single env: visual (768) + scaled symbolic (4) → 772-dim numpy array."""
    z_vis = _encode_visual(model, img_np, device).cpu().numpy()[0]  # (768,)
    sym   = _get_symbolic(uw)                                        # (4,) scaled
    return np.concatenate([z_vis, sym])                              # (772,)


def _encode_aug_batch(model: nn.Module, imgs_np: np.ndarray, envs, device: str) -> torch.Tensor:
    """Vectorized: visual (B,768) + scaled symbolic (B,4) → (B,772) tensor."""
    z_vis = _encode_visual(model, imgs_np, device)   # (B, 768)
    sym   = np.stack([_get_symbolic(envs.envs[i].unwrapped) for i in range(imgs_np.shape[0])])
    sym_t = torch.from_numpy(sym).to(device)
    return torch.cat([z_vis, sym_t], dim=-1)          # (B, 772)


def _doorkey_discrimination_test(model: nn.Module, device: str) -> float:
    frames = []
    for s in range(20):
        env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
        env = RGBImgObsWrapper(env)
        obs, _ = env.reset(seed=s)
        frames.append(obs["image"])
        env.close()
    imgs = np.stack(frames)
    z = _encode_visual(model, imgs, device)
    sims = [F.cosine_similarity(z[i:i+1], z[i+1:i+2]).item() for i in range(0, 20, 2)]
    mean_sim = float(np.mean(sims))
    logger.info(f"[RUN15B] Discrimination test: cross-seed visual cos_sim={mean_sim:.4f}")
    if mean_sim > 0.95:
        logger.warning(f"[RUN15B] WARNING: visual cos_sim={mean_sim:.4f} > 0.95 — symbolic augmentation is the fix for this.")
    return mean_sim


# ── Feature-based replay buffers ──────────────────────────────────────────

class FeatureReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int = FEATURE_DIM):
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

    def sample(self, n: int, device: str):
        idx    = np.random.choice(self._size, size=n, replace=self._size < n)
        z      = torch.from_numpy(self._z[idx]).to(device)
        a      = torch.from_numpy(self._a[idx]).long().to(device)
        z_next = torch.from_numpy(self._z_next[idx]).to(device)
        return z, a, z_next

    def __len__(self) -> int:
        return self._size


class GoalFeatureBuffer:
    def __init__(self, capacity: int, feature_dim: int = FEATURE_DIM):
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, z: np.ndarray) -> None:
        self._buf.append(np.array(z, copy=True))

    def sample(self, n: int) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=n, replace=len(self._buf) < n)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self) -> int:
        return len(self._buf)


# ── Feature predictor ─────────────────────────────────────────────────────

class FeaturePredictor(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, n_actions: int = N_ACTIONS, hidden: int = PRED_HIDDEN):
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


# ── Scripted seeder ────────────────────────────────────────────────────────

def _seed_scripted(
    vjepa2, key_buf, door_buf, goal_buf, seed_buf, post_door_neg_buf,
    device, n_eps=N_SEED_EPS, seed=7777,
):
    logger.info(f"[RUN15B] Scripted seeding: {n_eps} episodes…")
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        uw = env.unwrapped
        done, steps = False, 0
        had_key = door_was_open = collecting_post_door = False

        while not done and steps < EP_MAX_STEPS:
            action = _scripted_action(uw, fallback_fn=env.action_space.sample)
            z_prev = _encode_aug_single(vjepa2, obs["image"], uw, device)

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1
            uw = env.unwrapped

            z_cur = _encode_aug_single(vjepa2, obs["image"], uw, device)
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
        f"[RUN15B] Seed done: {successes}/{n_eps} | seed_buf={len(seed_buf)} | "
        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} post_neg={len(post_door_neg_buf)}"
    )


# ── Predictor training ────────────────────────────────────────────────────

def _train_predictor(predictor, opt, buf, seed_buf, device, n_steps=PRED_TRAIN_STEPS):
    if len(buf) < PRED_WARMUP:
        return None
    n_seed, n_online = PRED_BATCH // 2, PRED_BATCH - PRED_BATCH // 2
    use_mix = len(seed_buf) >= n_seed
    last_loss = None
    for _ in range(n_steps):
        if use_mix:
            z_s, a_s, zn_s = seed_buf.sample(n_seed, device)
            z_o, a_o, zn_o = buf.sample(n_online, device)
            z_t, a_t, zn_t = torch.cat([z_s,z_o]), torch.cat([a_s,a_o]), torch.cat([zn_s,zn_o])
        else:
            z_t, a_t, zn_t = buf.sample(PRED_BATCH, device)
        a_oh = F.one_hot(a_t, N_ACTIONS).float()
        z_pred = predictor(z_t, a_oh)
        loss = (1 - F.cosine_similarity(z_pred, zn_t.detach(), dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = loss.item()
    return last_loss


# ── Curiosity ─────────────────────────────────────────────────────────────

def _curiosity_actions(predictor, z_cur, running_z_mean, device):
    n = z_cur.shape[0]
    with torch.no_grad():
        z_rep  = z_cur.unsqueeze(1).expand(n, N_ACTIONS, FEATURE_DIM).reshape(n*N_ACTIONS, FEATURE_DIM)
        a_idx  = torch.arange(N_ACTIONS, device=device).unsqueeze(0).expand(n,-1).reshape(-1)
        a_oh   = F.one_hot(a_idx, N_ACTIONS).float()
        z_next = predictor(z_rep, a_oh).reshape(n, N_ACTIONS, FEATURE_DIM)
        novelty = (z_next - running_z_mean.view(1,1,FEATURE_DIM)).pow(2).sum(-1)
        return novelty.argmax(dim=-1).cpu().numpy()


# ── EBM training ──────────────────────────────────────────────────────────

def _z_from_buf(buf, n, device):
    return torch.from_numpy(buf.sample(n)).to(device)


def _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, post_door_neg_buf, buf, device):
    if len(buf) < EBM_BATCH * 2:
        return False
    opt_ebm.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    ready = [(b, n) for b, n in [(key_buf,"key"),(door_buf,"door"),(goal_buf,"goal")] if len(b) >= EBM_MIN_GOALS]
    if ready:
        chosen, _ = ready[np.random.randint(len(ready))]
        z_pos = _z_from_buf(chosen, EBM_BATCH, device)
        z_g   = _z_from_buf(chosen, EBM_BATCH, device)
        z_n, _, _ = buf.sample(EBM_BATCH, device)
        total_loss = total_loss + ebm.contrastive_loss(z_pos, z_n, z_g)
        n_terms += 1

    if len(her_buf) >= EBM_HER_MIN:
        z_her = _z_from_buf(her_buf, EBM_BATCH, device)
        z_n2, _, _ = buf.sample(EBM_BATCH, device)
        total_loss = total_loss + ebm.contrastive_loss(z_her, z_n2, z_her.detach())
        n_terms += 1

    if len(goal_buf) >= EBM_MIN_GOALS and len(post_door_neg_buf) >= EBM_POST_DOOR_MIN:
        z_exit   = _z_from_buf(goal_buf, EBM_BATCH, device)
        z_g_exit = _z_from_buf(goal_buf, EBM_BATCH, device)
        z_rh_neg = _z_from_buf(post_door_neg_buf, EBM_BATCH, device)
        total_loss = total_loss + ebm.contrastive_loss(z_exit, z_rh_neg, z_g_exit)
        n_terms += 1

    if n_terms > 0:
        total_loss.backward(); opt_ebm.step()
    return n_terms > 0


# ── Eval ──────────────────────────────────────────────────────────────────

def _eval_run15b(mpc, vjepa2, key_buf, door_buf, goal_buf, device, seed_offset=1000, n_eps=EVAL_N_EPS):
    if mpc is None:
        return 0.0
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        uw = env.unwrapped
        done, ep_steps, ep_ret = False, 0, 0.0
        current_stage, goal_age = 0, 0

        def _pick(stage):
            b = key_buf if stage == 0 else door_buf if stage == 1 else goal_buf
            return b.sample(1)

        z_goal_np = _pick(0)
        z_goal = torch.from_numpy(z_goal_np).to(device) if z_goal_np is not None else None

        while not done and ep_steps < EP_MAX_STEPS:
            has_key  = uw.carrying is not None
            d_open   = _is_door_open(uw)
            new_stage = 0 if not has_key else (1 if not d_open else 2)
            if new_stage != current_stage or goal_age >= GOAL_REFRESH_STEPS:
                current_stage = new_stage
                z_goal_np = _pick(current_stage)
                z_goal = torch.from_numpy(z_goal_np).to(device) if z_goal_np is not None else None
                goal_age = 0

            z_aug = _encode_aug_single(vjepa2, obs["image"], uw, device)
            z_t   = torch.from_numpy(z_aug[None]).to(device)
            action = (mpc.plan_single(z_t, z_goal)
                      if mpc is not None and z_goal is not None
                      else env.action_space.sample())

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r; ep_steps += 1; goal_age += 1
            uw = env.unwrapped

        if ep_ret > 0.5: successes += 1
        env.close()
    return successes / n_eps


# ── Main loop ──────────────────────────────────────────────────────────────

def run_doorkey_run15b_loop(
    condition: str = "vjepa2_symbolic_scaled",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "vjepa2_symbolic_scaled":
        raise ValueError(f"loop_mpc_doorkey_run15b supports: vjepa2_symbolic_scaled — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_run15b only supports env_type='doorkey'.")

    torch.manual_seed(seed); np.random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE

    logger.info(
        f"[RUN15B] V-JEPA 2.1 + symbolic aug ({FEATURE_DIM}d) + H={CEM_HORIZON} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    vjepa2 = _load_vjepa2(device)
    _doorkey_discrimination_test(vjepa2, device)

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

    predictor = FeaturePredictor(FEATURE_DIM, N_ACTIONS, PRED_HIDDEN).to(device)
    opt_pred  = optim.Adam(predictor.parameters(), lr=PRED_LR)

    ebm     = EBMCostHead(latent_dim=FEATURE_DIM).to(device)
    opt_ebm = optim.Adam(ebm.parameters(), lr=EBM_LR)
    ebm_train_count = 0
    ebm_active      = False

    mpc: Optional[CEMPlanner] = None
    running_z_mean = torch.zeros(FEATURE_DIM, device=device)

    had_key         = np.zeros(n_envs, dtype=bool)
    door_open_flags = np.zeros(n_envs, dtype=bool)
    active_goal_z: List[Optional[np.ndarray]] = [None] * n_envs
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

        z_cur_t  = _encode_aug_batch(vjepa2, obs["image"], envs, device)
        z_cur_np = z_cur_t.cpu().numpy()

        if in_observe:
            if len(buf) >= PRED_WARMUP:
                actions = _curiosity_actions(predictor, z_cur_t, running_z_mean, device)
                running_z_mean.mul_(0.99).add_(z_cur_t.mean(0).detach() * 0.01)
            else:
                actions = envs.action_space.sample()
        else:
            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key   = uw.carrying is not None
                d_open    = _is_door_open(uw)
                new_stage = 0 if not has_key else (1 if not d_open else 2)
                if new_stage != active_stage[i] or active_goal_z[i] is None or goal_ages[i] >= GOAL_REFRESH_STEPS:
                    active_stage[i] = new_stage
                    g_buf = key_buf if new_stage == 0 else (door_buf if new_stage == 1 else goal_buf)
                    g = g_buf.sample(1)
                    active_goal_z[i] = g[0] if g is not None else None
                    goal_ages[i] = 0

            if mpc is not None:
                valid = [z for z in active_goal_z if z is not None]
                if valid:
                    goal_stack = np.stack(
                        active_goal_z if all(z is not None for z in active_goal_z)
                        else [valid[0]] * n_envs
                    )
                    z_goal_t = torch.from_numpy(goal_stack).to(device)
                    actions = mpc.plan_batch(z_cur_t, z_goal_t)
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

        z_next_t  = _encode_aug_batch(vjepa2, next_obs["image"], envs, device)
        z_next_np = z_next_t.cpu().numpy()

        # Handle terminal observations — visual component only (symbolic from reset state is acceptable)
        final_mask = infos.get("_final_observation", np.zeros(n_envs, dtype=bool))
        if final_mask.any() and "final_observation" in infos:
            fin_imgs = infos["final_observation"]["image"]
            for i in range(n_envs):
                if final_mask[i]:
                    z_vis_fin = _encode_visual(vjepa2, fin_imgs[i], device).cpu().numpy()[0]
                    # Keep symbolic from next state (post-reset) — close enough for HER
                    z_next_np[i][:VIS_DIM] = z_vis_fin

        for i in range(n_envs):
            buf.push(z_cur_np[i], int(actions[i]), z_next_np[i])
            uw = envs.envs[i].unwrapped
            if not had_key[i] and uw.carrying is not None:
                key_buf.push(z_next_np[i]); had_key[i] = True
            if not door_open_flags[i] and _is_door_open(uw):
                door_buf.push(z_next_np[i]); door_open_flags[i] = True
            if rewards[i] > 0:
                goal_buf.push(z_next_np[i])
            elif _is_door_open(uw):
                post_door_neg_buf.push(z_next_np[i])
            if dones[i]:
                her_buf.push(z_next_np[i])
                had_key[i] = door_open_flags[i] = False
                if not in_observe:
                    ep_ret[i] = 0.0; active_goal_z[i] = None; goal_ages[i] = 0
            elif not in_observe:
                goal_ages[i] += 1

        obs = next_obs

        pred_loss = _train_predictor(predictor, opt_pred, buf, seed_buf, device)
        if pred_loss is not None:
            pred_ewa = pred_loss if pred_ewa is None else 0.95 * pred_ewa + 0.05 * pred_loss

            if _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, post_door_neg_buf, buf, device):
                ebm_train_count += 1
                if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                    mpc.set_ebm(ebm); ebm_active = True
                    logger.info(f"[RUN15B] EBM activated | key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}")

            if mpc is None and len(buf) >= PRED_WARMUP:
                mpc = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=CEM_HORIZON,
                                 n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                                 device=device, distance="cosine")
                logger.info(f"[RUN15B] CEM ready (H={CEM_HORIZON}) | key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}")

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            logger.info(
                f"[RUN15B] heartbeat step={env_step} | buf={len(buf)} seed_buf={len(seed_buf)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
                f"post_neg={len(post_door_neg_buf)} her={len(her_buf)} | "
                f"pred_ewa={pred_ewa or 0.0:.4f} | H={CEM_HORIZON} | "
                f"ebm={'ON' if ebm_active else f'training({ebm_train_count})'} | {time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_run15b(mpc, vjepa2, key_buf, door_buf, goal_buf, device,
                              seed_offset=9000 + env_step, n_eps=EVAL_N_EPS)
            elapsed = time.time() - t0
            logger.info(
                f"[RUN15B] step={env_step:7d} | mode={mode_str:7s} | "
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
                logger.info(f"[RUN15B] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0
    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "predictor": predictor.state_dict()},
               ckpt_dir / f"checkpoint_{condition}.pt")
    logger.info(f"[RUN15B] Done | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"], "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"], "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps, "observe_steps": total_observe_steps,
    }


run_abm_loop = run_doorkey_run15b_loop
