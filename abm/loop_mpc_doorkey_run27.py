"""
abm/loop_mpc_doorkey_run27.py — Run 27: Fix stage-2 goal episode-mixing bug from Run 26

Bug fixed (Run 26 root cause):
  DoorKey randomises the goal cell position each episode.  Run 26's stage-2 CEM
  sampled the goal latent from goal_buf, which stores terminal states from past
  episodes where the goal was at a DIFFERENT (x, y) than the current episode.
  Stages 0/1 are unaffected because has_key=1 and door_open=1 are binary flags
  independent of position.  Stage 2 navigates to the exit — position IS the signal.
  Planning toward the wrong cell produces random actions; the agent can never
  complete stage 2 reliably.

Fix:
  Instead of goal_buf.sample(1), construct the stage-2 target state directly from
  the current environment:
      gp = _find_cell(uw, "goal")
      active_goal_z[i] = np.array([1., 1., gp[0]/5., gp[1]/5., 0.])
  Same fix applied in the eval function so evaluation is consistent.

  Everything else is identical to Run 26:
    • Stages 0/1: CEM H=8 + EBM (proven effective, Run 24)
    • Stage 2:   CEM H=3 + EBM, goal read from current env every GOAL_REFRESH_STEPS
    • Predictor frozen at ACT start (prevents pred_ewa degradation, Run 24)
    • No BFS oracle.  No PPO.  Pure world-model planning for all three stages.

Condition: symbolic_exact_goal_s2
Loop module: abm.loop_mpc_doorkey_run27
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import RGBImgObsWrapper

from .cem_planner import CEMPlanner, EBMCostHead

logger = logging.getLogger(__name__)

N_ENVS                 = 16
N_ACTIONS              = 7
FEATURE_DIM            = 5
PRED_HIDDEN            = 64
PRED_LR                = 3e-4
PRED_BATCH             = 256
PRED_WARMUP            = 100
PRED_TRAIN_STEPS       = 4
REPLAY_CAPACITY        = 100_000
SEED_BUF_CAPACITY      = 20_000
POST_DOOR_NEG_CAPACITY = 5_000
GOAL_BUF_CAPACITY      = 1_024
HER_CAPACITY           = 4_096
EVAL_INTERVAL          = 5_000
EVAL_N_EPS             = 20
GOAL_REFRESH_STEPS     = 64
DEFAULT_OBSERVE        = 40_000
EP_MAX_STEPS           = 300
EBM_MIN_GOALS          = 3
EBM_HER_MIN            = 10
EBM_POST_DOOR_MIN      = 10
EBM_WARMUP_STEPS       = 200
EBM_LR                 = 3e-4
EBM_BATCH              = 32
N_SEED_EPS             = 200

CEM_HORIZON            = 8    # stages 0/1 — same as Run 24
CEM_HORIZON_S2         = 3    # stage 2 — short horizon per HWM Figure 6
CEM_SAMPLES            = 512
CEM_ELITES             = 64
CEM_ITERS              = 5


# ── Symbolic state ────────────────────────────────────────────────────────

def _get_symbolic(uw) -> np.ndarray:
    return np.array([
        1.0 if uw.carrying is not None else 0.0,
        1.0 if _is_door_open(uw) else 0.0,
        float(uw.agent_pos[0]) / 5.0,
        float(uw.agent_pos[1]) / 5.0,
        float(uw.agent_dir)    / 3.0,
    ], dtype=np.float32)


def _get_symbolic_batch(envs, n_envs: int) -> np.ndarray:
    return np.stack([_get_symbolic(envs.envs[i].unwrapped) for i in range(n_envs)])


# ── Replay buffers ────────────────────────────────────────────────────────

class FeatureReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int = FEATURE_DIM):
        self.capacity    = capacity
        self._z      = np.zeros((capacity, feature_dim), dtype=np.float32)
        self._a      = np.zeros(capacity, dtype=np.int64)
        self._z_next = np.zeros((capacity, feature_dim), dtype=np.float32)
        self._ptr    = 0
        self._size   = 0

    def push(self, z, a, z_next):
        self._z[self._ptr]      = z
        self._a[self._ptr]      = a
        self._z_next[self._ptr] = z_next
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n, device):
        idx = np.random.choice(self._size, size=n, replace=self._size < n)
        return (torch.from_numpy(self._z[idx]).to(device),
                torch.from_numpy(self._a[idx]).long().to(device),
                torch.from_numpy(self._z_next[idx]).to(device))

    def __len__(self):
        return self._size


class GoalFeatureBuffer:
    def __init__(self, capacity: int):
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, z):
        self._buf.append(np.array(z, copy=True))

    def sample(self, n) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        idx = np.random.choice(len(self._buf), size=n, replace=len(self._buf) < n)
        return np.stack([self._buf[i] for i in idx])

    def __len__(self):
        return len(self._buf)


# ── Predictor ─────────────────────────────────────────────────────────────

class FeaturePredictor(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, n_actions=N_ACTIONS, hidden=PRED_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, z, a_oh):
        return F.normalize(self.net(torch.cat([z, a_oh], dim=-1)), p=2, dim=-1)

    def forward_sequence(self, z_seq, a_oh_seq):
        z = z_seq[:, 0]
        preds = []
        for t in range(z_seq.shape[1]):
            z = self.forward(z, a_oh_seq[:, t]); preds.append(z)
        return torch.stack(preds, dim=1)


# ── Environment helpers ───────────────────────────────────────────────────

def _make_doorkey_env(seed=0):
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    return RGBImgObsWrapper(env)


def _make_doorkey_vec_env(n_envs, seed=0):
    fns = [lambda i=i: _make_doorkey_env(seed + i) for i in range(n_envs)]
    return gymnasium.vector.SyncVectorEnv(fns)


def _is_door_open(uw) -> bool:
    for col in range(uw.grid.width):
        for row in range(uw.grid.height):
            cell = uw.grid.get(col, row)
            if cell is not None and cell.type == "door" and cell.is_open:
                return True
    return False


# ── BFS scripted policy (seeder only) ────────────────────────────────────

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


# ── Scripted seeder ────────────────────────────────────────────────────────

def _seed_scripted(key_buf, door_buf, goal_buf, seed_buf, post_door_neg_buf,
                   n_eps=N_SEED_EPS, seed=7777):
    logger.info(f"[RUN27] Scripted seeding: {n_eps} episodes (5-dim symbolic)…")
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        uw = env.unwrapped
        done, steps = False, 0
        had_key = door_was_open = collecting_post_door = False

        while not done and steps < EP_MAX_STEPS:
            action = _scripted_action(uw, fallback_fn=env.action_space.sample)
            z_prev = _get_symbolic(uw)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; steps += 1
            uw = env.unwrapped
            z_cur = _get_symbolic(uw)
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
        f"[RUN27] Seed done: {successes}/{n_eps} | seed_buf={len(seed_buf)} | "
        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} post_neg={len(post_door_neg_buf)}"
    )


# ── Training ──────────────────────────────────────────────────────────────

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
        a_oh   = F.one_hot(a_t, N_ACTIONS).float()
        z_pred = predictor(z_t, a_oh)
        loss   = (1 - F.cosine_similarity(z_pred, zn_t.detach(), dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = loss.item()
    return last_loss


def _curiosity_actions(predictor, z_cur, running_z_mean, device):
    n = z_cur.shape[0]
    with torch.no_grad():
        z_rep  = z_cur.unsqueeze(1).expand(n,N_ACTIONS,FEATURE_DIM).reshape(n*N_ACTIONS,FEATURE_DIM)
        a_idx  = torch.arange(N_ACTIONS,device=device).unsqueeze(0).expand(n,-1).reshape(-1)
        a_oh   = F.one_hot(a_idx,N_ACTIONS).float()
        z_next = predictor(z_rep,a_oh).reshape(n,N_ACTIONS,FEATURE_DIM)
        novelty = (z_next - running_z_mean.view(1,1,FEATURE_DIM)).pow(2).sum(-1)
        return novelty.argmax(dim=-1).cpu().numpy()


def _z_from_buf(buf, n, device):
    arr = buf.sample(n)
    return torch.from_numpy(arr).to(device)


def _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, post_door_neg_buf, buf, device):
    if len(buf) < EBM_BATCH * 2:
        return False
    opt_ebm.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    ready = [(b,n) for b,n in [(key_buf,"key"),(door_buf,"door"),(goal_buf,"goal")] if len(b) >= EBM_MIN_GOALS]
    if ready:
        chosen, _ = ready[np.random.randint(len(ready))]
        z_pos = _z_from_buf(chosen, EBM_BATCH, device)
        z_g   = _z_from_buf(chosen, EBM_BATCH, device)
        z_n, _, _ = buf.sample(EBM_BATCH, device)
        total_loss = total_loss + ebm.contrastive_loss(z_pos, z_n, z_g); n_terms += 1

    if len(her_buf) >= EBM_HER_MIN:
        z_her = _z_from_buf(her_buf, EBM_BATCH, device)
        z_n2, _, _ = buf.sample(EBM_BATCH, device)
        total_loss = total_loss + ebm.contrastive_loss(z_her, z_n2, z_her.detach()); n_terms += 1

    if len(goal_buf) >= EBM_MIN_GOALS and len(post_door_neg_buf) >= EBM_POST_DOOR_MIN:
        z_exit   = _z_from_buf(goal_buf, EBM_BATCH, device)
        z_g_exit = _z_from_buf(goal_buf, EBM_BATCH, device)
        z_rh_neg = _z_from_buf(post_door_neg_buf, EBM_BATCH, device)
        total_loss = total_loss + ebm.contrastive_loss(z_exit, z_rh_neg, z_g_exit); n_terms += 1

    if n_terms > 0:
        total_loss.backward(); opt_ebm.step()
    return n_terms > 0


def _train_ebm_softplus(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf,
                        post_door_neg_buf, buf, device):
    if len(buf) < EBM_BATCH * 2:
        return False
    opt_ebm.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    ready = [(b, n) for b, n in [(key_buf, "key"), (door_buf, "door"), (goal_buf, "goal")]
             if len(b) >= EBM_MIN_GOALS]
    if ready:
        chosen, _ = ready[np.random.randint(len(ready))]
        z_pos = _z_from_buf(chosen, EBM_BATCH, device)
        z_g   = _z_from_buf(chosen, EBM_BATCH, device)
        z_n, _, _ = buf.sample(EBM_BATCH, device)
        e_pos = ebm(z_pos, z_g); e_neg = ebm(z_n, z_g)
        total_loss = total_loss + F.softplus(e_pos - e_neg).mean(); n_terms += 1

    if len(her_buf) >= EBM_HER_MIN:
        z_her = _z_from_buf(her_buf, EBM_BATCH, device)
        z_n2, _, _ = buf.sample(EBM_BATCH, device)
        z_g_her = z_her.detach()
        e_pos = ebm(z_her, z_g_her); e_neg = ebm(z_n2, z_g_her)
        total_loss = total_loss + F.softplus(e_pos - e_neg).mean(); n_terms += 1

    if len(goal_buf) >= EBM_MIN_GOALS and len(post_door_neg_buf) >= EBM_POST_DOOR_MIN:
        z_exit   = _z_from_buf(goal_buf, EBM_BATCH, device)
        z_g_exit = _z_from_buf(goal_buf, EBM_BATCH, device)
        z_rh_neg = _z_from_buf(post_door_neg_buf, EBM_BATCH, device)
        e_pos = ebm(z_exit, z_g_exit); e_neg = ebm(z_rh_neg, z_g_exit)
        total_loss = total_loss + F.softplus(e_pos - e_neg).mean(); n_terms += 1

    if n_terms > 0:
        total_loss.backward(); opt_ebm.step()
    return n_terms > 0


# ── Eval ──────────────────────────────────────────────────────────────────

def _eval_run27(mpc, mpc_s2, key_buf, door_buf, device,
                seed_offset=1000, n_eps=EVAL_N_EPS):
    """Stages 0/1: CEM H=8 + EBM.  Stage 2: CEM H=3 + EBM, goal from current env."""
    if mpc is None:
        return 0.0
    successes = 0
    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        uw = env.unwrapped
        done, ep_steps, ep_ret = False, 0, 0.0
        current_stage, goal_age = 0, 0

        def _pick_s01(stage):
            b = key_buf if stage == 0 else door_buf
            return b.sample(1)

        z_goal_np = _pick_s01(0)
        z_goal = torch.from_numpy(z_goal_np).to(device) if z_goal_np is not None else None

        while not done and ep_steps < EP_MAX_STEPS:
            has_key  = uw.carrying is not None
            d_open   = _is_door_open(uw)
            new_stage = 0 if not has_key else (1 if not d_open else 2)
            if new_stage != current_stage or goal_age >= GOAL_REFRESH_STEPS:
                current_stage = new_stage
                if current_stage == 2:
                    # Read actual goal position from current episode
                    gp = _find_cell(uw, "goal")
                    z_goal_np = (
                        np.array([1., 1., gp[0]/5., gp[1]/5., 0.], dtype=np.float32)[None]
                        if gp else None
                    )
                else:
                    z_goal_np = _pick_s01(current_stage)
                z_goal = torch.from_numpy(z_goal_np).to(device) if z_goal_np is not None else None
                goal_age = 0

            z_t = torch.from_numpy(_get_symbolic(uw)[None]).to(device)
            if current_stage == 2:
                planner = mpc_s2 if mpc_s2 is not None else mpc
                action = (planner.plan_single(z_t, z_goal)
                          if z_goal is not None
                          else env.action_space.sample())
            else:
                action = (mpc.plan_single(z_t, z_goal)
                          if z_goal is not None
                          else env.action_space.sample())

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; ep_ret += r; ep_steps += 1; goal_age += 1
            uw = env.unwrapped

        if ep_ret > 0.5: successes += 1
        env.close()
    return successes / n_eps


# ── Main loop ──────────────────────────────────────────────────────────────

def run_doorkey_run27_loop(
    condition: str = "symbolic_exact_goal_s2",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "symbolic_exact_goal_s2":
        raise ValueError(f"loop_mpc_doorkey_run27 supports: symbolic_exact_goal_s2 — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_run27 only supports env_type='doorkey'.")

    torch.manual_seed(seed); np.random.seed(seed)
    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE

    logger.info(
        f"[RUN27] Symbolic (5-dim) | CEM H={CEM_HORIZON} stages 0/1 | "
        f"CEM H={CEM_HORIZON_S2} stage 2 (short-horizon, HWM) | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    key_buf           = GoalFeatureBuffer(capacity=GOAL_BUF_CAPACITY)
    door_buf          = GoalFeatureBuffer(capacity=GOAL_BUF_CAPACITY)
    goal_buf          = GoalFeatureBuffer(capacity=GOAL_BUF_CAPACITY)
    her_buf           = GoalFeatureBuffer(capacity=HER_CAPACITY)
    post_door_neg_buf = GoalFeatureBuffer(capacity=POST_DOOR_NEG_CAPACITY)
    seed_buf          = FeatureReplayBuffer(capacity=SEED_BUF_CAPACITY)
    buf               = FeatureReplayBuffer(capacity=REPLAY_CAPACITY)

    _seed_scripted(key_buf, door_buf, goal_buf, seed_buf, post_door_neg_buf,
                   n_eps=N_SEED_EPS, seed=seed + 999)

    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    predictor = FeaturePredictor(FEATURE_DIM, N_ACTIONS, PRED_HIDDEN).to(device)
    opt_pred  = optim.Adam(predictor.parameters(), lr=PRED_LR)

    ebm     = EBMCostHead(latent_dim=FEATURE_DIM).to(device)
    opt_ebm = optim.Adam(ebm.parameters(), lr=EBM_LR)
    ebm_train_count = 0
    ebm_active      = False

    mpc:    Optional[CEMPlanner] = None
    mpc_s2: Optional[CEMPlanner] = None
    running_z_mean = torch.zeros(FEATURE_DIM, device=device)

    had_key         = np.zeros(n_envs, dtype=bool)
    door_open_flags = np.zeros(n_envs, dtype=bool)
    active_goal_z: List[Optional[np.ndarray]] = [None] * n_envs
    active_stage    = np.zeros(n_envs, dtype=np.int32)
    goal_ages       = np.zeros(n_envs, dtype=np.int32)
    ep_ret          = np.zeros(n_envs, dtype=np.float32)

    metrics: Dict = {"env_step":[],"success_rate":[],"ssl_loss_ewa":[],"mode":[],"wall_time_s":[],"per_tier":[]}
    pred_ewa = None
    steps_to_80 = None
    env_step = act_steps = total_observe_steps = 0

    while env_step < max_steps:
        in_observe = env_step < _observe_steps
        mode_str   = "OBSERVE" if in_observe else "ACT"

        z_cur_np = _get_symbolic_batch(envs, n_envs)
        z_cur_t  = torch.from_numpy(z_cur_np).to(device)

        if in_observe:
            if len(buf) >= PRED_WARMUP:
                actions = _curiosity_actions(predictor, z_cur_t, running_z_mean, device)
                running_z_mean.mul_(0.99).add_(z_cur_t.mean(0).detach() * 0.01)
            else:
                actions = envs.action_space.sample()
        else:
            # Update stages and goals for all stages including stage 2
            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key   = uw.carrying is not None
                d_open    = _is_door_open(uw)
                new_stage = 0 if not has_key else (1 if not d_open else 2)
                if new_stage != active_stage[i]:
                    active_stage[i] = new_stage
                    active_goal_z[i] = None
                    goal_ages[i] = 0
                if active_goal_z[i] is None or goal_ages[i] >= GOAL_REFRESH_STEPS:
                    if active_stage[i] == 0:
                        g = key_buf.sample(1)
                        active_goal_z[i] = g[0] if g is not None else None
                        goal_ages[i] = 0
                    elif active_stage[i] == 1:
                        g = door_buf.sample(1)
                        active_goal_z[i] = g[0] if g is not None else None
                        goal_ages[i] = 0
                    else:
                        # Stage 2: goal position is randomised per episode — read current env
                        gp = _find_cell(envs.envs[i].unwrapped, "goal")
                        active_goal_z[i] = (
                            np.array([1., 1., gp[0]/5., gp[1]/5., 0.], dtype=np.float32)
                            if gp else None
                        )
                        goal_ages[i] = 0

            actions = np.zeros(n_envs, dtype=np.int64)

            s01_idx = [i for i in range(n_envs) if active_stage[i] < 2]
            s2_idx  = [i for i in range(n_envs) if active_stage[i] == 2]

            # Stages 0/1: CEM H=8
            if s01_idx and mpc is not None:
                has_goal = [i for i in s01_idx if active_goal_z[i] is not None]
                no_goal  = [i for i in s01_idx if active_goal_z[i] is None]
                if has_goal:
                    z_s01 = z_cur_t[has_goal]
                    g_s01 = torch.from_numpy(
                        np.stack([active_goal_z[i] for i in has_goal])
                    ).to(device)
                    actions[has_goal] = mpc.plan_batch(z_s01, g_s01)
                for i in no_goal:
                    actions[i] = int(np.random.randint(N_ACTIONS))
            elif s01_idx:
                for i in s01_idx:
                    actions[i] = int(np.random.randint(N_ACTIONS))

            # Stage 2: CEM H=3 (short-horizon receding-horizon replanning)
            if s2_idx and mpc_s2 is not None:
                has_goal_s2 = [i for i in s2_idx if active_goal_z[i] is not None]
                no_goal_s2  = [i for i in s2_idx if active_goal_z[i] is None]
                if has_goal_s2:
                    z_s2 = z_cur_t[has_goal_s2]
                    g_s2 = torch.from_numpy(
                        np.stack([active_goal_z[i] for i in has_goal_s2])
                    ).to(device)
                    actions[has_goal_s2] = mpc_s2.plan_batch(z_s2, g_s2)
                for i in no_goal_s2:
                    actions[i] = int(np.random.randint(N_ACTIONS))
            elif s2_idx:
                for i in s2_idx:
                    actions[i] = int(np.random.randint(N_ACTIONS))

        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        dones = terms | truncs
        if not in_observe: ep_ret += rewards
        env_step += n_envs
        if in_observe: total_observe_steps += n_envs
        else:          act_steps += n_envs

        z_next_np = _get_symbolic_batch(envs, n_envs)

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

        # Predictor: train during OBSERVE only — freeze at ACT start
        if in_observe:
            pred_loss = _train_predictor(predictor, opt_pred, buf, seed_buf, device)
            if pred_loss is not None:
                pred_ewa = pred_loss if pred_ewa is None else 0.95 * pred_ewa + 0.05 * pred_loss

        # EBM: hinge during OBSERVE, softplus during ACT
        if len(buf) >= PRED_WARMUP:
            train_fn = _train_ebm if in_observe else _train_ebm_softplus
            if train_fn(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf, post_door_neg_buf, buf, device):
                ebm_train_count += 1
                if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                    mpc.set_ebm(ebm)
                    mpc_s2.set_ebm(ebm)
                    ebm_active = True
                    logger.info(f"[RUN27] EBM activated at step {env_step} | "
                                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}")

        # CEM: create both planners as soon as buf is ready
        if mpc is None and len(buf) >= PRED_WARMUP:
            mpc = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=CEM_HORIZON,
                             n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                             device=device, distance="cosine")
            mpc_s2 = CEMPlanner(predictor, n_actions=N_ACTIONS, horizon=CEM_HORIZON_S2,
                                n_samples=CEM_SAMPLES, n_elites=CEM_ELITES, n_iters=CEM_ITERS,
                                device=device, distance="cosine")
            logger.info(
                f"[RUN27] CEM ready | stages 0/1: H={CEM_HORIZON} | stage 2: H={CEM_HORIZON_S2} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
            )

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            ebm_status = ('ON(softplus)' if (ebm_active and not in_observe)
                          else 'ON(hinge)' if ebm_active
                          else f'training({ebm_train_count})')
            s2_count = sum(1 for s in active_stage if s == 2)
            logger.info(
                f"[RUN27] heartbeat step={env_step} | buf={len(buf)} seed_buf={len(seed_buf)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
                f"post_neg={len(post_door_neg_buf)} | "
                f"pred_ewa={pred_ewa or 0.0:.4f} | ebm={ebm_status} | "
                f"s2_envs={s2_count} | {time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_run27(mpc, mpc_s2, key_buf, door_buf, device,
                             seed_offset=9000 + env_step, n_eps=EVAL_N_EPS)
            elapsed = time.time() - t0
            logger.info(
                f"[RUN27] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | pred_ewa={pred_ewa or 0.0:.4f} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(pred_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({"key":len(key_buf),"door":len(door_buf),"goal":len(goal_buf),"post_neg":len(post_door_neg_buf)})
            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[RUN27] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0
    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "predictor": predictor.state_dict(),
                "ebm": ebm.state_dict()},
               ckpt_dir / f"checkpoint_{condition}.pt")
    logger.info(f"[RUN27] Done | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"], "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"], "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps, "observe_steps": total_observe_steps,
    }


run_abm_loop = run_doorkey_run27_loop
