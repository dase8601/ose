"""
abm/loop_mpc_doorkey_run11.py — Run 11: Post-door negative buffer for stage-2 EBM

Diagnosis from Run 10:
  Protected seed_buf fixed the WM OOD problem — door opened 5x in 60k ACT
  steps (better rate than Run 9). But goal stayed frozen at 200.

  Root cause: stage-2 EBM energy landscape is wrong in the right half.
  EBM negatives come exclusively from buf_lew, which is ~99% left-half
  states (OBSERVE curiosity + left-half ACT transitions).
  When the agent enters the right half (post-door), the EBM has never seen
  right-half non-exit states as negatives. Energy landscape there is either:
    (a) flat — CEM gets no gradient, wanders randomly, rarely hits exit
    (b) inverted — right-half states have LOWER energy than left-half
        (they look more like exit images than left-half does), so CEM
        plans BACK through the door to the "lower energy" left side

  In either case, the 5 brief stage-2 windows (10–30 steps each) never
  produce a goal completion.

Fix: post_door_neg_buf — a GoalImageBuffer of right-half, non-exit frames.
  - Seeder collects every frame after door opens until the exit is reached:
    ~4-8 frames × 200 episodes = ~1000 right-half non-exit images
  - Added as a 4th EBM training signal: goal_buf positives vs
    post_door_neg_buf negatives → teaches E(z_exit, z_goal) < E(z_right_non_exit, z_goal)
  - During ACT, every door-open non-success frame is also added online
  - Everything else from Run 10 unchanged (protected seed_buf, H=8, subgoals,
    curiosity, HER)

Condition: post_door_neg
Loop module: abm.loop_mpc_doorkey_run11
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
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import RGBImgObsWrapper

from .world_model import LeWM, ReplayBuffer, SequenceReplayBuffer
from .cem_planner import CEMPlanner, EBMCostHead

logger = logging.getLogger(__name__)

N_ENVS                  = 16
IMG_H = IMG_W           = 48
N_ACTIONS               = 7
LATENT_DIM              = 256
LEWM_LR                 = 1e-4
LEWM_BATCH              = 256
LEWM_WARMUP             = 500
REPLAY_CAPACITY         = 100_000
SEED_BUF_CAPACITY       = 20_000
POST_DOOR_NEG_CAPACITY  = 5_000
GOAL_CAPACITY           = 1_024
HER_CAPACITY            = 4_096
EVAL_INTERVAL           = 5_000
EVAL_N_EPS              = 10
GOAL_REFRESH_STEPS      = 64
N_TRAIN_STEPS           = 4
DEFAULT_OBSERVE         = 80_000
EP_MAX_STEPS            = 300
EBM_MIN_GOALS           = 5
EBM_HER_MIN             = 20
EBM_POST_DOOR_MIN       = 20
EBM_WARMUP_STEPS        = 500
EBM_LR                  = 3e-4
EBM_BATCH               = 32
N_SEED_EPS              = 200

CEM_HORIZON             = 8
CEM_SAMPLES             = 512
CEM_ELITES              = 64
CEM_ITERS               = 5


# ── Environment helpers ────────────────────────────────────────────────────

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


def _is_door_open(uw) -> bool:
    grid = uw.grid
    for col in range(grid.width):
        for row in range(grid.height):
            cell = grid.get(col, row)
            if cell is not None and cell.type == "door" and cell.is_open:
                return True
    return False


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


# ── BFS scripted policy ────────────────────────────────────────────────────

_DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
_VEC_TO_DIR = {(1,0):0, (0,1):1, (-1,0):2, (0,-1):3}


def _find_cell(uw, cell_type: str) -> Optional[Tuple[int, int]]:
    for row in range(uw.height):
        for col in range(uw.width):
            cell = uw.grid.get(col, row)
            if cell is not None and cell.type == cell_type:
                return (col, row)
    return None


def _bfs(uw, start: Tuple[int,int], goal: Tuple[int,int],
         allow_open_door: bool = False) -> List[Tuple[int,int]]:
    if start == goal:
        return [start]

    def passable(col: int, row: int) -> bool:
        if not (0 <= col < uw.width and 0 <= row < uw.height):
            return False
        cell = uw.grid.get(col, row)
        if cell is None:
            return True
        if cell.type == "wall":
            return False
        if cell.type == "door":
            return allow_open_door and cell.is_open
        return True

    queue: deque = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        col, row = path[-1]
        for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
            nc, nr = col + dc, row + dr
            nxt = (nc, nr)
            if nxt == goal:
                return path + [nxt]
            if nxt not in visited and passable(nc, nr):
                visited.add(nxt)
                queue.append(path + [nxt])
    return []


def _turn_toward(current_dir: int, target_dir: int) -> int:
    diff = (target_dir - current_dir) % 4
    return 1 if diff in (1, 2) else 0


def _step_toward(pos: Tuple[int,int], agent_dir: int,
                 next_pos: Tuple[int,int]) -> int:
    dc = next_pos[0] - pos[0]
    dr = next_pos[1] - pos[1]
    target_dir = _VEC_TO_DIR[(dc, dr)]
    if agent_dir == target_dir:
        return 2
    return _turn_toward(agent_dir, target_dir)


def _approach_and_interact(uw, pos, agent_dir, target_pos, interact_action: int) -> int:
    fwd = (pos[0] + _DIR_TO_VEC[agent_dir][0],
           pos[1] + _DIR_TO_VEC[agent_dir][1])
    if fwd == target_pos:
        return interact_action

    for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        adj = (target_pos[0]+dc, target_pos[1]+dr)
        if pos == adj:
            tgt_dir = _VEC_TO_DIR[(target_pos[0]-pos[0], target_pos[1]-pos[1])]
            if agent_dir == tgt_dir:
                return interact_action
            return _turn_toward(agent_dir, tgt_dir)

    best: Optional[List] = None
    for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
        adj = (target_pos[0]+dc, target_pos[1]+dr)
        if 0 <= adj[0] < uw.width and 0 <= adj[1] < uw.height:
            cell = uw.grid.get(adj[0], adj[1])
            if cell is None or cell.type in ("key",):
                path = _bfs(uw, pos, adj, allow_open_door=False)
                if path and (best is None or len(path) < len(best)):
                    best = path

    if best and len(best) >= 2:
        return _step_toward(pos, agent_dir, best[1])
    return -1


def _scripted_action(uw, fallback_fn) -> int:
    pos = tuple(map(int, uw.agent_pos))
    d   = int(uw.agent_dir)

    if uw.carrying is None:
        key_pos = _find_cell(uw, "key")
        if key_pos is None:
            return fallback_fn()
        action = _approach_and_interact(uw, pos, d, key_pos, interact_action=3)
        return action if action >= 0 else fallback_fn()

    door_pos = _find_cell(uw, "door")
    if door_pos is not None:
        door_cell = uw.grid.get(door_pos[0], door_pos[1])
        if not door_cell.is_open:
            action = _approach_and_interact(uw, pos, d, door_pos, interact_action=5)
            return action if action >= 0 else fallback_fn()

    goal_pos = _find_cell(uw, "goal")
    if goal_pos is None:
        return fallback_fn()

    fwd = (pos[0] + _DIR_TO_VEC[d][0], pos[1] + _DIR_TO_VEC[d][1])
    if fwd == goal_pos:
        return 2

    path = _bfs(uw, pos, goal_pos, allow_open_door=True)
    if path and len(path) >= 2:
        return _step_toward(pos, d, path[1])
    return fallback_fn()


# ── Scripted seeder — populates seed_buf AND post_door_neg_buf ─────────────

def _seed_scripted(
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    seed_buf: ReplayBuffer,
    post_door_neg_buf: GoalImageBuffer,   # NEW: right-half non-exit frames
    n_eps: int = N_SEED_EPS,
    seed: int = 7777,
) -> None:
    """
    BFS seeder. Populates:
      - key_buf / door_buf / goal_buf: EBM goal images (200+ each)
      - seed_buf: (obs, action, next_obs) transitions — full grid coverage
      - post_door_neg_buf: every frame between door-open and exit — teaches
        the stage-2 EBM what "in right half but not at exit" looks like
    """
    logger.info(f"[RUN11] Scripted seeding: {n_eps} episodes…")
    successes = 0
    total_transitions = 0

    for ep in range(n_eps):
        env = _make_doorkey_env(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        uw = env.unwrapped
        done = False
        steps = 0
        had_key = False
        door_was_open = False
        collecting_post_door = False   # True between door-open and exit

        while not done and steps < EP_MAX_STEPS:
            action = _scripted_action(uw, fallback_fn=env.action_space.sample)
            prev_img = _resize_imgs(obs["image"])[0]

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1
            uw = env.unwrapped

            img = _resize_imgs(obs["image"])[0]

            seed_buf.push(prev_img, action, img)
            total_transitions += 1

            if not had_key and uw.carrying is not None:
                key_buf.push(img)
                had_key = True

            if not door_was_open and _is_door_open(uw):
                door_buf.push(img)
                door_was_open = True
                collecting_post_door = True

            if r > 0:
                goal_buf.push(img)
                collecting_post_door = False
                successes += 1
            elif collecting_post_door:
                # Right half, door open, not at exit — perfect stage-2 negative
                post_door_neg_buf.push(img)

        env.close()

    logger.info(
        f"[RUN11] Seed done: {successes}/{n_eps} succeeded | "
        f"transitions={total_transitions} | seed_buf={len(seed_buf)} | "
        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
        f"post_door_neg={len(post_door_neg_buf)}"
    )


# ── World model training with protected seed mix ───────────────────────────

def _train_lewm_mixed(
    lewm: LeWM,
    opt_lewm: optim.Optimizer,
    buf_lew: ReplayBuffer,
    seed_buf: ReplayBuffer,
    device: str,
    n_steps: int = N_TRAIN_STEPS,
) -> Optional[float]:
    if len(buf_lew) < LEWM_WARMUP:
        return None

    last_loss = None
    n_seed = LEWM_BATCH // 2
    n_online = LEWM_BATCH - n_seed
    use_mixed = len(seed_buf) >= n_seed

    for _ in range(n_steps):
        if use_mixed:
            obs_s, acts_s, nxt_s = seed_buf.sample(n_seed, device)
            obs_o, acts_o, nxt_o = buf_lew.sample(n_online, device)
            obs_t  = torch.cat([obs_s, obs_o], 0)
            acts_t = torch.cat([acts_s, acts_o], 0)
            nxt_t  = torch.cat([nxt_s, nxt_o], 0)
        else:
            obs_t, acts_t, nxt_t = buf_lew.sample(LEWM_BATCH, device)

        opt_lewm.zero_grad()
        loss, info = lewm.loss(obs_t, acts_t, nxt_t)
        loss.backward()
        opt_lewm.step()
        last_loss = info["loss_total"]

    return last_loss


# ── Curiosity helpers ──────────────────────────────────────────────────────

def _curiosity_actions(lewm, obs, running_z_mean: torch.Tensor, device: str) -> np.ndarray:
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


# ── EBM training ──────────────────────────────────────────────────────────

def _imgs_to_z(imgs: np.ndarray, lewm: LeWM, device: str) -> torch.Tensor:
    t = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    return lewm.encode(t)


def _train_ebm(
    ebm: EBMCostHead,
    opt_ebm: optim.Optimizer,
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    her_buf: GoalImageBuffer,
    post_door_neg_buf: GoalImageBuffer,
    buf_lew: ReplayBuffer,
    lewm: LeWM,
    device: str,
) -> bool:
    if len(buf_lew) < EBM_BATCH * 2:
        return False

    opt_ebm.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    # Standard signal: random subgoal buffer vs left-half negatives
    ready = [(b, n) for b, n in [(key_buf, "key"), (door_buf, "door"), (goal_buf, "goal")]
             if len(b) >= EBM_MIN_GOALS]
    if ready:
        buf, _ = ready[np.random.randint(len(ready))]
        pos_imgs  = buf.sample(EBM_BATCH)
        goal_imgs = buf.sample(EBM_BATCH)
        neg_obs, _, _ = buf_lew.sample(EBM_BATCH, device)
        with torch.no_grad():
            z_pos = _imgs_to_z(pos_imgs, lewm, device)
            z_g   = _imgs_to_z(goal_imgs, lewm, device)
            z_neg = lewm.encode(neg_obs)
        total_loss = total_loss + ebm.contrastive_loss(z_pos, z_neg, z_g)
        n_terms += 1

    # HER signal: terminal states are their own goals
    if len(her_buf) >= EBM_HER_MIN:
        her_imgs   = her_buf.sample(EBM_BATCH)
        neg_obs2, _, _ = buf_lew.sample(EBM_BATCH, device)
        with torch.no_grad():
            z_her  = _imgs_to_z(her_imgs, lewm, device)
            z_neg2 = lewm.encode(neg_obs2)
        total_loss = total_loss + ebm.contrastive_loss(z_her, z_neg2, z_her.detach())
        n_terms += 1

    # Stage-2 specific signal: exit vs right-half non-exit states
    # Teaches the EBM the energy landscape INSIDE the right half of the grid
    if len(goal_buf) >= EBM_MIN_GOALS and len(post_door_neg_buf) >= EBM_POST_DOOR_MIN:
        pos_imgs2  = goal_buf.sample(EBM_BATCH)
        goal_imgs2 = goal_buf.sample(EBM_BATCH)
        neg_imgs2  = post_door_neg_buf.sample(EBM_BATCH)
        with torch.no_grad():
            z_pos2 = _imgs_to_z(pos_imgs2, lewm, device)
            z_g2   = _imgs_to_z(goal_imgs2, lewm, device)
            z_neg3 = _imgs_to_z(neg_imgs2, lewm, device)
        total_loss = total_loss + ebm.contrastive_loss(z_pos2, z_neg3, z_g2)
        n_terms += 1

    if n_terms > 0:
        total_loss.backward()
        opt_ebm.step()
    return n_terms > 0


# ── Eval ──────────────────────────────────────────────────────────────────

def _eval_run11(
    mpc: Optional[CEMPlanner],
    encoder_fn,
    key_buf: GoalImageBuffer,
    door_buf: GoalImageBuffer,
    goal_buf: GoalImageBuffer,
    device: str,
    seed_offset: int = 1000,
    n_eps: int = EVAL_N_EPS,
) -> float:
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

        def _pick_goal(stage: int):
            return (key_buf if stage == 0 else door_buf if stage == 1 else goal_buf).sample(1)

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
            action = (mpc.plan_single(z, z_goal)
                      if mpc is not None and z_goal is not None
                      else env.action_space.sample())

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r
            ep_steps += 1
            goal_age += 1

        if ep_ret > 0.5:
            successes += 1
        env.close()

    return successes / n_eps


# ── Main loop ──────────────────────────────────────────────────────────────

def run_doorkey_run11_loop(
    condition: str = "post_door_neg",
    device: str = "cuda",
    max_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = N_ENVS,
    env_type: str = "doorkey",
    observe_steps: Optional[int] = None,
    use_mpc: bool = True,
    use_rl: bool = False,
) -> Dict:
    if condition != "post_door_neg":
        raise ValueError(f"loop_mpc_doorkey_run11 supports: post_door_neg — got: {condition}")
    if env_type != "doorkey":
        raise ValueError("loop_mpc_doorkey_run11.py only supports env_type='doorkey'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    _observe_steps = observe_steps if observe_steps is not None else DEFAULT_OBSERVE
    logger.info(
        f"[RUN11] Post-door neg buffer + protected seed + H={CEM_HORIZON} | "
        f"device={device} max_steps={max_steps} n_envs={n_envs} observe={_observe_steps}"
    )
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    key_buf           = GoalImageBuffer(capacity=GOAL_CAPACITY)
    door_buf          = GoalImageBuffer(capacity=GOAL_CAPACITY)
    goal_buf          = GoalImageBuffer(capacity=GOAL_CAPACITY)
    her_buf           = GoalImageBuffer(capacity=HER_CAPACITY)
    post_door_neg_buf = GoalImageBuffer(capacity=POST_DOOR_NEG_CAPACITY)
    seed_buf          = ReplayBuffer(capacity=SEED_BUF_CAPACITY)

    _seed_scripted(
        key_buf, door_buf, goal_buf, seed_buf, post_door_neg_buf,
        n_eps=N_SEED_EPS, seed=seed + 999,
    )

    envs = _make_doorkey_vec_env(n_envs, seed=seed)
    obs, _ = envs.reset(seed=list(range(seed, seed + n_envs)))

    lewm    = LeWM(latent_dim=LATENT_DIM, n_actions=N_ACTIONS, img_size=IMG_H, predictor_type="mlp").to(device)
    buf_lew = ReplayBuffer(capacity=REPLAY_CAPACITY)
    buf_seq = SequenceReplayBuffer(capacity=REPLAY_CAPACITY, seq_len=8)
    opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)
    mpc: Optional[CEMPlanner] = None

    ebm        = EBMCostHead(latent_dim=LATENT_DIM).to(device)
    opt_ebm    = optim.Adam(ebm.parameters(), lr=EBM_LR)
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
    ep_ret          = np.zeros(n_envs, dtype=np.float32)

    metrics: Dict = {
        "env_step": [], "success_rate": [], "ssl_loss_ewa": [],
        "mode": [], "wall_time_s": [], "per_tier": [],
    }
    ssl_ewa      = None
    ssl_loss_val = None
    steps_to_80  = None
    env_step     = 0
    act_steps    = 0
    total_observe_steps = 0
    mode_str     = "OBSERVE"

    while env_step < max_steps:

        in_observe = (env_step < _observe_steps)

        if in_observe:
            mode_str = "OBSERVE"

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
                elif _is_door_open(uw):
                    # Live post-door non-exit frame: refine stage-2 EBM negatives online
                    post_door_neg_buf.push(next_imgs[i])
                if dones[i]:
                    her_buf.push(next_imgs[i])
                    had_key[i] = False
                    door_open_flags[i] = False

            obs = next_obs
            env_step += n_envs
            total_observe_steps += n_envs

            ssl_loss_val = _train_lewm_mixed(lewm, opt_lewm, buf_lew, seed_buf, device)
            if ssl_loss_val is not None:
                ssl_ewa = ssl_loss_val if ssl_ewa is None else 0.95 * ssl_ewa + 0.05 * ssl_loss_val

                if len(buf_seq) > buf_seq.seq_len * 2:
                    seq_data = buf_seq.sample_sequences(
                        batch_size=64, encoder_fn=lewm.encode,
                        n_actions=N_ACTIONS, device=device,
                    )
                    if seq_data is not None:
                        z_seq, a_oh_seq, z_next_t = seq_data
                        opt_lewm.zero_grad()
                        z_pred = lewm.predictor.forward_sequence(z_seq, a_oh_seq)
                        seq_loss = (1 - F.cosine_similarity(z_pred[:, -1], z_next_t.detach(), dim=-1)).mean()
                        seq_loss.backward()
                        opt_lewm.step()

                if _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf,
                              post_door_neg_buf, buf_lew, lewm, device):
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active and mpc is not None:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(
                            f"[RUN11] EBM activated | "
                            f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
                            f"post_neg={len(post_door_neg_buf)}"
                        )

                if mpc is None:
                    mpc = CEMPlanner(
                        lewm.predictor,
                        n_actions=N_ACTIONS,
                        horizon=CEM_HORIZON,
                        n_samples=CEM_SAMPLES,
                        n_elites=CEM_ELITES,
                        n_iters=CEM_ITERS,
                        device=device,
                        distance="cosine",
                    )
                    logger.info(
                        f"[RUN11] CEM ready (H={CEM_HORIZON}) | "
                        f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)}"
                    )

        else:
            mode_str = "ACT"

            for i in range(n_envs):
                uw = envs.envs[i].unwrapped
                has_key  = uw.carrying is not None
                d_open   = _is_door_open(uw)
                new_stage = 0 if not has_key else (1 if not d_open else 2)

                if (new_stage != active_stage[i]
                        or active_goal_imgs[i] is None
                        or goal_ages[i] >= GOAL_REFRESH_STEPS):
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
                            active_goal_imgs
                            if all(img is not None for img in active_goal_imgs)
                            else [valid[0]] * n_envs,
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
                elif _is_door_open(uw):
                    # Online post-door non-exit frames — refine stage-2 EBM continuously
                    post_door_neg_buf.push(next_imgs[i])
                if dones[i]:
                    her_buf.push(next_imgs[i])
                    had_key[i] = False
                    door_open_flags[i] = False
                    ep_ret[i] = 0.0
                    active_goal_imgs[i] = None
                    goal_ages[i] = 0
                else:
                    goal_ages[i] += 1

            obs = next_obs

            ssl_loss_val = _train_lewm_mixed(lewm, opt_lewm, buf_lew, seed_buf, device)
            if ssl_loss_val is not None:
                ssl_ewa = 0.95 * ssl_ewa + 0.05 * ssl_loss_val

                if _train_ebm(ebm, opt_ebm, key_buf, door_buf, goal_buf, her_buf,
                              post_door_neg_buf, buf_lew, lewm, device):
                    ebm_train_count += 1
                    if ebm_train_count >= EBM_WARMUP_STEPS and not ebm_active:
                        mpc.set_ebm(ebm)
                        ebm_active = True
                        logger.info(f"[RUN11] EBM activated in ACT")

        if (env_step // n_envs) % 1000 == 0 and env_step > 0:
            logger.info(
                f"[RUN11] heartbeat step={env_step} | buf={len(buf_lew)} seed_buf={len(seed_buf)} | "
                f"key={len(key_buf)} door={len(door_buf)} goal={len(goal_buf)} "
                f"post_neg={len(post_door_neg_buf)} her={len(her_buf)} | "
                f"ssl={ssl_ewa or 0.0:.4f} | H={CEM_HORIZON} | "
                f"ebm={'ON' if ebm_active else f'training({ebm_train_count})'} | "
                f"{time.time()-t0:.0f}s"
            )

        if env_step % EVAL_INTERVAL < n_envs:
            sr = _eval_run11(
                mpc=mpc, encoder_fn=encoder_fn,
                key_buf=key_buf, door_buf=door_buf, goal_buf=goal_buf,
                device=device, seed_offset=9000 + env_step, n_eps=EVAL_N_EPS,
            )
            elapsed = time.time() - t0
            logger.info(
                f"[RUN11] step={env_step:7d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"H={CEM_HORIZON} | key={len(key_buf)} door={len(door_buf)} "
                f"goal={len(goal_buf)} post_neg={len(post_door_neg_buf)} | {elapsed:.0f}s"
            )
            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)
            metrics["per_tier"].append({
                "key": len(key_buf), "door": len(door_buf),
                "goal": len(goal_buf), "post_neg": len(post_door_neg_buf),
            })
            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[RUN11] *** 80% at step {env_step} ***")

    envs.close()
    elapsed_total = time.time() - t0
    ckpt_dir = Path("results/doorkey_mpc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"condition": condition, "lewm": lewm.state_dict()},
        ckpt_dir / f"checkpoint_{condition}.pt",
    )
    logger.info(f"[RUN11] Done | total_time={elapsed_total:.0f}s")

    return {
        "condition": condition, "env_type": "doorkey",
        "env_steps": metrics["env_step"], "success_rate": metrics["success_rate"],
        "ssl_loss_ewa": metrics["ssl_loss_ewa"], "mode": metrics["mode"],
        "wall_time_s": metrics["wall_time_s"], "per_tier": metrics["per_tier"],
        "steps_to_80pct": steps_to_80, "n_switches": 0, "switch_log": [],
        "total_time_s": elapsed_total, "act_steps": act_steps,
        "observe_steps": total_observe_steps,
    }


run_abm_loop = run_doorkey_run11_loop
