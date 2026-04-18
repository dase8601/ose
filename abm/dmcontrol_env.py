"""
abm/dmcontrol_env.py — Gymnasium wrapper for DeepMind Control Suite tasks.

Install: pip install dm_control

dm_control provides physics-simulated continuous control tasks with
camera-rendered RGB observations. This is the same environment suite used
by DINO-WM (Yann's team): frozen DINOv2 + world model + MPC.

Our contribution: autonomous OBSERVE/ACT switching (System M) vs DINO-WM's
fixed observation budget.

Observations: {"image": (H, W, 3) uint8, "rgb": (H, W, 3) uint8}
Actions: Discrete(N_ACTIONS) — discretized from continuous action space

Discretization: the continuous action space is mapped to a fixed set of
action primitives (vertices + midpoints of the action hypercube). This
keeps our MPC architecture (random shooting with discrete actions) intact.
"""

import numpy as np

try:
    import gymnasium
except ImportError:
    import gym as gymnasium

from dm_control import suite


# Pre-computed action primitives for each task
def _build_action_set(action_spec, n_bins=3):
    """Build discrete action set from continuous spec.

    For each dimension, sample n_bins evenly spaced values in [min, max].
    Total actions = n_bins^dim, capped at 243 (3^5) to stay tractable.

    For low-dim tasks (1-2 dims): full grid.
    For high-dim tasks (6+ dims): sample representative actions.
    """
    dim = action_spec.shape[0]
    lo = action_spec.minimum
    hi = action_spec.maximum

    if dim <= 4:
        # Full grid is tractable
        import itertools
        vals_per_dim = [np.linspace(lo[d], hi[d], n_bins) for d in range(dim)]
        actions = np.array(list(itertools.product(*vals_per_dim)), dtype=np.float32)
    else:
        # For high-dim: use axis-aligned extremes + zero + random samples
        actions = [np.zeros(dim, dtype=np.float32)]  # neutral

        # Each dimension at min and max (other dims at 0)
        for d in range(dim):
            for val in [lo[d], hi[d]]:
                a = np.zeros(dim, dtype=np.float32)
                a[d] = val
                actions.append(a)

        # Coordinated actions: all max, all min, alternating
        actions.append(np.array(hi, dtype=np.float32))
        actions.append(np.array(lo, dtype=np.float32))
        alt = np.zeros(dim, dtype=np.float32)
        alt[::2] = hi[::2]
        alt[1::2] = lo[1::2]
        actions.append(alt)
        alt2 = np.zeros(dim, dtype=np.float32)
        alt2[::2] = lo[::2]
        alt2[1::2] = hi[1::2]
        actions.append(alt2)

        actions = np.array(actions, dtype=np.float32)

    return actions


# Task-specific configs
TASK_CONFIGS = {
    "cartpole-swingup": {"domain": "cartpole", "task": "swingup", "n_bins": 5},
    "cartpole-balance": {"domain": "cartpole", "task": "balance", "n_bins": 5},
    "reacher-easy":     {"domain": "reacher",  "task": "easy",    "n_bins": 5},
    "reacher-hard":     {"domain": "reacher",  "task": "hard",    "n_bins": 5},
    "walker-walk":      {"domain": "walker",   "task": "walk",    "n_bins": 3},
    "walker-stand":     {"domain": "walker",   "task": "stand",   "n_bins": 3},
    "cheetah-run":      {"domain": "cheetah",  "task": "run",     "n_bins": 3},
    "finger-spin":      {"domain": "finger",   "task": "spin",    "n_bins": 3},
}


class DMControlEnv(gymnasium.Env):
    """
    Gymnasium wrapper around DeepMind Control Suite.

    Renders camera observations at configurable resolution.
    Discretizes continuous actions for compatibility with our MPC planner.

    Observation: {"image": (H, W, 3) uint8, "rgb": (H, W, 3) uint8}
    Actions: Discrete(N) — mapped to continuous action primitives
    Reward: task-specific dense reward in [0, 1]
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str = "walker-walk",
        seed: int = 0,
        img_size: int = 84,
        camera_id: int = 0,
    ):
        super().__init__()

        cfg = TASK_CONFIGS[task_name]
        self._env = suite.load(
            cfg["domain"], cfg["task"],
            task_kwargs={"random": seed},
        )
        self._img_size = img_size
        self._camera_id = camera_id
        self._seed = seed

        # Build discrete action set
        action_spec = self._env.action_spec()
        self._action_set = _build_action_set(action_spec, n_bins=cfg["n_bins"])
        self.N_ACTIONS = len(self._action_set)

        self.observation_space = gymnasium.spaces.Dict({
            "image": gymnasium.spaces.Box(
                low=0, high=255,
                shape=(img_size, img_size, 3),
                dtype=np.uint8,
            ),
            "rgb": gymnasium.spaces.Box(
                low=0, high=255,
                shape=(img_size, img_size, 3),
                dtype=np.uint8,
            ),
        })
        self.action_space = gymnasium.spaces.Discrete(self.N_ACTIONS)

    def _render_obs(self):
        """Render camera observation from physics sim."""
        pixels = self._env.physics.render(
            height=self._img_size,
            width=self._img_size,
            camera_id=self._camera_id,
        )
        return {"image": pixels, "rgb": pixels}

    def reset(self, seed=None, **kwargs):
        self._env.reset()
        return self._render_obs(), {}

    def step(self, action: int):
        continuous_action = self._action_set[action]
        time_step = self._env.step(continuous_action)

        obs = self._render_obs()
        reward = float(time_step.reward or 0.0)
        terminated = time_step.last()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        return self._env.physics.render(
            height=self._img_size,
            width=self._img_size,
            camera_id=self._camera_id,
        )

    def close(self):
        pass


def make_dmcontrol_env(
    task_name: str = "walker-walk",
    seed: int = 0,
    img_size: int = 84,
) -> DMControlEnv:
    return DMControlEnv(task_name=task_name, seed=seed, img_size=img_size)


def make_dmcontrol_vec_env(
    n_envs: int,
    task_name: str = "walker-walk",
    seed: int = 0,
    use_async: bool = True,
    img_size: int = 84,
):
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    fns = [
        lambda i=i: make_dmcontrol_env(task_name=task_name, seed=seed + i, img_size=img_size)
        for i in range(n_envs)
    ]
    if use_async:
        return AsyncVectorEnv(fns, shared_memory=False)
    return SyncVectorEnv(fns)
