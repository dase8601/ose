"""
abm/miniworld_env.py — Gymnasium wrapper for MiniWorld 3D navigation.

Install: pip install miniworld

MiniWorld provides 3D first-person navigation in procedurally generated
indoor environments (mazes, rooms, hallways). Ego-centric RGB observations
with discrete movement actions.

Observations: {"image": (H, W, 3) uint8}  — uses "image" key for CNN encoder
              {"rgb":   (H, W, 3) uint8}  — alias for V-JEPA encoder
Actions: Discrete(3) — turn_left, turn_right, move_forward

The visual complexity (3D rendered rooms with textures, perspective, lighting)
is significantly higher than Crafter's 2D sprites, making it a better test
for whether V-JEPA abstract representations outperform raw pixels.
"""

import io
import os
import sys
import contextlib
import logging
import numpy as np

try:
    import gymnasium
except ImportError:
    import gym as gymnasium

# Suppress MiniWorld/pyglet noisy OpenGL messages
logging.getLogger("miniworld").setLevel(logging.ERROR)
logging.getLogger("pyglet").setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout to hide pyglet's 'Falling back to num_samples' spam."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


class MiniWorldNavEnv(gymnasium.Env):
    """
    Gymnasium wrapper around MiniWorld maze navigation.

    Uses MiniWorld-MazeS3-v0 by default (3x3 maze — solvable but non-trivial).
    Renders at a configurable resolution for V-JEPA compatibility.

    Observation: {"image": (H, W, 3) uint8, "rgb": (H, W, 3) uint8}
    Actions: 0=turn_left, 1=turn_right, 2=move_forward
    Reward: +1.0 on reaching the goal box, small time penalty per step
    """

    metadata = {"render_modes": ["rgb_array"]}
    N_ACTIONS = 3

    def __init__(
        self,
        env_id: str = "MiniWorld-MazeS3-v0",
        seed: int = 0,
        img_size: int = 160,
        max_steps: int = 500,
    ):
        super().__init__()
        import miniworld  # registers envs

        self._seed = seed
        self._img_size = img_size

        # MiniWorld/pyglet prints "Falling back to num_samples=4" on every env creation
        with _suppress_stdout():
            self._env = gymnasium.make(
                env_id,
                render_mode="rgb_array",
                view="agent",
                max_episode_steps=max_steps,
            )

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

    def _process_obs(self, obs):
        """Resize observation to target size and wrap in dict."""
        if isinstance(obs, dict):
            img = obs.get("image", obs.get("obs", None))
            if img is None:
                img = list(obs.values())[0]
        else:
            img = obs

        # Resize if needed
        if img.shape[0] != self._img_size or img.shape[1] != self._img_size:
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((self._img_size, self._img_size), Image.BILINEAR)
            img = np.array(pil_img)

        return {"image": img, "rgb": img}

    def reset(self, seed=None, **kwargs):
        obs, info = self._env.reset(seed=seed or self._seed)
        return self._process_obs(obs), info

    def step(self, action: int):
        obs, reward, term, trunc, info = self._env.step(int(action))
        return self._process_obs(obs), float(reward), bool(term), bool(trunc), info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Factory helpers (mirrors crafter_env.py pattern)
# ---------------------------------------------------------------------------

def make_miniworld_env(seed: int = 0, img_size: int = 160) -> MiniWorldNavEnv:
    """Create a single MiniWorld maze environment."""
    return MiniWorldNavEnv(seed=seed, img_size=img_size)


def make_miniworld_vec_env(
    n_envs: int,
    seed: int = 0,
    use_async: bool = True,
    img_size: int = 160,
):
    """Create vectorized MiniWorld environments."""
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    fns = [
        lambda i=i: make_miniworld_env(seed=seed + i, img_size=img_size)
        for i in range(n_envs)
    ]
    if use_async:
        return AsyncVectorEnv(fns, shared_memory=False)
    return SyncVectorEnv(fns)
