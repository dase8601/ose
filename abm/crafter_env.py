"""
abm/crafter_env.py — Gymnasium wrapper for Crafter.

Install: pip install crafter

Crafter is a 2D survival environment with 22 achievements organized in a
tech tree. Each first-unlock of an achievement gives +1 reward (sparse).
Observations: (64, 64, 3) RGB uint8.  Actions: 17 discrete.

Standard eval metric used here: fraction of the 22 achievements unlocked
at least once across N evaluation episodes.

Tier groupings are used in abm_experiment.py to show whether Autonomous
System M correctly front-loads OBSERVE time for harder achievement tiers.
"""

import numpy as np
import gymnasium
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


# All 22 Crafter achievements (alphabetical)
ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]

# Tech-tree tiers — used to measure whether OBSERVE time is front-loaded
# for harder tiers (the key claim for autonomous System M on Crafter)
ACHIEVEMENT_TIERS = {
    "tier1_basic":    ["collect_wood", "collect_sapling", "collect_drink",
                       "collect_stone", "place_stone", "wake_up"],
    "tier2_tools":    ["place_table", "place_plant", "collect_coal",
                       "make_wood_pickaxe", "make_wood_sword"],
    "tier3_advanced": ["collect_iron", "make_stone_pickaxe", "make_stone_sword",
                       "place_furnace", "eat_plant", "eat_cow"],
    "tier4_hard":     ["collect_diamond", "defeat_zombie", "defeat_skeleton",
                       "make_iron_pickaxe", "make_iron_sword"],
}


class CrafterEnv(gymnasium.Env):
    """
    Thin gymnasium wrapper around crafter.Env.

    Observation dict: {"image": (64, 64, 3) uint8}
    — matches the MiniGrid RGBImgObsWrapper format so batch_obs_to_tensor
    and the CNN encoder work without modification.

    Step returns the gymnasium 5-tuple; crafter's old gym API (4-tuple) is
    converted internally.  Achievement progress lives in info["achievements"].
    """

    metadata  = {"render_modes": ["rgb_array"]}
    IMG_H     = 64
    IMG_W     = 64
    N_ACTIONS = 17

    def __init__(self, seed: int = None):
        super().__init__()
        import crafter as _crafter
        self._crafter_mod = _crafter
        self._seed        = seed
        self._env         = _crafter.Env(seed=seed)
        self.observation_space = gymnasium.spaces.Dict({
            "image": gymnasium.spaces.Box(
                low=0, high=255,
                shape=(self.IMG_H, self.IMG_W, 3),
                dtype=np.uint8,
            )
        })
        self.action_space = gymnasium.spaces.Discrete(self.N_ACTIONS)

    def reset(self, seed: int = None, **kwargs):
        if seed is not None and seed != self._seed:
            self._env  = self._crafter_mod.Env(seed=seed)
            self._seed = seed
        obs = self._env.reset()
        return {"image": obs}, {}

    def step(self, action: int):
        obs, reward, done, info = self._env.step(int(action))
        # crafter uses old gym 4-tuple; convert to gymnasium 5-tuple
        return {"image": obs}, float(reward), bool(done), False, info

    def render(self):
        return self._env.render()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Factory helpers (mirrors make_env / make_vec_env in loop.py)
# ---------------------------------------------------------------------------

def make_crafter_env(seed: int = 0) -> CrafterEnv:
    return CrafterEnv(seed=seed)


def make_crafter_vec_env(
    n_envs:    int,
    seed:      int  = 0,
    use_async: bool = True,
):
    fns = [lambda i=i: CrafterEnv(seed=seed + i) for i in range(n_envs)]
    if use_async:
        return AsyncVectorEnv(fns, shared_memory=False)
    return SyncVectorEnv(fns)
