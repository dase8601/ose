"""
abm/habitat_env.py — Gymnasium wrapper for Habitat PointNav.

Habitat provides photorealistic indoor navigation using Matterport3D / HM3D
scene datasets.  The agent sees ego-centric RGB (384x384) and must navigate
to a target coordinate.

This wrapper presents a simple Gymnasium interface matching our existing
CrafterEnv / DoorKey patterns:
  obs_dict = {"rgb": (H, W, 3) uint8}
  action_space = Discrete(4)  — STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
  reward = distance reduction + success bonus

Install:
    pip install habitat-sim habitat-lab

Scene data:
    python -m habitat_sim.utils.datasets_download --uids hm3d_minival_v0.2
"""

import numpy as np

try:
    import gymnasium
except ImportError:
    import gym as gymnasium

# Habitat imports are deferred to avoid hard dependency when not using habitat


class HabitatPointNavEnv(gymnasium.Env):
    """
    Gymnasium wrapper around Habitat PointNav.

    Observation: {"rgb": (H, W, 3) uint8}
    Actions: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
    Reward: geodesic distance reduction + 2.5 success bonus
    """

    metadata = {"render_modes": ["rgb_array"]}
    N_ACTIONS = 4
    IMG_H = 384
    IMG_W = 384

    def __init__(self, scene_dataset: str = "hm3d", seed: int = 0):
        super().__init__()
        import habitat
        from habitat.config.default_structured_configs import (
            HabitatConfigPlugin,
            TaskConfig,
        )
        from omegaconf import OmegaConf

        self._seed = seed

        # Build Habitat config for PointNav
        config = habitat.get_config(
            "benchmark/nav/pointnav/pointnav_hm3d.yaml",
            overrides=[
                f"habitat.seed={seed}",
                f"habitat.simulator.habitat_sim_v0.gpu_device_id=0",
                f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height={self.IMG_H}",
                f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width={self.IMG_W}",
                # Use HM3D minival split for fast iteration
                "habitat.dataset.split=val",
            ],
        )

        self._env = habitat.Env(config=config)

        self.observation_space = gymnasium.spaces.Dict({
            "rgb": gymnasium.spaces.Box(
                low=0, high=255,
                shape=(self.IMG_H, self.IMG_W, 3),
                dtype=np.uint8,
            ),
        })
        self.action_space = gymnasium.spaces.Discrete(self.N_ACTIONS)

    def _extract_obs(self, hab_obs):
        """Extract RGB from Habitat observation dict."""
        rgb = hab_obs["rgb"]  # (H, W, 3) uint8
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        return {"rgb": rgb, "image": rgb}

    def reset(self, seed=None, **kwargs):
        hab_obs = self._env.reset()
        return self._extract_obs(hab_obs), {}

    def step(self, action: int):
        # Habitat action mapping: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
        hab_obs = self._env.step(action)

        # Compute reward: distance reduction + success bonus
        metrics = self._env.get_metrics()
        reward = metrics.get("distance_to_goal_reward", 0.0)
        success = metrics.get("success", 0.0)
        reward += 2.5 * success

        done = self._env.episode_over
        info = {
            "success": success,
            "spl": metrics.get("spl", 0.0),
            "distance_to_goal": metrics.get("distance_to_goal", float("inf")),
        }

        return self._extract_obs(hab_obs), float(reward), bool(done), False, info

    def render(self):
        obs = self._env.render("rgb")
        return obs

    def close(self):
        self._env.close()


class HabitatPointNavSimpleEnv(gymnasium.Env):
    """
    Fallback wrapper that uses habitat-sim directly (no habitat-lab configs).
    Simpler setup, works with just habitat-sim installed.

    For when the full habitat-lab config system isn't available.
    Uses a single scene file for fast prototyping.
    """

    metadata = {"render_modes": ["rgb_array"]}
    N_ACTIONS = 4
    IMG_H = 384
    IMG_W = 384

    def __init__(self, scene_path: str = None, seed: int = 0, max_steps: int = 500):
        super().__init__()
        import glob
        import os
        import habitat_sim

        self._seed = seed
        self._max_steps = max_steps
        self._step_count = 0

        # Auto-detect scene file if not provided
        if scene_path is None:
            candidates = []
            for base in [
                "data/scene_datasets/habitat-test-scenes",
                "/workspace/jepa/data/scene_datasets/habitat-test-scenes",
                os.path.join(os.path.dirname(__file__), "..", "data", "scene_datasets", "habitat-test-scenes"),
            ]:
                candidates.extend(glob.glob(os.path.join(base, "*.glb")))
            if not candidates:
                raise FileNotFoundError(
                    "No .glb scene files found. Run:\n"
                    "  python -m habitat_sim.utils.datasets_download "
                    "--uids habitat_test_scenes --data-path data/"
                )
            scene_path = os.path.abspath(candidates[0])

        # Minimal habitat-sim config
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = os.path.abspath(scene_path)
        sim_cfg.enable_physics = False

        # RGB sensor
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [self.IMG_H, self.IMG_W]
        rgb_sensor.position = [0.0, 1.5, 0.0]  # eye height

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor]
        agent_cfg.action_space = {
            0: habitat_sim.agent.ActionSpec("stop"),
            1: habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=0.25),
            ),
            2: habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=10.0),
            ),
            3: habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=10.0),
            ),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        self._agent = self._sim.get_agent(0)

        self.observation_space = gymnasium.spaces.Dict({
            "rgb": gymnasium.spaces.Box(
                low=0, high=255,
                shape=(self.IMG_H, self.IMG_W, 3),
                dtype=np.uint8,
            ),
        })
        self.action_space = gymnasium.spaces.Discrete(self.N_ACTIONS)

        # Random goal position (set on reset)
        self._goal_position = None

    def _get_obs(self):
        obs = self._sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]  # drop alpha if present
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        return {"rgb": rgb, "image": rgb}

    def _agent_position(self):
        state = self._agent.get_state()
        return np.array(state.position)

    def reset(self, seed=None, **kwargs):
        self._sim.reset()
        self._step_count = 0

        # Sample a random navigable goal
        self._start_position = self._agent_position()
        self._goal_position = self._sim.pathfinder.get_random_navigable_point()
        self._prev_dist = np.linalg.norm(
            self._agent_position() - self._goal_position
        )

        return self._get_obs(), {}

    def step(self, action: int):
        self._step_count += 1

        if action == 0:  # STOP
            dist = np.linalg.norm(self._agent_position() - self._goal_position)
            success = float(dist < 0.2)
            reward = 2.5 * success
            return self._get_obs(), reward, True, False, {
                "success": success,
                "distance_to_goal": dist,
            }

        action_names = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
        self._agent.act(action_names[action])

        # Distance-based reward
        dist = np.linalg.norm(self._agent_position() - self._goal_position)
        reward = self._prev_dist - dist  # positive when getting closer
        self._prev_dist = dist

        done = self._step_count >= self._max_steps
        success = float(dist < 0.2)
        if success:
            reward += 2.5
            done = True

        return self._get_obs(), float(reward), done, False, {
            "success": success,
            "distance_to_goal": dist,
        }

    def get_goal_obs(self):
        """
        Teleport agent to goal position, render observation, restore state.
        Mirrors MiniWorldNavEnv.get_goal_obs() pattern.
        """
        if self._goal_position is None:
            return None

        import habitat_sim

        state = self._agent.get_state()
        saved = habitat_sim.AgentState()
        saved.position = state.position
        saved.rotation = state.rotation

        goal_state = habitat_sim.AgentState()
        goal_state.position = self._goal_position
        goal_state.rotation = state.rotation
        self._agent.set_state(goal_state)

        goal_obs = self._get_obs()

        self._agent.set_state(saved)
        return goal_obs

    def close(self):
        self._sim.close()


# ---------------------------------------------------------------------------
# Factory helpers (mirrors crafter_env.py pattern)
# ---------------------------------------------------------------------------

def make_habitat_env(seed: int = 0, scene_path: str = None, simple: bool = False):
    """Create a single Habitat PointNav environment."""
    if simple or scene_path:
        return HabitatPointNavSimpleEnv(scene_path=scene_path, seed=seed)
    return HabitatPointNavEnv(seed=seed)


def make_habitat_vec_env(
    n_envs: int,
    seed: int = 0,
    use_async: bool = False,
    simple: bool = False,
    scene_path: str = None,
):
    """
    Create vectorized Habitat environments.

    Note: use_async=False by default because habitat-sim manages its own
    multiprocessing and doesn't play well with gymnasium's AsyncVectorEnv.
    """
    from gymnasium.vector import SyncVectorEnv

    fns = [
        lambda i=i: make_habitat_env(seed=seed + i, scene_path=scene_path, simple=simple)
        for i in range(n_envs)
    ]
    return SyncVectorEnv(fns)
