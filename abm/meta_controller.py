"""
abm/meta_controller.py — System M implementations.

Two variants:
  AutonomousSystemM — switches modes when SSL loss or reward plateaus
  FixedSystemM      — switches every K environment steps
"""

from collections import deque
from enum import Enum, auto
from typing import Optional

import numpy as np


class Mode(Enum):
    OBSERVE = auto()   # System A is learning (LeWM trains)
    ACT     = auto()   # System B is learning (PPO trains)


# ---------------------------------------------------------------------------
# Autonomous System M
# ---------------------------------------------------------------------------

class AutonomousSystemM:
    """
    FSM that switches between OBSERVE and ACT based on detected learning plateaus.
    All timing is in *environment steps* to avoid dependency on call frequency.

    OBSERVE → ACT:  SSL loss hasn't improved (relative) in obs_plateau_steps env steps.
    ACT → OBSERVE:  Success rate is still low and not improving after act_plateau_steps
                    env steps in ACT.
    STOP:           success_rate >= solve_threshold.

    Parameters
    ----------
    obs_plateau_steps:  env steps of flat SSL loss before switching OBSERVE→ACT
    act_plateau_steps:  env steps before checking whether PPO is making progress
    plateau_threshold:  minimum relative improvement to NOT trigger a switch
    solve_threshold:    success rate considered "solved"
    """

    def __init__(
        self,
        obs_plateau_steps: int   = 8_000,
        act_plateau_steps: int   = 20_000,
        plateau_threshold: float = 0.01,
        solve_threshold:   float = 0.80,
        min_sr_to_stay:    float = 0.30,
    ):
        """
        min_sr_to_stay: don't switch back to OBSERVE unless success/score is
        below this value.  Set lower for sparse-reward environments like
        Crafter where scores are naturally small (e.g. 0.03).
        Raising this from the original 0.20 stops autonomous System M from
        interrupting a well-functioning LSTM-PPO that hasn't yet crossed a
        high threshold but is steadily improving.
        """
        self.obs_plateau_steps = obs_plateau_steps
        self.act_plateau_steps = act_plateau_steps
        self.plateau_threshold = plateau_threshold
        self.solve_threshold   = solve_threshold
        self.min_sr_to_stay    = min_sr_to_stay

        self.mode          = Mode.OBSERVE
        self._mode_start   = 0        # env_step when mode last changed
        self._ssl_buf      = []       # list of (env_step, ssl_loss)
        self._sr_buf       = []       # list of (env_step, success_rate)
        self.switch_log    = []
        self._solved       = False

    @property
    def is_solved(self) -> bool:
        return self._solved

    def observe_step(self, ssl_loss: float, env_step: int) -> Mode:
        """Call during OBSERVE mode each time LeWM is trained."""
        self._ssl_buf.append((env_step, ssl_loss))
        # Keep only entries within the plateau window
        cutoff = env_step - self.obs_plateau_steps
        self._ssl_buf = [(s, l) for s, l in self._ssl_buf if s >= cutoff]

        time_in = env_step - self._mode_start
        if time_in < self.obs_plateau_steps or len(self._ssl_buf) < 20:
            return self.mode

        # Plateau: first half vs second half of the window
        losses = [l for _, l in self._ssl_buf]
        mid    = len(losses) // 2
        h1     = float(np.mean(losses[:mid]))
        h2     = float(np.mean(losses[mid:]))
        rel    = abs(h1 - h2) / (h1 + 1e-8)

        if rel < self.plateau_threshold:
            self._switch(Mode.ACT, env_step)
        return self.mode

    def act_step(
        self,
        episode_reward: Optional[float],
        success_rate:   Optional[float],
        env_step:       int,
    ) -> Mode:
        """Call during ACT mode after each episode end or eval."""
        if success_rate is not None:
            self._sr_buf.append((env_step, success_rate))
            if success_rate >= self.solve_threshold:
                self._solved = True
                return self.mode

        time_in = env_step - self._mode_start
        if time_in < self.act_plateau_steps:
            return self.mode

        # Need at least 2 eval samples to judge progress
        recent = [s for _, s in self._sr_buf[-4:]] if self._sr_buf else []
        if len(recent) < 2:
            return self.mode

        # Switch back to OBSERVE only if PPO is both performing poorly AND
        # making no meaningful progress.  Using min_sr_to_stay (default 0.30)
        # instead of the old hardcoded 0.20 prevents interrupting an LSTM-PPO
        # that is still learning but hasn't crossed an arbitrary threshold.
        improvement = recent[-1] - recent[0]
        if recent[-1] < self.min_sr_to_stay and improvement < 0.03:
            self._switch(Mode.OBSERVE, env_step)

        return self.mode

    def _switch(self, new_mode: Mode, env_step: int) -> None:
        if new_mode == self.mode:
            return  # already in this mode, skip
        self.switch_log.append({
            "env_step": env_step,
            "from":     self.mode.name,
            "to":       new_mode.name,
        })
        self.mode        = new_mode
        self._mode_start = env_step
        self._ssl_buf.clear()
        self._sr_buf.clear()

    def n_switches(self) -> int:
        return len(self.switch_log)


# ---------------------------------------------------------------------------
# Fixed-schedule System M
# ---------------------------------------------------------------------------

class FixedSystemM:
    """
    Switches between OBSERVE and ACT every `switch_every` environment steps.
    Used as the baseline condition.

    Parameters
    ----------
    switch_every:    number of steps between mode switches
    start_mode:      which mode to begin in
    solve_threshold: if success_rate >= this, mark as solved
    """

    def __init__(
        self,
        switch_every:    int   = 10_000,
        start_mode:      Mode  = Mode.OBSERVE,
        solve_threshold: float = 0.80,
    ):
        self.switch_every    = switch_every
        self.mode            = start_mode
        self.solve_threshold = solve_threshold
        self._last_switch    = 0
        self.switch_log      = []
        self._solved         = False

    @property
    def is_solved(self) -> bool:
        return self._solved

    def step(
        self,
        env_step:     int,
        success_rate: Optional[float] = None,
    ) -> Mode:
        """Call every environment step."""
        if success_rate is not None and success_rate >= self.solve_threshold:
            self._solved = True
            return self.mode

        if env_step - self._last_switch >= self.switch_every:
            new_mode = Mode.ACT if self.mode == Mode.OBSERVE else Mode.OBSERVE
            self.switch_log.append({
                "env_step": env_step,
                "from":     self.mode.name,
                "to":       new_mode.name,
            })
            self.mode         = new_mode
            self._last_switch = env_step

        return self.mode

    def n_switches(self) -> int:
        return len(self.switch_log)
