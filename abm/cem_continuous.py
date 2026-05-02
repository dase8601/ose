"""
abm/cem_continuous.py — ContinuousCEMPlanner for continuous action spaces.

Replaces the categorical-logit CEM from cem_planner.py with a Gaussian CEM that
operates over H×a_dim continuous action sequences. Designed for ManiSkill3 robot
manipulation environments where actions are normalised to [-1, 1].

Algorithm (per call to plan()):
  1. Initialise (or warm-start) a factorised Gaussian μ ∈ R^{H×a_dim}, σ ∈ R^{H×a_dim}
  2. For n_iters iterations:
       a. Sample K action sequences from N(μ, σ²), clipped to [-1, 1]
       b. Roll each sequence forward through the predictor for H steps
       c. Compute cosine distance to z_goal at the final latent state
       d. Refit μ, σ from the top n_elites sequences (momentum update)
  3. Execute only mu[0] — the first action of the best sequence (MPC)

Warm-starting: after each call plan() shifts the mean left by 1 step and pads
with zeros, which reuses solution overlap from the previous MPC step. Call
reset() at episode boundaries to zero the warm start.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ContinuousCEMPlanner:
    """
    Cross-Entropy Method planner for continuous action spaces.

    Parameters
    ----------
    predictor : nn.Module
        Must implement forward(z, a) → z_next where a is a (B, a_dim) float tensor.
        Typically the MLP Predictor from world_model.py with n_actions=a_dim.
    a_dim : int
        Action dimensionality (e.g. 6 for arm_pd_ee_delta_pose).
    H : int
        Planning horizon (number of steps to roll out).
    K : int
        Number of candidate action sequences sampled per CEM iteration.
    n_elites : int
        Number of top sequences used to refit the distribution.
    n_iters : int
        Number of CEM refinement iterations per plan() call.
    device : str
        Torch device.
    momentum : float
        Exponential moving average weight applied when updating μ from elites.
        0.0 = fully replace, 1.0 = never update. Default 0.1 (light smoothing).
    min_sigma : float
        Floor on σ to prevent premature convergence.
    """

    def __init__(
        self,
        predictor,
        a_dim: int = 6,
        H: int = 8,
        K: int = 300,
        n_elites: int = 30,
        n_iters: int = 30,
        device: str = "cuda",
        momentum: float = 0.1,
        min_sigma: float = 0.05,
    ):
        self.predictor = predictor
        self.a_dim     = a_dim
        self.H         = H
        self.K         = K
        self.n_elites  = n_elites
        self.n_iters   = n_iters
        self.device    = device
        self.momentum  = momentum
        self.min_sigma = min_sigma

        # Warm-start state (reset at episode boundaries)
        self._mu    = np.zeros((H, a_dim), dtype=np.float32)
        self._sigma = np.ones((H, a_dim), dtype=np.float32)

    def reset(self):
        """Zero the warm-start state. Call at the start of each episode."""
        self._mu[:]    = 0.0
        self._sigma[:] = 1.0

    def plan(self, z_cur: torch.Tensor, z_goal: torch.Tensor) -> np.ndarray:
        """
        Run CEM and return the first action of the best sequence.

        Parameters
        ----------
        z_cur  : (1, z_dim) torch tensor — current latent state
        z_goal : (1, z_dim) torch tensor — goal latent state

        Returns
        -------
        action : (a_dim,) numpy float32 array, clipped to [-1, 1]
        """
        mu    = self._mu.copy()
        sigma = self._sigma.copy()

        for _ in range(self.n_iters):
            # Sample K × H × a_dim action sequences
            noise   = np.random.randn(self.K, self.H, self.a_dim).astype(np.float32)
            actions = np.clip(mu[None] + sigma[None] * noise, -1.0, 1.0)  # (K, H, a_dim)

            costs = self._rollout_costs(z_cur, actions, z_goal)  # (K,)

            # Select elites
            elite_idx = np.argsort(costs)[: self.n_elites]
            elites    = actions[elite_idx]  # (n_elites, H, a_dim)

            # Refit with momentum
            new_mu    = elites.mean(axis=0)
            new_sigma = elites.std(axis=0) + self.min_sigma
            mu    = (1 - self.momentum) * new_mu    + self.momentum * mu
            sigma = (1 - self.momentum) * new_sigma + self.momentum * sigma

        # Warm-start shift: slide window left by 1, pad last step with 0/1
        self._mu    = np.concatenate([mu[1:],    np.zeros((1, self.a_dim), dtype=np.float32)],    axis=0)
        self._sigma = np.concatenate([sigma[1:], np.ones((1, self.a_dim),  dtype=np.float32)],    axis=0)

        return mu[0].copy()  # (a_dim,) — first action to execute

    @torch.no_grad()
    def _rollout_costs(
        self,
        z_cur: torch.Tensor,    # (1, z_dim)
        actions: np.ndarray,    # (K, H, a_dim)
        z_goal: torch.Tensor,   # (1, z_dim)
    ) -> np.ndarray:
        """Roll K action sequences through the predictor, return cosine costs."""
        K   = actions.shape[0]
        dev = z_cur.device

        z = z_cur.expand(K, -1).clone()  # (K, z_dim)
        a_tensor = torch.from_numpy(actions).to(dev)  # (K, H, a_dim)

        for h in range(self.H):
            z = self.predictor(z, a_tensor[:, h])  # (K, z_dim)

        # Cosine distance to goal
        costs = 1.0 - F.cosine_similarity(z, z_goal.expand(K, -1), dim=-1)
        return costs.cpu().numpy()  # (K,)
