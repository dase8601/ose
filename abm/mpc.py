"""
abm/mpc.py — Random Shooting Model-Predictive Control planner.

Implements DINO-WM style planning: given a trained world model predictor
and a goal encoding in representation space, find the action sequence that
minimizes distance to the goal.

No reinforcement learning, no policy gradient, no catastrophic forgetting.
Works zero-shot once the world model is trained during OBSERVE phase.

Reference:
  Zhou et al. "DINO-WM: World Models on Pre-trained Visual Features
  Enable Zero-shot Planning" — dino-wm.github.io

  Bar et al. "Navigation World Models: MPC planning from natural
  motion-conditioned videos" — arXiv:2412.03572

Usage:
    mpc = RandomShootingMPC(predictor, n_actions=3)

    # Plan for a batch of N envs simultaneously
    actions = mpc.plan_batch(z_current, z_goal)   # (N,) int

    # Plan for a single env
    action  = mpc.plan_single(z_current, z_goal)  # int
"""

import torch
import torch.nn.functional as F


class RandomShootingMPC:
    """
    Random Shooting MPC planner in DINOv2 representation space.

    Samples K random action sequences of length H, rolls each through the
    trained world model predictor, and selects the sequence whose final
    predicted state is closest to the goal state in L2 distance.

    Executes only the first action (receding horizon / replanning).

    Parameters
    ----------
    predictor  : VJEPAPredictor — trained action-conditioned world model
    n_actions  : int — size of the discrete action space
    horizon    : int — planning horizon H (number of steps to look ahead)
    n_samples  : int — number of random action sequences K to evaluate
    device     : str — torch device
    """

    def __init__(
        self,
        predictor,
        n_actions: int,
        horizon:   int = 7,
        n_samples: int = 256,
        device:    str = "cuda",
    ):
        self.predictor = predictor
        self.n_actions = n_actions
        self.horizon   = horizon
        self.n_samples = n_samples
        self.device    = device

    @torch.no_grad()
    def plan_batch(
        self,
        z_current: torch.Tensor,   # (B, feat_dim)
        z_goal:    torch.Tensor,   # (B, feat_dim)  or  (1, feat_dim) broadcast
    ) -> "np.ndarray":
        """
        Plan for a batch of B environments simultaneously.

        Returns: (B,) numpy int array — best first action per environment.
        """
        import numpy as np

        B        = z_current.shape[0]
        K        = self.n_samples
        feat_dim = z_current.shape[1]

        # Sample K random action sequences for each env: (B, K, H)
        actions = torch.randint(
            0, self.n_actions, (B, K, self.horizon), device=self.device
        )

        # Expand z_current → (B*K, feat_dim)
        z = z_current.unsqueeze(1).expand(B, K, feat_dim).reshape(B * K, feat_dim).clone()

        # Roll out through predictor for H steps
        for t in range(self.horizon):
            a_flat   = actions[:, :, t].reshape(B * K)          # (B*K,)
            a_onehot = F.one_hot(a_flat, self.n_actions).float() # (B*K, n_actions)
            z        = self.predictor(z, a_onehot)               # (B*K, feat_dim)

        # Compute L2 distance to goal in representation space
        z_goal_exp = z_goal.unsqueeze(1).expand(B, K, feat_dim).reshape(B * K, feat_dim)
        dist = ((z - z_goal_exp) ** 2).sum(-1).reshape(B, K)    # (B, K)

        # Select best first action per env
        best_idx     = dist.argmin(dim=1)                        # (B,)
        best_actions = actions[torch.arange(B, device=self.device), best_idx, 0]

        return best_actions.cpu().numpy()

    @torch.no_grad()
    def plan_single(
        self,
        z_current: torch.Tensor,   # (1, feat_dim) or (feat_dim,)
        z_goal:    torch.Tensor,   # (1, feat_dim) or (feat_dim,)
    ) -> int:
        """Plan for a single environment. Returns best first action as int."""
        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)
        return int(self.plan_batch(z_current, z_goal)[0])


class GoalBuffer:
    """
    Maintains a rolling buffer of DINOv2 goal encodings.

    Populated passively during OBSERVE phase whenever a positive reward
    is observed (agent accidentally reached the goal during random exploration).
    Used during PLAN phase to provide the MPC planner with z_goal.

    If empty (goal never seen), returns None and MPC falls back to random actions.
    """

    def __init__(self, max_size: int = 100, device: str = "cuda"):
        self.max_size = max_size
        self.device   = device
        self._buf     = []          # list of (1, feat_dim) cpu tensors

    def push(self, z: torch.Tensor) -> None:
        """Add a goal encoding. z: (1, feat_dim) or (feat_dim,)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        self._buf.append(z.cpu())
        if len(self._buf) > self.max_size:
            self._buf = self._buf[-self.max_size:]

    def get_goal(self) -> "torch.Tensor | None":
        """Return mean goal encoding on device, or None if buffer is empty."""
        if not self._buf:
            return None
        return torch.cat(self._buf, dim=0).mean(0, keepdim=True).to(self.device)

    def __len__(self) -> int:
        return len(self._buf)
