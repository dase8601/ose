"""
abm/mpc.py — Cross-Entropy Method (CEM) Model-Predictive Control planner.

Upgraded from Random Shooting to CEM per "What Drives Success in Physical
Planning with JEPA-WMs" (Terver et al., Jan 2026 — Yann's team).
Uses cosine distance (not L2) since DINOv2 features are cosine-trained.

CEM iteratively refines action sequences by:
1. Sample K action sequences from current distribution
2. Evaluate each by rolling out through the world model
3. Keep top-E elite sequences (lowest distance to goal)
4. Refit distribution to elite sequences
5. Repeat for N_ITER iterations

This finds much better plans than random shooting because it concentrates
samples in promising regions rather than covering the space uniformly.

Reference:
  Terver et al. "What Drives Success in Physical Planning with JEPA-WMs"
  Zhou et al. "DINO-WM: World Models on Pre-trained Visual Features
  enable Zero-shot Planning"

Usage:
    mpc = CEMPlanner(predictor, n_actions=3)
    actions = mpc.plan_batch(z_current, z_goal)   # (N,) int
    action  = mpc.plan_single(z_current, z_goal)  # int
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EBMCostHead(nn.Module):
    """
    Learned energy function E(z, z_goal) → scalar cost for CEM planning.

    Low energy  = state is close to goal (CEM prefers low-cost sequences).
    High energy = state is far from goal.

    Trained contrastively:
      Positive pairs: (z_success, z_goal) from GoalImageBuffer — low energy
      Negative pairs: (z_random,  z_goal) from replay buffer  — high energy

    Loss: max(0, margin + E_pos - E_neg)  [margin-ranking / hinge loss]

    Replaces cosine distance in CEMPlanner._rollout once trained.
    Directly implements LeCun's energy-based objective-driven AI cost module.
    """

    def __init__(self, latent_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """z, z_goal: (B, D) → (B,) energy scalars"""
        return self.net(torch.cat([z, z_goal], dim=-1)).squeeze(-1)

    def contrastive_loss(
        self,
        z_pos:  torch.Tensor,   # (B, D) — success-adjacent states
        z_neg:  torch.Tensor,   # (B, D) — random states
        z_goal: torch.Tensor,   # (B, D) — goal encodings
        margin: float = 1.0,
    ) -> torch.Tensor:
        e_pos = self.forward(z_pos,  z_goal)
        e_neg = self.forward(z_neg,  z_goal)
        return torch.clamp(margin + e_pos - e_neg, min=0).mean()


class CEMPlanner:
    """
    Cross-Entropy Method planner in DINOv2 representation space.

    For discrete actions: maintains a categorical distribution over actions
    at each timestep. CEM iteratively refines this distribution toward
    sequences that reach the goal.

    Parameters
    ----------
    predictor  : VJEPAPredictor — trained action-conditioned world model
    n_actions  : int — size of the discrete action space
    horizon    : int — planning horizon H
    n_samples  : int — candidates per CEM iteration
    n_elites   : int — top candidates to refit distribution
    n_iters    : int — CEM iterations (more = better plans, slower)
    device     : str — torch device
    """

    def __init__(
        self,
        predictor,
        n_actions: int,
        horizon:   int = 7,
        n_samples: int = 256,
        n_elites:  int = 32,
        n_iters:   int = 3,
        device:    str = "cuda",
        distance:  str = "cosine",
    ):
        self.predictor = predictor
        self.n_actions = n_actions
        self.horizon   = horizon
        self.n_samples = n_samples
        self.n_elites  = n_elites
        self.n_iters   = n_iters
        self.device    = device
        self.distance  = distance
        self.ebm       = None   # set via set_ebm() once trained

    def set_ebm(self, ebm: EBMCostHead) -> None:
        """Activate learned EBM cost in place of cosine distance."""
        self.ebm = ebm

    @torch.no_grad()
    def _rollout(
        self,
        z_start: torch.Tensor,    # (B*K, feat_dim)
        actions: torch.Tensor,     # (B, K, H) int
        z_goal:  torch.Tensor,     # (B, feat_dim)
    ) -> torch.Tensor:
        """Roll out action sequences through predictor, return distances to goal."""
        B, K, H = actions.shape
        feat_dim = z_start.shape[-1]

        z = z_start.clone()
        for t in range(H):
            a_flat   = actions[:, :, t].reshape(B * K)
            a_onehot = F.one_hot(a_flat, self.n_actions).float()
            z        = self.predictor(z, a_onehot)

        z_goal_exp = z_goal.unsqueeze(1).expand(B, K, feat_dim).reshape(B * K, feat_dim)
        if self.ebm is not None:
            dist = self.ebm(z, z_goal_exp).reshape(B, K)
        elif self.distance == "l2":
            dist = (z - z_goal_exp).pow(2).sum(dim=-1).reshape(B, K)
        else:
            cos_sim = F.cosine_similarity(z, z_goal_exp, dim=-1)
            dist = (1.0 - cos_sim).reshape(B, K)
        return dist

    @torch.no_grad()
    def plan_batch(
        self,
        z_current: torch.Tensor,   # (B, feat_dim)
        z_goal:    torch.Tensor,   # (B, feat_dim) or (1, feat_dim)
    ) -> "np.ndarray":
        """Plan for a batch of B environments. Returns (B,) numpy int array."""
        import numpy as np

        B        = z_current.shape[0]
        K        = self.n_samples
        H        = self.horizon
        feat_dim = z_current.shape[1]

        # Initialize uniform categorical distribution: (B, H, n_actions)
        logits = torch.zeros(B, H, self.n_actions, device=self.device)

        for _iter in range(self.n_iters):
            # Sample from current distribution
            probs = F.softmax(logits, dim=-1)  # (B, H, n_actions)
            actions = torch.zeros(B, K, H, dtype=torch.long, device=self.device)
            for t in range(H):
                actions[:, :, t] = torch.multinomial(
                    probs[:, t].repeat_interleave(K, dim=0),
                    num_samples=1,
                ).reshape(B, K)

            # Expand z_current for K candidates
            z_start = z_current.unsqueeze(1).expand(B, K, feat_dim).reshape(B * K, feat_dim)

            # Evaluate all candidates
            dist = self._rollout(z_start, actions, z_goal)  # (B, K)

            # Select elites (top-E lowest distance)
            _, elite_idx = dist.topk(self.n_elites, dim=1, largest=False)  # (B, E)

            # Gather elite actions
            elite_actions = torch.gather(
                actions,
                dim=1,
                index=elite_idx.unsqueeze(-1).expand(B, self.n_elites, H),
            )  # (B, E, H)

            # Refit distribution: count action frequencies in elites
            logits = torch.zeros(B, H, self.n_actions, device=self.device)
            for t in range(H):
                for a in range(self.n_actions):
                    logits[:, t, a] = (elite_actions[:, :, t] == a).float().sum(dim=1)
                logits[:, t] = logits[:, t] + 1.0  # Laplace smoothing

        # Final: pick best first action from last iteration's best candidate
        best_idx = dist.argmin(dim=1)  # (B,)
        best_actions = actions[torch.arange(B, device=self.device), best_idx, 0]
        self._last_best_cost = dist[torch.arange(B, device=self.device), best_idx].mean().item()

        return best_actions.cpu().numpy()

    @torch.no_grad()
    def plan_single(
        self,
        z_current: torch.Tensor,
        z_goal:    torch.Tensor,
    ) -> int:
        """Plan for a single environment. Returns best first action as int."""
        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)
        return int(self.plan_batch(z_current, z_goal)[0])


# Keep backward compatibility alias
RandomShootingMPC = CEMPlanner


class GoalBuffer:
    """
    Maintains a rolling buffer of DINOv2 goal encodings.

    Populated during OBSERVE phase when high-reward states are observed,
    or pre-seeded via explicit goal capture (teleport/rollout).
    Used during ACT phase to provide z_goal for MPC planning.
    """

    def __init__(self, max_size: int = 100, device: str = "cuda"):
        self.max_size = max_size
        self.device   = device
        self._buf     = []

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
