"""
abm/rnd.py — Random Network Distillation (RND) for intrinsic reward.

Burda et al., "Exploration by Random Network Distillation" (ICLR 2019).

RND gives PPO a curiosity bonus on every step, which is critical for
sparse-reward environments like Crafter where most episodes yield 0
extrinsic reward.

Usage:
    rnd = RND(input_dim=512).to(device)
    intrinsic_reward = rnd.reward(z)           # (N,) float
    rnd_loss = rnd.loss(z)                     # scalar, call .backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RND(nn.Module):
    """
    Two small MLPs:
      - target:    random weights, frozen, never trained
      - predictor: trained to match target's output

    Intrinsic reward = per-sample MSE(predictor(z), target(z)).
    High when z is novel (predictor hasn't learned this region yet).
    Low when z is familiar.
    """

    def __init__(self, input_dim: int, hidden: int = 256, output_dim: int = 128):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        # Freeze target — never trained
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Running stats for reward normalization
        self.register_buffer("reward_mean", torch.zeros(1))
        self.register_buffer("reward_var", torch.ones(1))
        self._reward_count = 0

    @torch.no_grad()
    def reward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized intrinsic reward.
        z: (N, input_dim) — detached latent embeddings
        Returns: (N,) float — per-env intrinsic reward
        """
        pred = self.predictor(z)
        targ = self.target(z)
        raw = (pred - targ).pow(2).mean(dim=-1)  # (N,)

        # Update running mean/var for normalization
        batch_mean = raw.mean()
        batch_var = raw.var().clamp(min=1e-8)
        self._reward_count += 1
        alpha = max(1.0 / self._reward_count, 0.01)
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
        self.reward_var = (1 - alpha) * self.reward_var + alpha * batch_var

        # Normalize so intrinsic reward has roughly unit scale
        return (raw - self.reward_mean) / (self.reward_var.sqrt() + 1e-8)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predictor training loss. Call during ACT mode to update predictor.
        z: (N, input_dim) — latent embeddings (can have grad or not)
        Returns: scalar loss
        """
        pred = self.predictor(z.detach())
        with torch.no_grad():
            targ = self.target(z.detach())
        return F.mse_loss(pred, targ)
