"""
abm/lewm.py — LeWorldModel (LeWM) for MiniGrid

Implements the two core components described in the LeWM paper:
  - Encoder:   CNN → latent_dim=128 embedding
  - Predictor: action-conditioned next-latent predictor
  - SIGReg:    Gaussian regulariser (isotropic N(0,I) enforcement)
  - Loss:      MSE(pred_z_next, sg(z_next)) + lambda * SIGReg(z)

Also contains the ReplayBuffer used during OBSERVE mode.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder  (3, 48, 48) → (latent_dim,)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    CNN encoder mapping pixel observations to latent embeddings.

    Input:  (B, 3, H, W) float32 in [0, 1]  — H=W=48 for MiniGrid-DoorKey-6x6
    Output: (B, latent_dim)

    Architecture (verified output sizes for 48×48 input):
      Conv(3→32,  k=4, s=2)  → (B, 32, 23, 23)
      Conv(32→64, k=3, s=2)  → (B, 64, 11, 11)
      Conv(64→64, k=3, s=1)  → (B, 64,  9,  9)
      Flatten                 → (B, 5184)
      Linear(5184, latent)    → (B, latent_dim)

    Compared to the 128-dim encoder: second conv uses k=3 instead of k=4
    (retains more spatial resolution) and latent_dim is doubled to 256
    for richer representation of the partial-observable scene.
    """

    def __init__(self, latent_dim: int = 256, img_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flat size by dry-run
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, 48, 48)
            flat_size = self.cnn(dummy).shape[1]

        self.head = nn.Linear(flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, latent_dim)"""
        return self.head(self.cnn(x))


# ---------------------------------------------------------------------------
# Predictor  (latent_dim + n_actions) → latent_dim
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """
    Action-conditioned next-latent predictor.

    Input:  z_t (B, latent_dim) + one_hot(a_t, n_actions)
    Output: z_{t+1}_pred (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 128, n_actions: int = 7, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        """
        z:        (B, latent_dim)
        a_onehot: (B, n_actions)
        Returns:  (B, latent_dim)
        """
        return self.net(torch.cat([z, a_onehot], dim=-1))


# ---------------------------------------------------------------------------
# SIGReg — enforce z ~ N(0, I) via random projections
# ---------------------------------------------------------------------------

def sigreg(z: torch.Tensor, n_proj: int = 512) -> torch.Tensor:
    """
    Gaussian regulariser from the LeWM / SIGReg paper.

    Projects z onto n_proj random unit vectors and enforces each
    projection to be N(0, 1):  mean → 0,  std → 1.

    z: (B, D)  →  scalar loss
    """
    B, D = z.shape
    W = torch.randn(D, n_proj, device=z.device, dtype=z.dtype)
    W = F.normalize(W, dim=0)          # unit columns
    proj = z @ W                        # (B, n_proj)

    mean_loss = proj.mean(0).pow(2).mean()
    std_loss  = (proj.std(0) - 1.0).pow(2).mean()
    return mean_loss + std_loss


# ---------------------------------------------------------------------------
# LeWM — combined world model
# ---------------------------------------------------------------------------

class LeWM(nn.Module):
    """
    LeWorldModel for MiniGrid.

    Usage (OBSERVE mode):
        lewm = LeWM(latent_dim=256, n_actions=7).to(device)
        loss, info = lewm.loss(obs_t, action_t, obs_next)
        loss.backward()

    Usage (ACT mode — frozen encoder):
        with torch.no_grad():
            z = lewm.encode(obs)  # (B, 256)
    """

    SIGREG_LAMBDA = 0.1    # weight of regularisation term

    def __init__(self, latent_dim: int = 256, n_actions: int = 7):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions  = n_actions
        self.encoder    = Encoder(latent_dim)
        self.predictor  = Predictor(latent_dim, n_actions)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, H, W) float32 [0,1]  →  (B, latent_dim)"""
        return self.encoder(obs)

    def loss(
        self,
        obs_t:    torch.Tensor,   # (B, 3, H, W)
        action_t: torch.Tensor,   # (B,) int64
        obs_next: torch.Tensor,   # (B, 3, H, W)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute:
          L = MSE(predictor(z_t, a_t), sg(z_{t+1})) + λ·SIGReg(z_t)

        sg = stop-gradient (detach) on the target embedding — prevents
        collapse by making the target a fixed signal rather than a moving
        one that the encoder can trivially match.  SIGReg handles the
        representation diversity constraint instead of EMA/momentum.
        """
        z_t    = self.encoder(obs_t)                           # (B, D)
        z_next = self.encoder(obs_next).detach()               # (B, D) — target

        a_onehot = F.one_hot(action_t, self.n_actions).float() # (B, n_actions)
        z_pred   = self.predictor(z_t, a_onehot)               # (B, D)

        pred_loss = F.mse_loss(z_pred, z_next)
        reg_loss  = sigreg(z_t)
        total     = pred_loss + self.SIGREG_LAMBDA * reg_loss

        # Monitor representation rank (move to CPU — MPS lacks linalg.svd)
        with torch.no_grad():
            cov = (z_t.T @ z_t / z_t.shape[0]).cpu()
            rank = torch.linalg.matrix_rank(cov, atol=1e-3).item()

        return total, {
            "loss_pred":  pred_loss.item(),
            "loss_reg":   reg_loss.item(),
            "loss_total": total.item(),
            "z_rank":     rank,
            "z_std":      z_t.std().item(),
        }


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular replay buffer storing (obs_t, action_t, obs_next) tuples.
    Used to train LeWM during OBSERVE mode.
    """

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs_t:    np.ndarray,   # (H, W, C) uint8
        action_t: int,
        obs_next: np.ndarray,   # (H, W, C) uint8
    ) -> None:
        self._buf.append((obs_t, action_t, obs_next))

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, ...]:
        """Returns (obs_t, action_t, obs_next) tensors on device."""
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))
        obs_t, acts, obs_next = zip(*batch)

        def to_tensor(imgs):
            arr = np.stack(imgs).astype(np.float32) / 255.0   # (B, H, W, C)
            return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # (B, C, H, W)

        return (
            to_tensor(obs_t),
            torch.tensor(acts, dtype=torch.long, device=device),
            to_tensor(obs_next),
        )

    def __len__(self) -> int:
        return len(self._buf)
