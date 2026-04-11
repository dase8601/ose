"""
abm/ppo.py — Minimal self-contained PPO for discrete action spaces.

Used as System B in the A-B-M loop.  Takes pre-computed latent
embeddings (from a frozen LeWM encoder) as state — no CNN inside.

Supports vectorized environments: RolloutBuffer stores (n_steps, n_envs, ...)
so each outer loop iteration adds one row of N transitions.

Based on CleanRL PPO (vwxyzjn/cleanrl) condensed to ~200 lines.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Policy + Value networks
# ---------------------------------------------------------------------------

class PPOAgent(nn.Module):
    """
    Separate actor / critic networks operating on latent features.

    Input:  z (B, latent_dim)
    Actor:  → logits (B, n_actions)
    Critic: → value  (B, 1)
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Orthogonal init (recommended for PPO)
        for layer in [*self.actor, *self.critic]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def get_action_and_value(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        logits = self.actor(z)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)

    def get_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage — supports n_envs parallel environments
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores a single PPO rollout across n_envs parallel environments.

    Shape convention: (n_steps, n_envs, ...) — each call to add() fills one
    row across all envs simultaneously.  Total transitions = n_steps * n_envs.
    """

    def __init__(self, n_steps: int, n_envs: int, latent_dim: int, device: str):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.latent_dim = latent_dim
        self.device     = device
        self._ptr       = 0

        self.latents   = torch.zeros(n_steps, n_envs, latent_dim, device=device)
        self.actions   = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.rewards   = torch.zeros(n_steps, n_envs, device=device)
        self.dones     = torch.zeros(n_steps, n_envs, device=device)
        self.values    = torch.zeros(n_steps, n_envs, device=device)

    def add(
        self,
        z:        torch.Tensor,   # (n_envs, latent_dim)
        action:   torch.Tensor,   # (n_envs,)
        log_prob: torch.Tensor,   # (n_envs,)
        reward:   torch.Tensor,   # (n_envs,)
        done:     torch.Tensor,   # (n_envs,)
        value:    torch.Tensor,   # (n_envs,)
    ) -> None:
        i = self._ptr
        self.latents[i]   = z
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self._ptr += 1

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.n_steps

    def reset(self):
        self._ptr = 0

    def compute_gae(
        self,
        last_value: torch.Tensor,   # (n_envs,)
        gamma:      float = 0.99,
        gae_lam:    float = 0.95,
    ) -> tuple:
        """
        Generalised Advantage Estimation for n_envs parallel envs.
        last_value: bootstrap value for each env's current obs.
        """
        adv  = torch.zeros_like(self.rewards)              # (T, N)
        last = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.n_steps)):
            next_val  = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta     = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            adv[t]    = last = delta + gamma * gae_lam * (1 - next_done) * last

        returns = adv + self.values
        return adv, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

class PPO:
    """
    Proximal Policy Optimisation trainer.

    Works with both single-env (n_envs=1) and vectorized (n_envs>1) buffers.
    """

    CLIP_EPS   = 0.2
    ENT_COEF   = 0.01
    VF_COEF    = 0.5
    MAX_GRAD   = 0.5
    N_EPOCHS   = 4
    MINI_BATCH = 256   # larger mini-batch for vectorized data

    def __init__(self, agent: PPOAgent, lr: float = 2.5e-4):
        self.agent = agent
        self.opt   = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    def update(
        self,
        buf:        RolloutBuffer,
        last_value: torch.Tensor,   # (n_envs,) bootstrap values
        last_done:  torch.Tensor,   # (n_envs,) done flags
    ) -> dict:
        # Bootstrap: zero out value for finished envs
        last_val = last_value * (1 - last_done.float())
        adv, returns = buf.compute_gae(last_val)

        # Flatten (T, N, ...) → (T*N, ...)
        n_total = buf.n_steps * buf.n_envs
        b_lat   = buf.latents.reshape(n_total, buf.latent_dim).detach()
        b_act   = buf.actions.reshape(n_total).detach()
        b_lp    = buf.log_probs.reshape(n_total).detach()
        b_ret   = returns.reshape(n_total).detach()
        b_adv   = adv.reshape(n_total).detach()

        # Normalise advantages over the full batch
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        indices = np.arange(n_total)
        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, n_total, self.MINI_BATCH):
                mb = indices[start: start + self.MINI_BATCH]

                _, new_lp, ent, new_val = self.agent.get_action_and_value(
                    b_lat[mb], b_act[mb]
                )
                ratio = (new_lp - b_lp[mb]).exp()

                pg_loss1 = -b_adv[mb] * ratio
                pg_loss2 = -b_adv[mb] * ratio.clamp(1 - self.CLIP_EPS, 1 + self.CLIP_EPS)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = 0.5 * F.mse_loss(new_val, b_ret[mb])
                ent_loss = ent.mean()

                loss = pg_loss + self.VF_COEF * vf_loss - self.ENT_COEF * ent_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.MAX_GRAD)
                self.opt.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        buf.reset()
        return {
            "pg_loss":  np.mean(pg_losses),
            "vf_loss":  np.mean(vf_losses),
            "ent_loss": np.mean(ent_losses),
        }
