"""
abm/ppo.py — LSTM-PPO for discrete action spaces.

Used as System B in the A-B-M loop.  Takes pre-computed latent
embeddings (from a frozen LeWM encoder) as state — no CNN inside.

LSTM provides working memory across timesteps, which is critical for
partial-observability in MiniGrid (the agent only sees a forward cone
and cannot directly observe whether it has already picked up the key).

Rollout shape convention: (n_steps, n_envs, ...).
Total transitions per update = n_steps × n_envs.

Based on CleanRL PPO-LSTM (vwxyzjn/cleanrl).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Policy + Value networks (LSTM-backed)
# ---------------------------------------------------------------------------

class PPOAgent(nn.Module):
    """
    LSTM actor-critic operating on latent features.

    Input:  z (N, latent_dim)  +  lstm_state (h, c)  +  done (N,)
    LSTM:   z → h (N, hidden_size)
    Actor:  h → logits (N, n_actions)
    Critic: h → value  (N,)

    The LSTM hidden state carries memory across timesteps, enabling the
    agent to remember sub-goal completions (key pickup, door open) even
    when the agent's view cone no longer shows those objects.
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.hidden_size = hidden
        self.lstm   = nn.LSTMCell(latent_dim, hidden)
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init (PPO best practice)
        for name, p in self.named_parameters():
            if "weight" in name and p.ndim >= 2:
                if "actor" in name:
                    nn.init.orthogonal_(p, gain=0.01)
                elif "critic" in name:
                    nn.init.orthogonal_(p, gain=1.0)
                else:
                    nn.init.orthogonal_(p, gain=np.sqrt(2))
            elif "bias" in name:
                nn.init.zeros_(p)

    def get_initial_state(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero LSTM state for n environments."""
        h = torch.zeros(n, self.hidden_size, device=device)
        c = torch.zeros(n, self.hidden_size, device=device)
        return h, c

    def _step_lstm(
        self,
        z:    torch.Tensor,   # (N, latent_dim)
        h:    torch.Tensor,   # (N, hidden_size)
        c:    torch.Tensor,   # (N, hidden_size)
        done: torch.Tensor,   # (N,) float — 1 if episode just ended
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One LSTMCell step with episode-boundary reset."""
        mask = (1.0 - done).unsqueeze(-1)   # (N, 1)
        h = h * mask
        c = c * mask
        return self.lstm(z, (h, c))         # h_new, c_new  each (N, H)

    def get_action_and_value(
        self,
        z:          torch.Tensor,                    # (N, latent_dim)
        lstm_state: Tuple[torch.Tensor, torch.Tensor],  # (h, c)
        done:       torch.Tensor,                    # (N,) float
        action:     Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Step the LSTM and sample an action.

        Returns: action, log_prob, entropy, value, (h_new, c_new)
        """
        h, c = lstm_state
        h, c = self._step_lstm(z, h, c, done)

        logits = self.actor(h)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1), (h, c)

    def get_action_and_value_from_hidden(
        self,
        h:      torch.Tensor,                    # (B, hidden_size) — pre-computed
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Actor-critic pass using a pre-computed LSTM hidden state.
        Used during the PPO update where LSTM outputs are replayed once
        per epoch (not per mini-batch) for correctness + efficiency.

        Returns: action, log_prob, entropy, value
        """
        logits = self.actor(h)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1)

    def get_value(
        self,
        z:          torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        done:       torch.Tensor,
    ) -> torch.Tensor:
        """Bootstrap value estimate for the last observation."""
        h, c = lstm_state
        h, c = self._step_lstm(z, h, c, done)
        return self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage — supports n_envs parallel environments
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores a single PPO rollout across n_envs parallel environments.

    Shape convention: (n_steps, n_envs, ...) — each call to add() fills one
    row across all envs simultaneously.  Total transitions = n_steps * n_envs.

    Also stores lstm_h0, lstm_c0: the LSTM hidden state at the very start
    of this rollout (used to replay the LSTM sequentially during the update).
    """

    def __init__(self, n_steps: int, n_envs: int, latent_dim: int, device: str,
                 hidden_size: int = 256):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.device     = device
        self._ptr       = 0

        self.latents   = torch.zeros(n_steps, n_envs, latent_dim, device=device)
        self.actions   = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.rewards   = torch.zeros(n_steps, n_envs, device=device)
        self.dones     = torch.zeros(n_steps, n_envs, device=device)
        self.values    = torch.zeros(n_steps, n_envs, device=device)

        # Initial LSTM state for the rollout — set via set_lstm_initial_state()
        self.lstm_h0: Optional[torch.Tensor] = None
        self.lstm_c0: Optional[torch.Tensor] = None

    def set_lstm_initial_state(
        self,
        h: torch.Tensor,   # (N, hidden_size)
        c: torch.Tensor,   # (N, hidden_size)
    ) -> None:
        """Call once at the very start of each rollout (when _ptr == 0)."""
        self.lstm_h0 = h.detach().clone()
        self.lstm_c0 = c.detach().clone()

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
        self.lstm_h0 = None
        self.lstm_c0 = None

    def compute_gae(
        self,
        last_value: torch.Tensor,   # (n_envs,)
        gamma:      float = 0.99,
        gae_lam:    float = 0.95,
    ) -> tuple:
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
# PPO update (LSTM-aware)
# ---------------------------------------------------------------------------

class PPO:
    """
    Proximal Policy Optimisation trainer with LSTM support.

    LSTM replay strategy:
      Each update epoch starts by replaying the stored rollout sequentially
      through the LSTM (using the stored initial hidden state + dones for
      boundary resets).  This produces a (T×N, hidden_size) tensor of LSTM
      outputs which is then mini-batch sampled for the policy gradient loss.

      This is equivalent to the CleanRL LSTM-PPO approach: compute LSTM
      outputs once per epoch (correct gradient-wise), not per mini-batch
      (which would require carrying gradients through the full sequence).
    """

    CLIP_EPS   = 0.2
    ENT_COEF   = 0.01
    VF_COEF    = 0.5
    MAX_GRAD   = 0.5
    N_EPOCHS   = 4
    MINI_BATCH = 256

    def __init__(self, agent: PPOAgent, lr: float = 2.5e-4):
        self.agent = agent
        self.opt   = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    def _replay_lstm(self, buf: RolloutBuffer) -> torch.Tensor:
        """
        Sequential LSTM forward pass through the entire rollout.

        Uses buf.lstm_h0/c0 as initial state; resets on episode boundaries
        (buf.dones).  Returns (T*N, hidden_size) of LSTM outputs (detached).
        """
        h = buf.lstm_h0    # (N, H)
        c = buf.lstm_c0    # (N, H)
        outs = []
        for t in range(buf.n_steps):
            h, c = self.agent._step_lstm(buf.latents[t], h, c, buf.dones[t])
            outs.append(h)                             # (N, H)
        b_hidden = torch.stack(outs, dim=0)            # (T, N, H)
        return b_hidden.reshape(-1, self.agent.hidden_size).detach()   # (T*N, H)

    def update(
        self,
        buf:        RolloutBuffer,
        last_value: torch.Tensor,   # (n_envs,) bootstrap values
        last_done:  torch.Tensor,   # (n_envs,) done flags
    ) -> dict:
        last_val = last_value * (1 - last_done.float())
        adv, returns = buf.compute_gae(last_val)

        n_total = buf.n_steps * buf.n_envs
        b_act   = buf.actions.reshape(n_total).detach()
        b_lp    = buf.log_probs.reshape(n_total).detach()
        b_ret   = returns.reshape(n_total).detach()
        b_adv   = adv.reshape(n_total).detach()
        b_adv   = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        indices = np.arange(n_total)
        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.N_EPOCHS):
            # Replay LSTM once per epoch to get fresh hidden states
            b_hid = self._replay_lstm(buf)   # (T*N, H)

            np.random.shuffle(indices)
            for start in range(0, n_total, self.MINI_BATCH):
                mb = indices[start: start + self.MINI_BATCH]

                _, new_lp, ent, new_val = self.agent.get_action_and_value_from_hidden(
                    b_hid[mb], b_act[mb]
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
