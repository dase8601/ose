"""
record_episodes.py — Record MP4 video of trained agent episodes.

Loads a checkpoint from results/abm/checkpoint_{condition}.pt,
runs N greedy episodes, and saves MP4s to results/abm/videos/{condition}/.

Usage:
    python record_episodes.py --condition fixed --n 5
    python record_episodes.py --condition autonomous --n 10
    python record_episodes.py --condition ppo_only --n 5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch
import numpy as np
import gymnasium
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import RGBImgObsWrapper

from abm.lewm import LeWM
from abm.ppo import PPOAgent
from abm.loop import ShapedRewardWrapper

LATENT_DIM  = 256
HIDDEN_SIZE = 256
N_ACTIONS   = 7
IMG_H = IMG_W = 48


def make_record_env(video_dir: Path, episode_prefix: str, seed: int = 0):
    """Build env with RecordVideo wrapper around it."""
    base = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    base = ShapedRewardWrapper(base)
    base = RecordVideo(
        base,
        video_folder=str(video_dir),
        name_prefix=episode_prefix,
        episode_trigger=lambda ep: True,  # record every episode
    )
    env = RGBImgObsWrapper(base, tile_size=8)
    env.reset(seed=seed)
    return env


def obs_to_tensor(obs_dict, device):
    img = obs_dict["image"]
    x   = torch.from_numpy(img.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


def record(condition: str, n_episodes: int, device: str, seed_start: int = 9999):
    ckpt_path = Path(f"results/abm/checkpoint_{condition}.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run the experiment first: python abm_experiment.py --condition {condition}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Rebuild encoder and agent
    hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE)

    if condition == "ppo_only":
        flat_dim = ckpt["flat_dim"]
        agent    = PPOAgent(latent_dim=flat_dim, n_actions=N_ACTIONS, hidden=hidden_size).to(device)
        agent.load_state_dict(ckpt["agent"])

        def encoder(obs_dict):
            img = obs_dict["image"]
            x   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return x.flatten().unsqueeze(0).to(device)
    else:
        latent_dim = ckpt.get("latent_dim", LATENT_DIM)
        lewm  = LeWM(latent_dim=latent_dim, n_actions=N_ACTIONS).to(device)
        lewm.load_state_dict(ckpt["lewm"])
        lewm.eval()
        agent = PPOAgent(latent_dim=latent_dim, n_actions=N_ACTIONS, hidden=hidden_size).to(device)
        agent.load_state_dict(ckpt["agent"])

        def encoder(obs_dict):
            return lewm.encode(obs_to_tensor(obs_dict, device))

    agent.eval()

    video_dir = Path(f"results/abm/videos/{condition}")
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recording {n_episodes} episodes → {video_dir}/")

    successes = 0
    for ep in range(n_episodes):
        seed = seed_start + ep
        env  = make_record_env(video_dir, f"{condition}_ep{ep:03d}", seed=seed)
        obs, _ = env.reset(seed=seed)
        lstm_state = agent.get_initial_state(1, device)
        done_t   = torch.zeros(1, device=device)
        done     = False
        ep_steps = 0
        ep_ret   = 0.0

        while not done and ep_steps < 300:
            with torch.no_grad():
                z = encoder(obs)
                action, _, _, _, lstm_state = agent.get_action_and_value(
                    z, lstm_state, done_t
                )
            obs, reward, term, trunc, _ = env.step(action.item())
            done      = term or trunc
            done_t    = torch.tensor([float(done)], device=device)
            ep_ret   += reward
            ep_steps += 1

        success = term and ep_ret > 0.5
        if success:
            successes += 1
        env.close()
        print(f"  Episode {ep+1}/{n_episodes}: steps={ep_steps}, return={ep_ret:.2f}, {'SUCCESS' if success else 'fail'}")

    print(f"\nSuccess rate: {successes}/{n_episodes} = {successes/n_episodes:.1%}")
    print(f"Videos saved to: {video_dir}/")
    print("Open with: open results/abm/videos/{condition}/  (macOS)")


def main():
    parser = argparse.ArgumentParser(description="Record agent episode videos")
    parser.add_argument("--condition", required=True,
                        choices=["autonomous", "fixed", "ppo_only"])
    parser.add_argument("--n",      type=int, default=5,    help="Number of episodes to record")
    parser.add_argument("--device", default="auto",          help="auto | cpu | cuda | mps")
    parser.add_argument("--seed",   type=int, default=9999,  help="Starting seed")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    record(args.condition, args.n, args.device, seed_start=args.seed)


if __name__ == "__main__":
    main()
