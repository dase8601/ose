"""
record_episodes.py — Record MP4 video of trained agent episodes.

Loads a checkpoint from results/{env}/checkpoint_{condition}.pt,
runs N greedy episodes, and saves MP4s to results/{env}/videos/{condition}/.

Usage:
    # DoorKey
    python record_episodes.py --condition fixed --n 5
    python record_episodes.py --condition autonomous --n 10 --filter success
    python record_episodes.py --condition ppo_only --n 5 --filter fail

    # Crafter
    python record_episodes.py --condition autonomous --env crafter --n 5
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


# ---------------------------------------------------------------------------
# DoorKey env factory (with video recording)
# ---------------------------------------------------------------------------

def make_doorkey_record_env(video_dir: Path, episode_prefix: str, seed: int = 0):
    base = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    base = ShapedRewardWrapper(base)
    base = RecordVideo(
        base,
        video_folder=str(video_dir),
        name_prefix=episode_prefix,
        episode_trigger=lambda ep: True,
    )
    env = RGBImgObsWrapper(base, tile_size=8)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Crafter env factory (with video recording)
# ---------------------------------------------------------------------------

def make_crafter_record_env(video_dir: Path, episode_prefix: str, seed: int = 0):
    from abm.crafter_env import CrafterEnv

    class _CrafterRGB(CrafterEnv):
        """Crafter with render_mode metadata for RecordVideo compatibility."""
        metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

        def render(self):
            frame = self._env.render()
            if frame is None:
                # crafter.Env.render() returns None until first reset in some versions
                return np.zeros((64, 64, 3), dtype=np.uint8)
            return frame

    env = _CrafterRGB(seed=seed)
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=episode_prefix,
        episode_trigger=lambda ep: True,
    )
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def obs_to_tensor(obs_dict, device):
    img = obs_dict["image"]
    x   = torch.from_numpy(img.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Main recording logic
# ---------------------------------------------------------------------------

def record(
    condition:   str,
    n_episodes:  int,
    device:      str,
    env_type:    str = "doorkey",
    seed_start:  int = 9999,
    ep_filter:   str = "all",   # "all" | "success" | "fail"
):
    ckpt_path = Path(f"results/{env_type}/checkpoint_{condition}.pt")
    if not ckpt_path.exists():
        # Fall back to legacy results/abm/ path
        ckpt_path = Path(f"results/abm/checkpoint_{condition}.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Run: python abm_experiment.py --condition {condition} --env {env_type}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    n_actions   = ckpt.get("n_actions", 7 if env_type == "doorkey" else 17)
    hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE)

    if condition == "ppo_only":
        flat_dim = ckpt["flat_dim"]
        agent    = PPOAgent(latent_dim=flat_dim, n_actions=n_actions, hidden=hidden_size).to(device)
        agent.load_state_dict(ckpt["agent"])

        def encoder(obs_dict):
            img = obs_dict["image"]
            x   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return x.flatten().unsqueeze(0).to(device)
    else:
        latent_dim = ckpt.get("latent_dim", LATENT_DIM)
        lewm  = LeWM(latent_dim=latent_dim, n_actions=n_actions).to(device)
        lewm.load_state_dict(ckpt["lewm"])
        lewm.eval()
        agent = PPOAgent(latent_dim=latent_dim, n_actions=n_actions, hidden=hidden_size).to(device)
        agent.load_state_dict(ckpt["agent"])

        def encoder(obs_dict):
            return lewm.encode(obs_to_tensor(obs_dict, device))

    agent.eval()

    video_dir = Path(f"results/{env_type}/videos/{condition}")
    video_dir.mkdir(parents=True, exist_ok=True)

    max_steps = 300 if env_type == "doorkey" else 10_000
    make_env  = (make_doorkey_record_env if env_type == "doorkey"
                 else make_crafter_record_env)

    print(f"Recording up to {n_episodes} episodes (filter={ep_filter}) → {video_dir}/")

    recorded   = 0
    attempt    = 0
    successes  = 0

    while recorded < n_episodes:
        seed = seed_start + attempt
        attempt += 1
        prefix = f"{condition}_ep{recorded:03d}"
        env    = make_env(video_dir, prefix, seed=seed)
        obs, _ = env.reset(seed=seed)

        lstm_state = agent.get_initial_state(1, device)
        done_t   = torch.zeros(1, device=device)
        done     = False
        ep_steps = 0
        ep_ret   = 0.0
        ep_achievements = set()

        while not done and ep_steps < max_steps:
            with torch.no_grad():
                z = encoder(obs)
                action, _, _, _, lstm_state = agent.get_action_and_value(
                    z, lstm_state, done_t
                )
            obs, reward, term, trunc, info = env.step(action.item())
            done      = term or trunc
            done_t    = torch.tensor([float(done)], device=device)
            ep_ret   += reward
            ep_steps += 1

            # Track Crafter achievements
            for k, v in info.get("achievements", {}).items():
                if v:
                    ep_achievements.add(k)

        success = (ep_ret > 0.5) if env_type == "doorkey" else (len(ep_achievements) > 0)
        env.close()

        # Apply filter — delete the video file if it doesn't match
        if ep_filter == "success" and not success:
            for f in video_dir.glob(f"{prefix}*"):
                f.unlink(missing_ok=True)
            continue
        if ep_filter == "fail" and success:
            for f in video_dir.glob(f"{prefix}*"):
                f.unlink(missing_ok=True)
            continue

        recorded += 1
        if success:
            successes += 1

        if env_type == "doorkey":
            status = "SUCCESS" if success else "fail"
            print(f"  Episode {recorded}/{n_episodes}: steps={ep_steps}, "
                  f"return={ep_ret:.2f}, {status}")
        else:
            ach_str = ", ".join(sorted(ep_achievements)) or "none"
            print(f"  Episode {recorded}/{n_episodes}: steps={ep_steps}, "
                  f"reward={ep_ret:.1f}, achievements=[{ach_str}]")

    print(f"\nSuccess rate: {successes}/{recorded} = {successes/max(recorded,1):.1%}")
    print(f"Videos saved to: {video_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Record agent episode videos")
    parser.add_argument("--condition", required=True,
                        choices=["autonomous", "fixed", "ppo_only"])
    parser.add_argument("--env",    default="doorkey", choices=["doorkey", "crafter"])
    parser.add_argument("--n",      type=int, default=5,   help="Episodes to record")
    parser.add_argument("--filter", default="all",          choices=["all", "success", "fail"],
                        help="Only keep success, fail, or all episodes")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed",   type=int, default=9999)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}  |  Env: {args.env}  |  Filter: {args.filter}")
    record(args.condition, args.n, args.device,
           env_type=args.env, seed_start=args.seed, ep_filter=args.filter)


if __name__ == "__main__":
    main()
