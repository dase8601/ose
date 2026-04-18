"""
test_dmcontrol_dinov2.py — Test DINOv2 feature discrimination on dm_control.

This is the critical sanity check before integrating dm_control into the
A-B-M architecture. If cos_sim < 0.85, DINOv2 CLS token can distinguish
different physics states and our world model will actually learn dynamics.

MiniWorld failed (cos_sim=0.9969): consecutive 3D frames too similar.
ARC-AGI failed (cos_sim=0.9953): synthetic grids too alien for DINOv2.
dm_control should work: natural-looking physics renders are what DINOv2
was trained on (ImageNet-style images with objects, textures, poses).

Usage:
    pip install dm_control
    python test_dmcontrol_dinov2.py
    python test_dmcontrol_dinov2.py --task reacher-easy --steps 50
"""

import argparse
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="walker-walk",
                        choices=["walker-walk", "walker-stand", "cartpole-swingup",
                                 "cartpole-balance", "reacher-easy", "reacher-hard",
                                 "cheetah-run", "finger-spin"])
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of random actions to collect states")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Render resolution")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print("=" * 60)
    print(f"DINOv2 Discrimination Test — dm_control / {args.task}")
    print("=" * 60)

    # ── Step 1: Collect diverse states ───────────────────────────────────────
    print(f"\n[1] Collecting {args.steps} states from {args.task}...")

    from abm.dmcontrol_env import make_dmcontrol_env

    env = make_dmcontrol_env(task_name=args.task, seed=42, img_size=args.img_size)
    obs, _ = env.reset()
    print(f"  Action space: Discrete({env.N_ACTIONS}) (discretized from continuous)")
    print(f"  Observation: {obs['rgb'].shape} uint8")

    frames = [obs["rgb"].copy()]
    rewards = [0.0]

    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        frames.append(obs["rgb"].copy())
        rewards.append(reward)

        if term or trunc:
            obs, _ = env.reset()
            print(f"  Episode ended at step {i}, reset")

    env.close()
    print(f"  Collected {len(frames)} frames")
    print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")

    # Save a few frames
    for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]:
        img = Image.fromarray(frames[i])
        img.save(f"dmcontrol_{args.task}_{i}.png")
        print(f"  Saved: dmcontrol_{args.task}_{i}.png")

    # ── Step 2: DINOv2 encoding ─────────────────────────────────────────────
    print(f"\n[2] Encoding with DINOv2 ViT-B/14 on {args.device}...")

    import torch
    from torchvision import transforms

    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2.eval()
    dinov2 = dinov2.to(args.device)
    print("  DINOv2 loaded")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def encode_patches(frame_rgb):
        """Encode using patch features (mean+max pool) — same as our VJEPAEncoder."""
        img = Image.fromarray(frame_rgb)
        tensor = transform(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = dinov2.forward_features(tensor)
            patches = out["x_norm_patchtokens"]  # (1, N, 768)
            mean_pool = patches.mean(dim=1)       # (1, 768)
            max_pool = patches.max(dim=1).values   # (1, 768)
            feat = torch.cat([mean_pool, max_pool], dim=-1)  # (1, 1536)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat

    # Also compute CLS token for comparison
    def encode_cls(frame_rgb):
        img = Image.fromarray(frame_rgb)
        tensor = transform(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            feat = dinov2.forward_features(tensor)["x_norm_clstoken"]
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat

    features = []
    features_cls = []
    for i, frame in enumerate(frames):
        features.append(encode_patches(frame))
        features_cls.append(encode_cls(frame))

    print(f"  Encoded {len(features)} frames → {features[0].shape[-1]}-dim (patch mean+max)")
    print(f"  Also encoded CLS token → {features_cls[0].shape[-1]}-dim (for comparison)")

    # ── Step 3: Pairwise cosine similarity ──────────────────────────────────
    print(f"\n[3] Pairwise cosine similarities...")

    # Consecutive frame similarities (most important for world model)
    consec_sims = []
    for i in range(len(features) - 1):
        sim = torch.nn.functional.cosine_similarity(
            features[i], features[i+1]
        ).item()
        consec_sims.append(sim)

    print(f"\n  CONSECUTIVE frame similarities (world model signal):")
    print(f"    Mean: {np.mean(consec_sims):.4f}")
    print(f"    Min:  {np.min(consec_sims):.4f}")
    print(f"    Max:  {np.max(consec_sims):.4f}")
    print(f"    Std:  {np.std(consec_sims):.4f}")

    # Distant frame similarities (should be lower = more discriminative)
    distant_sims = []
    stride = max(1, len(features) // 10)
    for i in range(0, len(features) - stride, stride):
        sim = torch.nn.functional.cosine_similarity(
            features[i], features[i + stride]
        ).item()
        distant_sims.append(sim)

    print(f"\n  DISTANT frame similarities (stride={stride}):")
    print(f"    Mean: {np.mean(distant_sims):.4f}")
    print(f"    Min:  {np.min(distant_sims):.4f}")
    print(f"    Max:  {np.max(distant_sims):.4f}")

    # First vs last
    first_last = torch.nn.functional.cosine_similarity(
        features[0], features[-1]
    ).item()
    print(f"\n  First vs Last: {first_last:.4f}")

    # Overall pairwise (subsample to avoid O(n^2) for large n)
    sample_idx = np.linspace(0, len(features)-1, min(15, len(features)), dtype=int)
    all_sims = []
    for i in range(len(sample_idx)):
        for j in range(i+1, len(sample_idx)):
            sim = torch.nn.functional.cosine_similarity(
                features[sample_idx[i]], features[sample_idx[j]]
            ).item()
            all_sims.append(sim)

    avg_sim = np.mean(all_sims)
    print(f"\n  OVERALL pairwise (subsampled {len(sample_idx)} frames):")
    print(f"    Mean: {avg_sim:.4f}")
    print(f"    Min:  {np.min(all_sims):.4f}")
    print(f"    Max:  {np.max(all_sims):.4f}")

    # ── Step 4: Verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    consec_mean = np.mean(consec_sims)
    if avg_sim < 0.85:
        print(f"\n  HIGHLY DISCRIMINATIVE (avg={avg_sim:.4f})")
        print("  DINOv2 CLS token distinguishes states well.")
        print("  → World model predictor WILL learn meaningful dynamics")
        print("  → MPC planning WILL differentiate action sequences")
        print("  → This is the right environment for our architecture")
    elif avg_sim < 0.95:
        print(f"\n  MODERATELY DISCRIMINATIVE (avg={avg_sim:.4f})")
        print("  Some signal — predictor can learn but may need more data")
    else:
        print(f"\n  POOR (avg={avg_sim:.4f})")
        print("  Same collapse problem as MiniWorld/ARC-AGI")
        print("  → DINOv2 CLS token can't distinguish these states")

    print(f"\n  Consecutive similarity: {consec_mean:.4f}")
    if consec_mean < 0.95:
        print("  → Adjacent frames are distinguishable (good for ssl_ewa)")
    else:
        print("  → Adjacent frames too similar (ssl_ewa will collapse to ~0)")

    # CLS token comparison
    cls_sims = []
    for i in range(len(sample_idx)):
        for j in range(i+1, len(sample_idx)):
            sim = torch.nn.functional.cosine_similarity(
                features_cls[sample_idx[i]], features_cls[sample_idx[j]]
            ).item()
            cls_sims.append(sim)
    cls_avg = np.mean(cls_sims)

    print(f"""
  Comparison (patch features vs CLS token):
    Patch mean+max (1536-dim): avg_sim = {avg_sim:.4f}  {'(PASS)' if avg_sim < 0.95 else '(FAIL)'}
    CLS token (768-dim):       avg_sim = {cls_avg:.4f}  {'(PASS)' if cls_avg < 0.95 else '(FAIL)'}

  Previous failures (CLS token):
    MiniWorld:     avg_sim ≈ 0.997  (FAILED — ssl_ewa=0.0001)
    ARC-AGI grids: avg_sim ≈ 0.995  (FAILED — synthetic grids)
    """)


if __name__ == "__main__":
    main()
