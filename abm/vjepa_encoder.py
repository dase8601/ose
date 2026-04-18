"""
abm/vjepa_encoder.py — DINOv2 ViT-B/14 frozen encoder for the A-B-M loop.

Key fix (2026-04-17): switched from CLS token to PATCH FEATURES.

CLS token produced cos_sim ≈ 0.997 between different MiniWorld frames and
≈ 0.995 between different ARC-AGI grids. The predictor learned nothing
(ssl_ewa = 0.0001) because all states mapped to the same direction.

Per "What Drives Success in JEPA-WMs" (Terver et al., Jan 2026 — Yann's team):
"DINO encoders substantially outperformed V-JEPA encoders due to superior
fine-grained object segmentation capabilities" — and they use PATCH features,
not CLS token. DINO-WM also operates on all patch tokens.

New output: concat(mean_pool, max_pool) of patch tokens = 1536-dim.
Mean pool captures global scene; max pool captures most distinctive patches.
Much more discriminative than CLS token alone.

Usage:
    encoder = VJEPAEncoder(device="cuda")
    z = encoder.encode(obs_dict)         # (B, 1536)
    z = encoder.encode_single(obs_dict)  # (1, 1536)
"""

import torch
import torch.nn.functional as F
import numpy as np


class VJEPAEncoder:
    """
    Frozen DINOv2 ViT-B/14 encoder using PATCH features.

    Loads DINOv2 ViT-B/14 (86M params) via torch.hub.
    Input images are resized to 224x224 and normalized.
    Output: concat(mean_pool, max_pool) of patch tokens → (B, 1536).

    224/14 = 16 patches per side → 256 patch tokens, each 768-dim.
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, device: str = "cuda", img_size: int = 224):
        self.device   = device
        self.img_size = img_size

        print("Loading DINOv2 ViT-B/14 (patch features — mean+max pool)...")
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
        )
        self.encoder = self.encoder.to(device).eval()

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self._patch_dim = 768
        self.feature_dim = 768 * 2   # concat(mean_pool, max_pool)

        self._mean = self.MEAN.to(device)
        self._std  = self.STD.to(device)

        # Sanity check — patch features should be MUCH more discriminative
        with torch.no_grad():
            black  = torch.zeros(1, 3, img_size, img_size, device=device)
            white  = torch.ones(1, 3, img_size, img_size, device=device)
            noise1 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            noise2 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            all_in = torch.cat([black, white, noise1, noise2])
            feats  = self._encode_raw(all_in)
            bw_sim = F.cosine_similarity(feats[0:1], feats[1:2]).item()
            n_sim  = F.cosine_similarity(feats[2:3], feats[3:4]).item()
            print(f"  DINOv2 loaded — feature_dim={self.feature_dim} (patch mean+max pool)")
            print(f"    cos_sim(black,white)={bw_sim:.4f}, cos_sim(noise1,noise2)={n_sim:.4f}")
            if bw_sim > 0.90:
                print("  WARNING: patch features still not discriminative!")
            else:
                print("  Patch features are discriminative — good to train.")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False)
        return (imgs - self._mean) / self._std

    def _encode_raw(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images to patch-pooled features."""
        x = self._preprocess(imgs)
        out = self.encoder.forward_features(x)
        patches = out["x_norm_patchtokens"]  # (B, N_patches, 768)

        mean_pool = patches.mean(dim=1)                  # (B, 768)
        max_pool  = patches.max(dim=1).values            # (B, 768)
        combined  = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 1536)
        return F.normalize(combined, p=2, dim=-1)

    def _obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        imgs = obs_dict.get("rgb", obs_dict.get("image"))
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[np.newaxis]
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2)
        else:
            x = imgs
        return x.to(self.device)

    @torch.no_grad()
    def encode(self, obs_dict: dict) -> torch.Tensor:
        x = self._obs_to_tensor(obs_dict)
        return self._encode_raw(x)

    @torch.no_grad()
    def encode_single(self, obs_dict: dict) -> torch.Tensor:
        return self.encode(obs_dict)

    @torch.no_grad()
    def encode_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._encode_raw(imgs)
