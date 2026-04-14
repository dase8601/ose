"""
abm/vjepa_encoder.py — V-JEPA 2.1 frozen encoder for the A-B-M loop.

Loads V-JEPA 2.1 ViT-B (80M params, distilled from ViT-G 2B) as System A's
perception backbone.  The encoder is permanently frozen — it was pretrained
on massive video data (Ego4D, SSv2, YT-1B, LVD-142M images), representing
"years of passive observation" in LeCun's framework.

What trains during OBSERVE is the action-conditioned predictor (in lewm.py),
not this encoder.

Usage:
    encoder = VJEPAEncoder(device="cuda")
    z = encoder.encode(obs_rgb)          # (B, 768)
    z = encoder.encode_single(obs_rgb)   # (1, 768)

Input:  (B, H, W, 3) uint8 RGB or (B, 3, H, W) float32 [0, 1]
Output: (B, 768) float32 — avg-pooled patch features
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Correct public URL — the torch.hub copy may have a broken localhost URL
VJEPA_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"


class VJEPAEncoder:
    """
    Frozen V-JEPA 2.1 ViT-B encoder.

    Loads the distilled ViT-B model (80M params) via torch.hub.
    Input images are resized to 384x384 and normalized.
    Output is average-pooled patch features: (B, 768).
    """

    # ImageNet normalization (V-JEPA uses same as ViT/DINOv2)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "vjepa2_1_vit_base_384",
        img_size: int = 384,
    ):
        self.device   = device
        self.img_size = img_size

        # Load V-JEPA 2.1 ViT-B architecture (no weights)
        # Then load weights from the correct public URL ourselves,
        # because the hub repo's URL sometimes points to localhost (Meta internal).
        self.encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2:main",
            model_name,
            pretrained=False,  # don't let hub download weights
        )

        # Download weights from the correct public URL
        cache_dir = torch.hub.get_dir()
        os.makedirs(os.path.join(cache_dir, "checkpoints"), exist_ok=True)
        ckpt_path = os.path.join(cache_dir, "checkpoints", "vjepa2_1_vitb_dist_vitG_384.pt")
        if not os.path.exists(ckpt_path):
            print(f"Downloading V-JEPA 2.1 weights from {VJEPA_WEIGHTS_URL}")
            torch.hub.download_url_to_file(VJEPA_WEIGHTS_URL, ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Extract encoder weights and clean keys (same as hub's _clean_backbone_key)
        enc_sd = state_dict["ema_encoder"]
        cleaned = {}
        for k, v in enc_sd.items():
            k = k.replace("module.", "").replace("backbone.", "")
            cleaned[k] = v

        # Log weight loading diagnostics
        model_keys = set(self.encoder.state_dict().keys())
        ckpt_keys = set(cleaned.keys())
        matched = model_keys & ckpt_keys
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        print(f"V-JEPA weight loading: {len(matched)}/{len(model_keys)} keys matched")
        if missing:
            print(f"  MISSING from checkpoint ({len(missing)}): {list(missing)[:5]}...")
        if unexpected:
            print(f"  UNEXPECTED in checkpoint ({len(unexpected)}): {list(unexpected)[:5]}...")

        result = self.encoder.load_state_dict(cleaned, strict=False)
        if result.missing_keys:
            print(f"  WARNING: {len(result.missing_keys)} missing keys — encoder may have random weights!")
        if not result.missing_keys and not result.unexpected_keys:
            print("  All weights loaded successfully.")

        self.encoder = self.encoder.to(device).eval()

        # Freeze all parameters — never trains
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Cache normalization tensors on device
        self._mean = self.MEAN.to(device)
        self._std  = self.STD.to(device)

        # Determine feature dimension by dry run
        # V-JEPA expects 5D video input: (B, C, T, H, W) — use T=1 for single frames
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 1, img_size, img_size, device=device)
            out = self.encoder(dummy)
            if out.ndim == 3:
                # (B, num_patches, embed_dim)
                self.feature_dim = out.shape[-1]
                self.num_patches = out.shape[1]
            else:
                # (B, embed_dim) — already pooled
                self.feature_dim = out.shape[-1]
                self.num_patches = 1

            # Sanity check: random noise input should produce varied features
            noise = torch.randn(2, 3, 1, img_size, img_size, device=device)
            feat = self.encoder(noise)
            if feat.ndim == 3:
                feat = feat.mean(dim=1)
            cos_sim = F.cosine_similarity(feat[0:1], feat[1:2]).item()
            feat_std = feat.std().item()
            print(f"  Feature sanity: dim={self.feature_dim}, std={feat_std:.4f}, "
                  f"cos_sim(2 random inputs)={cos_sim:.4f}")
            if feat_std < 0.001:
                print("  WARNING: Near-zero feature variance — weights likely not loaded!")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Resize to 384x384, normalize, and add temporal dim for V-JEPA.
        Input: (B, 3, H, W) float32 [0, 1]
        Output: (B, 3, 1, 384, 384) float32 normalized — T=1 for single frames
        """
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(
                imgs, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        imgs = (imgs - self._mean) / self._std
        # Add temporal dimension: (B, C, H, W) → (B, C, 1, H, W)
        return imgs.unsqueeze(2)

    def _obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        """
        Convert observation dict to (B, 3, H, W) float32 tensor.
        Handles both batch (N, H, W, C) and single (H, W, C) observations.
        """
        imgs = obs_dict.get("rgb", obs_dict.get("image"))  # support both keys
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[np.newaxis]  # (H, W, C) → (1, H, W, C)
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            x = imgs  # already a tensor
        return x.to(self.device)

    @torch.no_grad()
    def encode(self, obs_dict: dict) -> torch.Tensor:
        """
        Encode a batch of observations.
        obs_dict: {"rgb": (B, H, W, 3) uint8}
        Returns: (B, 768) avg-pooled features
        """
        x = self._obs_to_tensor(obs_dict)
        x = self._preprocess(x)
        out = self.encoder(x)
        if out.ndim == 3:
            return out.mean(dim=1)  # avg pool over patches → (B, D)
        return out

    @torch.no_grad()
    def encode_single(self, obs_dict: dict) -> torch.Tensor:
        """
        Encode a single observation (for evaluation).
        obs_dict: {"rgb": (H, W, 3) uint8}
        Returns: (1, 768)
        """
        return self.encode(obs_dict)

    @torch.no_grad()
    def encode_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode pre-processed image tensors directly.
        imgs: (B, 3, H, W) float32 [0, 1]
        Returns: (B, 768)
        """
        x = self._preprocess(imgs)
        out = self.encoder(x)
        if out.ndim == 3:
            return out.mean(dim=1)
        return out
