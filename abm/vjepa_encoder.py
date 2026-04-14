"""
abm/vjepa_encoder.py — DINOv2 ViT-B/14 frozen encoder for the A-B-M loop.

DINOv2 is a JEPA-class encoder: self-supervised, no pixel reconstruction,
trained on massive passive observation (1.2B images). Yann LeCun explicitly
calls DINOv2 "probably the best image encoder that we have at the moment"
and identifies it as a joint embedding method — the same class as V-JEPA.

Why DINOv2 instead of V-JEPA 2.1:
    V-JEPA 2.1 in single-frame (T=1) mode produces collapsed features:
    cos_sim(black_image, white_image) = 0.9969 regardless of pooling strategy.
    The feature anisotropy is fundamental to its image-mode representation.
    DINOv2 uses self-distillation with local/global crops, producing a CLS
    token explicitly trained for discriminative single-image encoding.

Output: 768-dim (identical to V-JEPA ViT-B) — zero changes to VJEPAPredictor.

Usage:
    encoder = VJEPAEncoder(device="cuda")
    z = encoder.encode(obs_dict)         # (B, 768)
    z = encoder.encode_single(obs_dict)  # (1, 768)

Input:  (B, H, W, 3) uint8 RGB or (B, 3, H, W) float32 [0, 1]
Output: (B, 768) float32 — DINOv2 CLS token (L2 normalized)
"""

import torch
import torch.nn.functional as F
import numpy as np


class VJEPAEncoder:
    """
    Frozen DINOv2 ViT-B/14 encoder (drop-in replacement for V-JEPA).

    Loads DINOv2 ViT-B/14 (86M params) via torch.hub.
    Input images are resized to 224x224 and normalized.
    Output is the L2-normalized CLS token: (B, 768).
    """

    # ImageNet normalization (same as V-JEPA — both use standard ImageNet stats)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        device: str = "cuda",
        img_size: int = 224,
    ):
        self.device   = device
        self.img_size = img_size

        # Load DINOv2 ViT-B/14 — discriminative JEPA-class encoder
        # Yann LeCun (AI Alliance 2026): "probably the best image encoder we have"
        print("Loading DINOv2 ViT-B/14 (JEPA-class frozen encoder)...")
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
        )
        self.encoder = self.encoder.to(device).eval()

        # Freeze all parameters — never trains
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.feature_dim = 768  # DINOv2 ViT-B CLS token dim

        # Cache normalization tensors on device
        self._mean = self.MEAN.to(device)
        self._std  = self.STD.to(device)

        # Sanity check: different images should produce different features
        with torch.no_grad():
            black = torch.zeros(1, 3, img_size, img_size, device=device)
            white = torch.ones(1, 3, img_size, img_size, device=device)
            noise1 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            noise2 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            all_inputs = torch.cat([black, white, noise1, noise2], dim=0)
            all_inputs = (all_inputs - self._mean) / self._std
            feat = self.encoder.forward_features(all_inputs)["x_norm_clstoken"]
            feat = F.normalize(feat, p=2, dim=-1)
            bw_sim   = F.cosine_similarity(feat[0:1], feat[1:2]).item()
            n_sim    = F.cosine_similarity(feat[2:3], feat[3:4]).item()
            feat_std = feat.std().item()
            print(f"  DINOv2 loaded — feature_dim={self.feature_dim}")
            print(f"  Feature sanity: overall_std={feat_std:.4f}")
            print(f"    cos_sim(black,white)={bw_sim:.4f}, cos_sim(noise1,noise2)={n_sim:.4f}")
            if bw_sim > 0.90:
                print("  WARNING: cos_sim still high — unexpected for DINOv2!")
            else:
                print("  Features are discriminative — good to train.")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Resize to 224x224 and normalize.
        Input: (B, 3, H, W) float32 [0, 1]
        Output: (B, 3, 224, 224) float32 normalized
        """
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(
                imgs, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        return (imgs - self._mean) / self._std

    def _obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        """
        Convert observation dict to (B, 3, H, W) float32 tensor.
        Handles both batch (N, H, W, C) and single (H, W, C) observations.
        """
        imgs = obs_dict.get("rgb", obs_dict.get("image"))
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[np.newaxis]  # (H, W, C) → (1, H, W, C)
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            x = imgs
        return x.to(self.device)

    @torch.no_grad()
    def encode(self, obs_dict: dict) -> torch.Tensor:
        """
        Encode a batch of observations.
        obs_dict: {"rgb": (B, H, W, 3) uint8}
        Returns: (B, 768) L2-normalized CLS token
        """
        x = self._obs_to_tensor(obs_dict)
        x = self._preprocess(x)
        out = self.encoder.forward_features(x)["x_norm_clstoken"]  # (B, 768)
        return F.normalize(out, p=2, dim=-1)

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
        out = self.encoder.forward_features(x)["x_norm_clstoken"]
        return F.normalize(out, p=2, dim=-1)
