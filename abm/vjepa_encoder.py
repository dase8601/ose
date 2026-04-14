"""
abm/vjepa_encoder.py — DINOv3 ViT-B/16 frozen encoder for the A-B-M loop.

DINOv3 is Meta's strongest universal vision backbone (2025):
- 6x larger training run than DINOv2, 1.7B images (LVD-1689M dataset)
- First SSL model to outperform weakly-supervised models on dense probing tasks
- ViT-B/16 variant: 86M params, 768-dim output, includes register tokens

DINOv3 is a JEPA-class encoder: self-supervised, no pixel reconstruction,
trained on massive passive observation. Yann LeCun (AI Alliance 2026) identifies
joint embedding methods as the foundation of world models and explicitly cites
DINOv2/DINOv3-class architectures as the best image encoders available.

Output: 768-dim (identical to V-JEPA ViT-B) — zero changes to VJEPAPredictor.

Usage:
    encoder = VJEPAEncoder(device="cuda")
    z = encoder.encode(obs_dict)         # (B, 768)
    z = encoder.encode_single(obs_dict)  # (1, 768)

Input:  (B, H, W, 3) uint8 RGB or (B, 3, H, W) float32 [0, 1]
Output: (B, 768) float32 — DINOv3 pooled CLS token (L2 normalized)
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class VJEPAEncoder:
    """
    Frozen DINOv3 ViT-B/16 encoder (drop-in replacement for V-JEPA / DINOv2).

    Uses HuggingFace transformers to load DINOv3 ViT-B/16.
    Output is the L2-normalized pooled CLS token: (B, 768).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        print(f"Loading DINOv3 ViT-B/16 from {MODEL_ID}...")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.encoder = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

        # Freeze all parameters — never trains
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.feature_dim = 768  # ViT-B CLS token dim

        # Sanity check: different images should produce different features
        with torch.no_grad():
            black = torch.zeros(1, 3, 224, 224)
            white = torch.ones(1, 3, 224, 224)
            noise1 = torch.randn(1, 3, 224, 224).clamp(0, 1)
            noise2 = torch.randn(1, 3, 224, 224).clamp(0, 1)

            def _encode_raw(imgs):
                # imgs: (B, 3, H, W) float [0,1] → numpy (B, H, W, 3) uint8
                imgs_np = (imgs.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
                imgs_list = [imgs_np[i] for i in range(len(imgs_np))]
                inputs = self.processor(images=imgs_list, return_tensors="pt").to(device)
                out = self.encoder(**inputs).pooler_output  # (B, 768)
                return F.normalize(out, p=2, dim=-1)

            bw_feat = _encode_raw(torch.cat([black, white]))
            n_feat  = _encode_raw(torch.cat([noise1, noise2]))
            bw_sim  = F.cosine_similarity(bw_feat[0:1], bw_feat[1:2]).item()
            n_sim   = F.cosine_similarity(n_feat[0:1],  n_feat[1:2]).item()
            print(f"  DINOv3 loaded — feature_dim={self.feature_dim}")
            print(f"    cos_sim(black,white)={bw_sim:.4f}, cos_sim(noise1,noise2)={n_sim:.4f}")
            if bw_sim > 0.90:
                print("  WARNING: cos_sim still high — features may not be discriminative!")
            else:
                print("  Features are discriminative — good to train.")

    def _to_pil_list(self, obs_dict: dict):
        """Convert obs dict to list of uint8 numpy arrays for processor."""
        imgs = obs_dict.get("rgb", obs_dict.get("image"))
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[np.newaxis]  # (H, W, C) → (1, H, W, C)
            return [imgs[i] for i in range(len(imgs))]
        else:
            # tensor (B, C, H, W) or (B, H, W, C)
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = imgs.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
            imgs_np = (imgs.cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            return [imgs_np[i] for i in range(len(imgs_np))]

    @torch.no_grad()
    def encode(self, obs_dict: dict) -> torch.Tensor:
        """
        Encode a batch of observations.
        obs_dict: {"rgb": (B, H, W, 3) uint8}
        Returns: (B, 768) L2-normalized CLS token
        """
        imgs_list = self._to_pil_list(obs_dict)
        inputs = self.processor(images=imgs_list, return_tensors="pt").to(self.device)
        out = self.encoder(**inputs).pooler_output  # (B, 768)
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def encode_single(self, obs_dict: dict) -> torch.Tensor:
        """Encode a single observation. Returns (1, 768)."""
        return self.encode(obs_dict)

    @torch.no_grad()
    def encode_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode pre-processed image tensors directly.
        imgs: (B, 3, H, W) float32 [0, 1]
        Returns: (B, 768)
        """
        imgs_np = (imgs.permute(0, 2, 3, 1).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
        imgs_list = [imgs_np[i] for i in range(len(imgs_np))]
        inputs = self.processor(images=imgs_list, return_tensors="pt").to(self.device)
        out = self.encoder(**inputs).pooler_output
        return F.normalize(out, p=2, dim=-1)
