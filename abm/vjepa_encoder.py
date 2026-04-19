"""
abm/vjepa_encoder.py — DINOv2 ViT-B/14 frozen encoder for the A-B-M loop.

DINOv2 CLS token discrimination results:
  - MiniWorld 3D maze:  cos_sim ≈ 0.997 (FAIL — subtle camera shifts)
  - ARC-AGI-3 grids:    cos_sim ≈ 0.995 (FAIL — synthetic colored grids)
  - dm_control walker:  cos_sim ≈ 0.846 (PASS — natural physics renders)

CLS token works on dm_control because these are natural-looking images with
dramatic pose changes — exactly what DINOv2 was trained on (142M images).
Patch mean+max pooling actually performed worse (0.975) because it dilutes
the discriminative CLS signal with redundant background patches.

Output: 768-dim L2-normalized CLS token.

Usage:
    encoder = VJEPAEncoder(device="cuda")
    z = encoder.encode(obs_dict)         # (B, 768)
    z = encoder.encode_single(obs_dict)  # (1, 768)
"""

import torch
import torch.nn.functional as F
import numpy as np


class VJEPAEncoder:
    """
    Frozen DINOv2 ViT-B/14 encoder.

    Loads DINOv2 ViT-B/14 (86M params) via timm.
    Input images are resized to 224x224 and normalized.
    Output is the L2-normalized CLS token: (B, 768).
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, device: str = "cuda", img_size: int = 224):
        self.device   = device
        self.img_size = img_size

        print("Loading DINOv2 ViT-B/14 (JEPA-class frozen encoder)...")
        import timm
        self.encoder = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )
        self.encoder = self.encoder.to(device).eval()

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.feature_dim = 768

        self._mean = self.MEAN.to(device)
        self._std  = self.STD.to(device)

        # Sanity check
        with torch.no_grad():
            black  = torch.zeros(1, 3, img_size, img_size, device=device)
            white  = torch.ones(1, 3, img_size, img_size, device=device)
            noise1 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            noise2 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            all_in = torch.cat([black, white, noise1, noise2])
            all_in = (all_in - self._mean) / self._std
            feat   = self.encoder(all_in)
            feat   = F.normalize(feat, p=2, dim=-1)
            bw_sim = F.cosine_similarity(feat[0:1], feat[1:2]).item()
            n_sim  = F.cosine_similarity(feat[2:3], feat[3:4]).item()
            print(f"  DINOv2 loaded — feature_dim={self.feature_dim}")
            print(f"    cos_sim(black,white)={bw_sim:.4f}, cos_sim(noise1,noise2)={n_sim:.4f}")
            if bw_sim > 0.70:
                print("  WARNING: features may not be discriminative on synthetic images!")
                print("  (OK for dm_control — natural physics renders are discriminative)")
            else:
                print("  Features are discriminative — good to train.")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False)
        return (imgs - self._mean) / self._std

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
        x   = self._obs_to_tensor(obs_dict)
        x   = self._preprocess(x)
        out = self.encoder(x)
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def encode_single(self, obs_dict: dict) -> torch.Tensor:
        return self.encode(obs_dict)

    @torch.no_grad()
    def encode_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        x   = self._preprocess(imgs)
        out = self.encoder(x)
        return F.normalize(out, p=2, dim=-1)
