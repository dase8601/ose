# Changelog

## 2026-04-14 00:30 — Stage 1 fix: V-JEPA max pool + L2 normalize (anti-collapse)

### Problem
V-JEPA 2.1 mean pooling over 576 patches causes feature anisotropy: cos_sim(black,white)=0.9969.
All images map to nearly identical directions in 768-dim space → 350K steps at 0-5% success.

### Fix
- `abm/vjepa_encoder.py` — Replace `mean(dim=1)` with `max(dim=1).values` everywhere.
  Max pool extracts the most activated patch per feature dimension instead of averaging everything flat.
  Then L2 normalize to remove scale-based anisotropy.
- Sanity check threshold lowered to 0.90. If still > 0.90 after this fix → Stage 2 (DINOv2).

---

## 2026-04-14 00:00 — Fix V-JEPA 5D video input + weight loading

### Fixes
- `abm/vjepa_encoder.py` — V-JEPA 2.1 expects 5D video tensors `(B, C, T, H, W)`. Added `unsqueeze(2)` to insert `T=1` temporal dimension for single-frame encoding. Fixed state_dict loading to match hub's `_clean_backbone_key` (extract `ema_encoder`, strip `module.`/`backbone.` prefixes). Added `weights_only=False` to suppress torch.load warning.

---

## 2026-04-13 23:50 — Fix V-JEPA weights download URL (localhost → public)

### Fixes
- `abm/vjepa_encoder.py` — The V-JEPA 2.1 torch.hub repo changed its weight download URL to `localhost:8300` (Meta internal). Now loads architecture with `pretrained=False` and downloads weights directly from `dl.fbaipublicfiles.com/vjepa2/`.

---

## 2026-04-13 23:40 — Add einops dependency for V-JEPA 2.1

### Fixes
- `setup_cloud.sh` — V-JEPA 2.1 hub model requires `einops` (not listed in its own deps). Added to pip install line.

---

## 2026-04-13 23:30 — Fix MiniWorld AsyncVectorEnv X crash + persistent Xvfb

### Fixes
- `abm/loop.py` — Force `use_async=False` for MiniWorld vectorized envs. AsyncVectorEnv forks 16 processes that all compete for the same X display, crashing the X server. SyncVectorEnv runs all envs in one process reliably.
- `setup_cloud.sh` — Use persistent `Xvfb :1` instead of `xvfb-run -a`. More stable for long training runs. Run commands no longer need `xvfb-run` prefix.

---

## 2026-04-13 23:10 — Fix MiniWorld headless rendering on RunPod

### Fixes
- `setup_cloud.sh` — Install OpenGL system libraries (libglu1-mesa-dev, xvfb) for MiniWorld's pyglet 3D rendering on headless GPU servers. Run commands now use `xvfb-run -a` prefix.

---

## 2026-04-13 23:00 — Replace Habitat with MiniWorld (pip-installable 3D navigation)

### Why
habitat-sim only supports Python <=3.9 via conda, requires multi-GB scene datasets
with academic registration, and has fragile installation on RunPod. MiniWorld provides
3D first-person maze navigation via `pip install miniworld` with zero friction.

### New files
- `abm/miniworld_env.py` — MiniWorld-MazeS3 Gymnasium wrapper (160x160 RGB, 3 discrete actions)

### Modified files
- `abm/loop.py` — Replaced habitat config block with miniworld; replaced `eval_habitat()` with `eval_miniworld()`
- `abm/vjepa_encoder.py` — Accept both "rgb" and "image" obs keys
- `abm_experiment.py` — `--env miniworld` replaces `--env habitat`
- `setup_cloud.sh` — Simple `pip install miniworld` replaces broken habitat-sim conda flow

---

## 2026-04-13 22:45 — Fix habitat-sim Python version conflict

### Fixes
- `setup_cloud.sh` — habitat-sim requires Python <=3.9, but RunPod has 3.11+. Script now creates a dedicated conda env with Python 3.9 when run with `source setup_cloud.sh habitat`. Standard DoorKey/Crafter mode unchanged.

---

## 2026-04-13 22:30 — Fix setup_cloud.sh for RunPod A100

### Fixes
- `setup_cloud.sh` — Fixed `total_mem` → `total_memory` typo in GPU verification script
- `setup_cloud.sh` — habitat-sim now installs via conda (auto-installs miniconda if needed) instead of broken `pip install habitat-sim-headless`
- `setup_cloud.sh` — habitat-lab install now chains fallbacks properly

---

## 2026-04-13 — V-JEPA 2.1 + Habitat PointNav (Paper 2 foundation)

**Commit:** `f10af5d` — "Add V-JEPA 2.1 + Habitat PointNav as System A for A-B-M loop"

### New files
- `abm/vjepa_encoder.py` — V-JEPA 2.1 ViT-B frozen encoder wrapper (384x384 → 768-dim features)
- `abm/habitat_env.py` — Habitat PointNav Gymnasium wrapper (2 variants: full habitat-lab, simple habitat-sim fallback)

### Modified files
- `abm/lewm.py` — Added `VJEPAPredictor` (action-conditioned MLP in 768-dim repr space) and `VJEPAReplayBuffer` (stores pre-computed features)
- `abm/loop.py` — Habitat config block, V-JEPA encode pipeline, predictor-based intrinsic reward, `eval_habitat()` function
- `abm_experiment.py` — `--env habitat` option, V-JEPA plot titles, 50% nav success target
- `setup_cloud.sh` — habitat-sim-headless, habitat-lab, omegaconf, timm installs; A100 note
