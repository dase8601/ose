# Changelog

## 2026-04-18 — Habitat PointNav environment integration

### Why
dm_control walker-walk is a locomotion task, not a goal-reaching task. Goal-conditioned
MPC (DINO-WM style) naturally fits navigation — "get from A to B." Walker-walk also
collected zero goals (random actions → reward=0 → empty goal buffer → MPC defaults to
random). Habitat PointNav provides photorealistic indoor navigation with explicit goal
positions, discrete actions, and DINOv2-discriminative RGB observations.

### Changes
- `abm/habitat_env.py` — Added `get_goal_obs()` to HabitatPointNavSimpleEnv (teleports
  agent to goal, renders, restores state). Added "image" key alias to obs dicts.
- `abm/loop.py` — Added habitat env_type config block, `eval_habitat_mpc()` function,
  habitat goal pre-seeding block, and eval dispatch branch.
- `abm_experiment.py` — Added "habitat" to --env choices, plot configs, condition selection.
- `setup_cloud.sh` — Added habitat-sim installation with conda fallback + test scene download.

## 2026-04-18 — Verbose logging for training diagnostics

### Changes
- `abm/loop.py` — Added detail log line at each eval checkpoint: replay buffer size,
  goal buffer size, predictor loss, predictor z_pred_std, cos_sim, MPC status,
  goal availability, and System M internal state (ssl_buf size, sr_buf size, time in mode).
  Also logs first goal collection event during both OBSERVE and ACT modes.
- `abm/meta_controller.py` — Added logging to AutonomousSystemM: plateau detection
  values (h1, h2, rel_change vs threshold), mode switch announcements, and
  early-return reasons (waiting for min_initial_observe, not enough data).
- `abm/lewm.py` — VJEPAPredictor now tracks _last_z_pred_std and _last_cos_sim
  for external logging, and returns cos_sim_mean in the info dict.

## 2026-04-18 — Fix: Cosine distance loss + action_repeat=4 (ssl_ewa=0.0001 bug)

### Why
Two dm_control runs failed: ssl_ewa=0.0001 from step 0, success stuck at 4-5% (random).
Root cause: MSE loss on L2-normalized 768-dim vectors is always near-zero.
Even with cos_sim=0.99 between consecutive frames, MSE = 2(1-0.99)/768 = 0.000026.
The predictor trivially learns the identity function. The switching controller sees
a "plateaued" loss and switches to ACT prematurely with an untrained world model.

action_repeat=2 alone didn't help because MSE is fundamentally the wrong loss metric
for unit-normalized high-dimensional vectors.

### Changes
- `abm/lewm.py` — VJEPAPredictor: switch loss from MSE to cosine distance (1 - cos_sim).
  Cosine distance is scale-invariant and dimension-invariant. Range [0, 2] instead of
  [0, ~0.00003]. Also updated intrinsic_reward to use cosine distance.
- `abm/dmcontrol_env.py` — Increase action_repeat from 2 to 4. Each action executes for
  4 physics steps, creating larger visual differences between consecutive observations.
- `abm/loop.py` — Update ssl_freeze_thr from 0.02 to 0.05 for cosine distance scale.

---

## 2026-04-18 — Architecture upgrade: patch features + CEM planner + dm_control

### Why
MiniWorld (cos_sim=0.997) and ARC-AGI (cos_sim=0.995) both failed because DINOv2
CLS token is too coarse — maps all frames to nearly identical directions. Predictor
learned "z_next ≈ z_current" instantly (ssl_ewa=0.0001), MPC degraded to random.

Per "What Drives Success in JEPA-WMs" (Terver et al., Jan 2026 — Yann's team):
- CLS token is wrong → use patch features (DINO-WM uses all patch tokens)
- Random shooting is wrong → CEM L2 is optimal planning algorithm
- MLP predictor is suboptimal → ViT with AdaLN is best (future upgrade)

### Changes

#### `abm/vjepa_encoder.py` — Patch features instead of CLS token
- Switched from `x_norm_clstoken` (768-dim) to `x_norm_patchtokens` (256 patches × 768-dim)
- Pool patches via concat(mean_pool, max_pool) → 1536-dim output
- Preserves spatial information: max_pool captures most distinctive patches
- Sanity check updated for patch features

#### `abm/mpc.py` — CEM planner replaces Random Shooting
- New `CEMPlanner` class: iteratively refines action distributions
- K=256 candidates, E=32 elites, 3 CEM iterations per planning step
- Categorical distribution over discrete actions, Laplace smoothing
- `RandomShootingMPC` kept as alias for backward compatibility

#### `abm/lewm.py` — Updated default feature dims
- `VJEPAPredictor` default: feature_dim=1536 (was 768)
- `VJEPAReplayBuffer` default: feature_dim=1536 (was 768)

#### `abm/dmcontrol_env.py` — New: DeepMind Control Suite wrapper
- Gymnasium wrapper for dm_control tasks (walker-walk, cartpole-swingup, etc.)
- Discretizes continuous actions into fixed primitives for MPC compatibility
- Camera-rendered RGB observations at configurable resolution
- Same interface as miniworld_env.py: {"image": ..., "rgb": ...}

#### `abm/loop.py` — dm_control integration
- Added `eval_dmcontrol_mpc()` function (reward-based evaluation)
- Added `dmcontrol` env_type config block
- Updated MPC initialization to use CEMPlanner
- Goal buffer pre-seeding via random rollouts (collect r > 0.8 states)
- Updated latent_dim from 768 to 1536 for miniworld and dmcontrol

#### `test_dmcontrol_dinov2.py` — New: DINOv2 discrimination test
- Tests both patch features and CLS token for comparison
- Reports consecutive, distant, and overall cosine similarities
- Saves sample frames as PNG

#### `setup_cloud.sh` — Added dm_control + mujoco install

---

## 2026-04-17 — Option A: Fix goal specification — DINO-WM style explicit goal image

### Why
Paper 3 MPC experiment (all 4 conditions scored 20-30%) showed MPC was never
actually planning. GoalBuffer was empty — random exploration rarely reaches the
goal — so MPC fell back to random actions every step. We were testing broken MPC,
not Yann's approach.

Yann (Harvard 2026): "You can show that you can use this to get a robot to
accomplish a task zero shot. There's no training whatsoever. No RL."
This requires an explicit goal image. We weren't providing one.

### Changes
- `abm/miniworld_env.py` — Add `get_goal_obs()` method to `MiniWorldNavEnv`.
  Teleports agent to face the goal box (inner.box), captures DINOv2-ready
  observation from agent's perspective, restores agent state. No side effects.
- `abm/loop.py` — Pre-seed GoalBuffer before training loop starts. Creates n_envs
  temporary envs (one per training seed), captures goal encoding from each,
  pushes to goal_buf. MPC has valid z_goal from step 0, not after accidental discovery.
- `abm/loop.py` / `eval_miniworld_mpc` — Per-episode goal in eval. Each eval episode
  calls `env.get_goal_obs()` after reset to capture that maze's specific goal
  (eval seeds differ from training seeds, so positions vary). Falls back to
  goal_buf only if teleport fails.

---

## 2026-04-16 — Paper 3 pivot: Replace PPO with Random Shooting MPC (DINO-WM style)

### Why
Paper 2 confirmed Yann LeCun's recommendation empirically: autonomous A-B-M hit 35%
with only 18K ACT steps; PPO-only needed 410K steps to reach 35%. RL was the bottleneck,
not the world model. Yann's Harvard slide: "Abandon RL → model-predictive control."
DINO-WM (his team's paper) beats DreamerV3 4x with zero RL — same architecture we already have.

### Changes

#### `abm/mpc.py` — New file
- `RandomShootingMPC`: batched MPC planner in DINOv2 representation space.
  Samples K=256 random action sequences of horizon H=7, rolls each through
  the trained VJEPAPredictor, selects sequence minimizing ||z_H - z_goal||².
  Executes first action (receding horizon), replans every step.
  Batched over all N_ENVS simultaneously for GPU efficiency.
- `GoalBuffer`: rolling buffer (max 100) of DINOv2 goal encodings.
  Populated passively when reward>0 (agent accidentally reaches goal during
  random OBSERVE exploration). Provides z_goal to MPC during ACT phase.

#### `abm/loop.py`
- ACT block (miniworld + use_vjepa): replaced PPO with MPC planning.
  `agent=None`, `ppo=None`, `buf_ppo=None` in miniworld path.
  MPC lazy-initialized after predictor is warm (buf_vjepa ≥ LEWM_WARMUP).
- OBSERVE block: passive goal collection when reward>0 during random exploration.
- Added `eval_miniworld_mpc()`: evaluates MPC planner on N=20 episodes,
  falls back to random actions if goal buffer is empty.
- Periodic eval: routes miniworld to `eval_miniworld_mpc` when `use_vjepa=True`.
- New conditions: `mpc_only` (always ACT, no world model training — random MPC)
  and `random` (pure random baseline). Both always stay in ACT mode.
- `steps_to_80` threshold: 50% for miniworld, 80% for doorkey/crafter.
- `mode_str` init and encoder freeze guard updated for new conditions.

#### `abm_experiment.py`
- Added `mpc_only` and `random` to COLORS, LABELS, conditions lists.
- `--all` with `--env miniworld` now runs Paper 3 conditions:
  autonomous, fixed, mpc_only, random.
- Plot functions use `COLORS.get()` / `LABELS.get()` to handle all conditions.
- HTML report description updated to reflect MPC (Yann: abandon RL, use MPC).
- Bar chart and results summary include all four conditions.
- Steps milestone label: "steps_to_50" for miniworld.

---

## 2026-04-15 — Add act_steps / observe_steps tracking for experimental fairness

### Why
Ryan (PhD ME) raised a valid scientific critique: if autonomous condition spent far
fewer steps in ACT mode than fixed, the comparison is unfair (less PPO training).
Need to report ACT-phase step counts alongside success rates in Paper 2.

### Changes
- `abm/loop.py` — Add `act_steps` and `observe_steps` counters, incremented in
  each respective branch. Logged at every eval checkpoint. Returned in metrics dict.
- `abm_experiment.py` — HTML report table and terminal summary now show
  `act_steps (% of total budget)` per condition.

---

## 2026-04-14 02:15 — Final encoder: DINOv2 ViT-B/14 (DINOv3 weights 403 Forbidden)

DINOv3 fully gated — 403 at both HuggingFace and dl.fbaipublicfiles.com.
DINOv2: public weights, confirmed discriminative (cos_sim black/white=0.62),
Yann calls it "the best image encoder we have." Final encoder for Paper 2.

---

## 2026-04-14 02:00 — DINOv3 ViT-B/16 via torch.hub (public, no auth)

### Why
DINOv3 via HuggingFace transformers was gated (401). Root cause: we used the HF model
card URL. Fix: use torch.hub with GitHub source + weights from dl.fbaipublicfiles.com —
both public, no authentication required.

### Changes
- `abm/vjepa_encoder.py` — Rewrite for DINOv3 via `torch.hub.load('facebookresearch/dinov3',
  'dinov3_vitb16', pretrained=True)`. API identical to DINOv2: `forward_features(x)["x_norm_clstoken"]`.
  Input 224×224, output 768-dim L2-normalized CLS token. Weights: LVD1689M (1.7B images).
- `abm/loop.py` — Update log message to DINOv3.

---

## 2026-04-14 01:45 — Revert DINOv3 → DINOv2 (gated HuggingFace repo)

DINOv3 requires manual HuggingFace approval (401 Unauthorized). Reverting
to DINOv2 which is freely available, already confirmed discriminative
(cos_sim(black,white)=0.62), and ready to run.

---

## 2026-04-14 01:30 — Upgrade DINOv2 → DINOv3 ViT-B/16

### Why
DINOv3 is Meta's strongest universal vision backbone: 6x larger training run,
1.7B images (LVD-1689M), first SSL model to outperform weakly-supervised models.
ViT-B/16 has same 86M params and 768-dim output — zero changes to predictor/loop.
Loaded via HuggingFace transformers (pooler_output = CLS token).

### Changes
- `abm/vjepa_encoder.py` — Rewrite to use DINOv3 via transformers AutoModel.
- `setup_cloud.sh` — Add `transformers` to pip install line.
- `abm/loop.py` — Update log message.

---

## 2026-04-14 01:00 — Stage 2: Replace V-JEPA with DINOv2 ViT-B/14

### Why
Stage 1 (max pool + L2 normalize) did not fix feature collapse: cos_sim(black,white)
remained 0.9969. The anisotropy is fundamental to V-JEPA's single-frame image mode.

### Fix
- `abm/vjepa_encoder.py` — Complete rewrite. Use DINOv2 ViT-B/14 (86M params) via
  torch.hub. Input resized to 224x224, output is L2-normalized CLS token (768-dim).
  DINOv2 is a JEPA-class encoder — Yann LeCun: "probably the best image encoder we have."
  Zero changes to VJEPAPredictor or training loop (same 768-dim output).
- `abm/loop.py` — Update log message to reflect DINOv2.

### Thesis alignment
DINOv2 satisfies the JEPA requirement: self-supervised, no pixel reconstruction,
trained on massive passive observation (1.2B images). Per Yann's AI Alliance talk,
it is explicitly a joint embedding method — the same architectural class as V-JEPA.

---

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
