# Changelog

## 2026-04-27 — Run 20 built: two-phase EBM (hinge→softplus)

### Why
Run 18 (softplus throughout): peak=25%, goal=220 — EBM sustained but softer gradient limited early conversion to 14% vs Run 15c's 83%. Run 19 (margin=10): effectively 0%, margin too large for EBM to learn useful discrimination. Two-phase approach: hinge during OBSERVE builds sharp discrimination fast (same as Run 15c), softplus during ACT maintains it without ever saturating. One flag: `train_fn = _train_ebm if in_observe else _train_ebm_softplus`.

### What
- New file: `abm/loop_mpc_doorkey_run20.py` — condition `symbolic_two_phase_ebm`
- OBSERVE: `_train_ebm` (hinge margin=1) — identical to Run 15c
- ACT: `_train_ebm_softplus` (softplus) — identical to Run 18's training function
- Phase switch: single line `train_fn = _train_ebm if in_observe else _train_ebm_softplus`
- Heartbeat logs show `ON(hinge)` during OBSERVE, `ON(softplus)` during ACT
- `abm_experiment.py`: added `symbolic_two_phase_ebm`
- EXPERIMENTS.md: Run 18 final results logged (peak=25%), Run 19 closed (failed), Run 20 entry added

---

## 2026-04-27 — Run 19 built: large-margin hinge EBM (stage-3 margin=10)

### Why
Run 18 (softplus) shows pred_ewa climbing to 0.013 and real success signal (15% peak) but stage-3 conversion is ~14% vs Run 15c's 83%. Softplus gradient = sigmoid(·) ≤ 1.0 is softer than hinge's sharp gradient = 1.0. Run 19 tests whether we can keep hinge's sharp gradient by raising the margin ceiling: margin=10.0 for stage-3 only. Saturation requires E_neg - E_pos > 10 — unlikely at Adam lr=3e-4 in 200k steps.

### What
- New file: `abm/loop_mpc_doorkey_run19.py` — condition `symbolic_large_margin`
- Single change from Run 15c: `ebm.contrastive_loss(z_exit, z_rh_neg, z_g_exit, margin=10.0)` on the stage-3 signal only
- All other signals keep margin=1.0 (stages 0/1, HER — they never saturated)
- Running in parallel with Run 18 to compare soft vs hard non-saturation
- `abm_experiment.py`: added `symbolic_large_margin` to condition choices
- EXPERIMENTS.md: Run 19 entry added to summary table

---

## 2026-04-27 — Run 18 built: non-saturating EBM (softplus)

### Why
Run 17 (L2 stage-3 cost) failed: only 1 goal success in 60k ACT steps vs Run 15c's 19 in 24k. Root cause: H=8 compound prediction errors make L2 position regression too inaccurate. EBM's pattern-matching was better (83% conversion before saturation in 15c). The fix is making EBM non-saturating: replace hinge clamp with softplus loss whose gradient is sigmoid(·) — never exactly 0.

### What
- New file: `abm/loop_mpc_doorkey_run18.py` — condition `symbolic_bce_ebm`
- Single change from Run 15c: `_train_ebm_softplus()` uses `F.softplus(e_pos - e_neg).mean()` instead of `ebm.contrastive_loss()` (hinge clamp)
- Applied to all 3 EBM signals: subgoal matching, HER, goal-vs-post_door_neg
- All other architecture identical to Run 15c
- `abm_experiment.py`: added `symbolic_bce_ebm` to condition choices
- EXPERIMENTS.md: Run 18 entry added, Run 17 mid-run status noted

---

## 2026-04-27 — Run 17 built; Run 15c final results logged

### Why
Run 15c (symbolic_only) confirmed the CEM+EBM architecture works when given a clean world model — pred_ewa stable at 0.009–0.011 throughout, goal grew 200→224. But goal froze after ~85k of ACT due to EBM saturation: once post_door_neg_buf fills to 5000 and hinge margins are satisfied, gradients → 0 and CEM gets uniform stage-3 energy. Run 17 replaces the saturating EBM cost for stage 3 with a direct L2 distance on position dims [x/5, y/5] — non-saturating, provably correct given the predictor learns real dynamics.

### What
- New file: `abm/loop_mpc_doorkey_run17.py` — condition `symbolic_l2_stage3`
- `_cem_stage3_l2_batch()`: batched CEM with `cost = (z_H[2]−goal_x)² + (z_H[3]−goal_y)²`, 512 samples, 64 elites, 5 iters, horizon=8
- `_cem_stage3_l2_single()`: single-env eval variant
- Stages 0/1: unchanged EBM-guided `mpc.plan_batch`/`mpc.plan_single`
- Stage 2 ACT: uses `_cem_stage3_l2_batch` with `goal_xy = active_goal_z[i][2:4]`
- Stage 2 eval: uses `_cem_stage3_l2_single` with `goal_xy = z_goal[0, 2:4]`
- `abm_experiment.py`: added `symbolic_l2_stage3` to condition choices
- EXPERIMENTS.md: Run 15c final results logged (peak=15%, EBM saturation confirmed), Run 17 entry added

---

## 2026-04-27 — Run 16 built; Run 15b killed; Run 15b/15c partial results logged

### Why
Run 15b (SYM_SCALE=10) confirmed scaling works — pred_ewa started at 0.010 (vs 14b's 0.001). But early EBM activation during OBSERVE decayed pred_ewa to 0.0017 by step 48k, same failure mode as 14a. Run killed at step 48k to free the pod. Run 15c (pure symbolic) showed goal buffer growing 200→219 in first 24k ACT steps — architecture confirmed working when predictor learns real dynamics.

Run 16 = 15b's SYM_SCALE=10 + 15a's `if not in_observe:` EBM gate. Single change over 15b.

### What
- New file: `abm/loop_mpc_doorkey_run16.py` — condition `vjepa2_symbolic_scaled_late_ebm`
- EBM training block wrapped in `if not in_observe:` (only change from 15b)
- `abm_experiment.py`: added `vjepa2_symbolic_scaled_late_ebm` to condition choices
- EXPERIMENTS.md: Run 15b killed result logged, Run 15c partial logged, Run 16 entry added

---

## 2026-04-26 — Run 14a/14b results logged; Runs 15a/15b/15c built

### Results logged
- **Run 14a (vjepa2_adapter):** peak=30%, final=20%, pred_ewa~0.02, goal=384, 11001s — adapter learned but EBM misaligned (started too early during random adapter init)
- **Run 14b (vjepa2_symbolic):** peak=50%, final=0%, pred_ewa≈0.001, goal=260, 9494s — 50% peak despite pred_ewa≈0 (EBM benefit), but symbolic dims drowned in 772-dim cosine space

### Runs built and pushed
- `abm/loop_mpc_doorkey_run15a.py` (`vjepa2_adapter_late_ebm`): delayed EBM — gated to `if not in_observe`, activates only after adapter has 80k OBSERVE steps to stabilize
- `abm/loop_mpc_doorkey_run15b.py` (`vjepa2_symbolic_scaled`): SYM_SCALE=10.0 — symbolic dims scaled 10× so one-tile move → cos_sim≈0.980, forcing pred_ewa above 0.02
- `abm/loop_mpc_doorkey_run15c.py` (`symbolic_only`): pure 5-dim symbolic state [has_key, door_open, x/5, y/5, dir/3], no visual encoder — tests whether CEM+EBM architecture works at all with a perfect world model

---

## 2026-04-25 — Run 13: Frozen V-JEPA 2.1 ViT-Base-384 encoder for DoorKey planning

### Why
Runs 1–12 repeatedly failed because a tiny CNN trained from scratch (LeWM) cannot produce a latent space with meaningful geometry. Run 12 (DINOv2) and Run 13 (V-JEPA 2.1) are the first runs that cleanly test the planning architecture on top of rich pretrained features. V-JEPA 2.1 is more theoretically motivated than DINOv2: trained on ~2M hours of video with a JEPA objective (predict future latents from current ones), it's sensitive to motion and temporal dynamics — exactly what DoorKey planning requires.

### What
- New file: `abm/loop_mpc_doorkey_run13.py` — identical to Run 12 except encoder
- Encoder: `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384')` → returns `(encoder, predictor)`, use encoder only
- Input: `(B, 3, 1, 384, 384)` — T=1 single frame; output: mean-pool 576 patch tokens → `(B, 768)`, L2-normalized
- V-JEPA 2.1 supports single-frame natively via `img_temporal_dim_size=1` + `interpolate_rope=True`
- All Run 12 architecture retained: FeatureReplayBuffer, GoalFeatureBuffer, FeaturePredictor (MLP 768→1024→768), EBMCostHead(768), protected seed_buf, post_door_neg_buf, 3-stage subgoals, curiosity OBSERVE, HER
- Condition: `vjepa2_frozen` added to `abm_experiment.py` choices
- Help text for `--loop-module` updated to include run12 and run13

## 2026-04-22 — Add loop_mpc_doorkey.py: Phase 1 DoorKey planner-only scout

### Why
Phase 1 of the two-claim scientific plan: prove "world model + planning > RL" in isolation before testing autonomous System M. DoorKey chosen because it has a clean 42% PPO-only baseline to beat on a short-horizon task. Fixed schedule (80k observe → always ACT with CEM) removes System M as a confound entirely.

### What
- New file: `abm/loop_mpc_doorkey.py` — DoorKey-specific planner-only loop
- Conditions: `planner_only` (CEM on LeWM) vs `random` baseline
- Fixed schedule: observe_steps=80k, then always use CEM
- LeWM: `latent_dim=256`, `predictor_type="mlp"`, `img_size=48`
- CEM: `horizon=10`, `n_samples=512`, `n_elites=64`, `n_iters=5`, `distance="cosine"`
- `GoalImageBuffer`: collects raw success frames (reward > 0), re-encodes at eval time
- 30-episode eval every 5k steps, same metrics dict format as all other loops
- `run_abm_loop = run_doorkey_mpc_loop` alias for `abm_experiment.py` dispatch

### Threshold to promote
`planner_only` success rate > 42% (PPO-only baseline on DoorKey)

---

## 2026-04-22 — Rename abm/ files for agent clarity

### Why
File names were ambiguous: `loop.py` gave no hint it was PPO-based, `loop_acwm.py` used an internal acronym, `vjepa_encoder.py` actually ran DINOv2 (V-JEPA was abandoned after feature collapse). Renamed all 6 files so any agent reading the codebase immediately knows what each does.

### Renames
- `abm/loop.py` → `abm/loop_ppo_lewm.py` (PPO on LeWM features, all envs)
- `abm/loop_acwm.py` → `abm/loop_mpc_crafter.py` (pure CEM planner, no PPO, Crafter-only)
- `abm/loop_acwm_tiered.py` → `abm/loop_mpc_tiered_crafter.py` (tiered+sticky goal MPC scout)
- `abm/vjepa_encoder.py` → `abm/dinov2_encoder.py` (DINOv2 ViT-B/14, not V-JEPA)
- `abm/mpc.py` → `abm/cem_planner.py` (Cross-Entropy Method planner + GoalBuffer)
- `abm/lewm.py` → `abm/world_model.py` (LeWM world model, CNN encoder, replay buffers)

### Also
- `VJEPAEncoder` class renamed to `DINOv2Encoder` in `dinov2_encoder.py`
- All internal imports updated across all loop files
- `abm_experiment.py` `--loop-module` default updated to `abm.loop_ppo_lewm`

---

## 2026-04-20 — MPC+RL hybrid, bug fixes from MacBook test, RunPod prep

### Why
Pure MPC-only ran 11+ hours on MacBook with 0% success. Root causes: stale goal
embeddings after encoder updates, cosine distance wrong for LeWM (needs L2),
eval too slow (50 eps × MPC planning). Yann says "minimize RL, not eliminate it"
— MPC+RL hybrid is the right approach.

### Changes
- `abm/mpc.py` — Added `distance` param to CEMPlanner ("l2" or "cosine")
- `abm/loop.py` — Re-encode goals on OBSERVE→ACT switch (fixes stale embeddings),
  4x predictor training for MPC, eval uses PPO agent when available (MPC+RL),
  encoder freeze threshold 0.08→0.03, horizon 7→12
- `abm/meta_controller.py` — Plateau check logs changed to debug level

### Revised Test Matrix (RunPod A100)
1. 30K observe, MPC+PPO (`--observe-steps 30000 --use-mpc`)
2. 50K observe, MPC+PPO (`--observe-steps 50000 --use-mpc`)

## 2026-04-19 — MPC goal-conditioned planning on DoorKey (no RL)

### Why
Yann LeCun: "abandon RL in favor of model-predictive control." DoorKey's LeWM
encoder discriminates states well (unlike DINOv2 CLS on indoor scenes), so it's
the right env to prove the full LeCun architecture: observe → build world model →
imagine actions → plan toward goal image. No RL needed.

### Changes
- `abm/loop.py` — Added `GoalObsWrapper` (teleports MiniGrid agent to goal square,
  renders, restores). Added `eval_doorkey_mpc()` for MPC-based evaluation. Added
  MPC-only and MPC+RL ACT paths for DoorKey. CEM planner initializes with
  `lewm.predictor` after warmup (horizon=7, 256 samples, 32 elites). Goal buffer
  pre-seeded by teleporting to goal in each training env.
- `abm/loop.py` — `run_abm_loop()` now accepts `observe_steps`, `use_mpc`, `use_rl`
  parameters for the 4-test matrix.
- `abm_experiment.py` — Added `--observe-steps`, `--use-mpc`, `--no-rl` CLI flags.

### 4-Test Matrix (MacBook M3 Pro)
1. 30K observe, MPC only (`--observe-steps 30000 --use-mpc --no-rl`)
2. 30K observe, MPC + RL (`--observe-steps 30000 --use-mpc`)
3. 50K observe, MPC only (`--observe-steps 50000 --use-mpc --no-rl`)
4. 50K observe, MPC + RL (`--observe-steps 50000 --use-mpc`)

## 2026-04-19 — Auto-save test results to GitHub

### Why
RunPod results are lost when pods shut down. Need persistent storage.

### Changes
- `abm_experiment.py` — Added `push_results_to_github()` that copies JSON/PNG/HTML
  results to `tests/{env}_{timestamp}/`, writes a summary.json, and auto-pushes to
  GitHub after each experiment completes.

## 2026-04-19 — Fix goal diversity + predictor normalization (Run 3 prep)

### Why
Run 2 (200K steps) showed goal_div=0.01 (all 100 goal embeddings nearly identical)
and cem_cost=0.74 flat (planner can't differentiate actions). Two root causes:
single-view goal capture produces identical DINOv2 CLS embeddings across rooms,
and unnormalized predictor output drifts off the unit sphere over 15 CEM rollout steps.

### Changes
- `abm/habitat_env.py` — `get_goal_obs()` now renders 4 views (0°, 90°, 180°, 270°)
  at goal position, returns list of 4 obs dicts instead of 1. Gives goal buffer
  diverse spatial perspectives of each goal location.
- `abm/lewm.py` — `VJEPAPredictor.forward()` now L2-normalizes output. Keeps
  predicted states on the unit sphere so cosine distance stays meaningful across
  all CEM rollout steps.
- `abm/loop.py` — Updated habitat goal pre-seeding to push all 4 views individually.
  Updated eval functions to mean-pool multi-view goal encodings.

## 2026-04-19 — Run 2 tuning: H=15, cosine distance, diagnostic logging

### Why
Run 1 (200K steps) achieved 0% success despite working architecture. Root causes:
H=7 horizon too short for PointNav (1.75m reach vs 5-10m episodes), L2 distance
wrong for cosine-trained DINOv2 features, and no diagnostic data to debug CEM planning.

### Changes
- `abm/mpc.py` — CEM scoring: L2² → cosine distance (1 - cos_sim). Added
  `_last_best_cost` tracking for logging.
- `abm/loop.py` — CEM horizon 7→15. Added cem_cost and goal_div (goal embedding
  std) to verbose logging.
- `abm/habitat_env.py` — Suppress noisy C++ warnings (HABITAT_SIM_LOG, MAGNUM_LOG,
  GLOG_minloglevel).

## 2026-04-19 — Replace torch.hub with timm for DINOv2 loading

### Why
torch.hub loads DINOv2 by importing Python source from the repo cache. The latest
DINOv2 code uses PEP 604 (`float | None`) which crashes on Python 3.9 (required by
habitat-sim). The previous `_patch_dinov2_for_py39()` workaround failed on fresh
machines because the cache didn't exist yet when the patch ran.

### Fix
`abm/vjepa_encoder.py` — Replaced `torch.hub.load("facebookresearch/dinov2")` with
`timm.create_model("vit_base_patch14_dinov2.lvd142m")`. timm downloads only weights
(no Python source import), so it works on any Python version. Same model, same weights,
no patching needed. Removed `_patch_dinov2_for_py39()` entirely.

## 2026-04-18 — Standalone Habitat setup script

### Why
habitat-sim only ships conda packages for Python 3.9. The RunPod base image uses
Python 3.11. The old `setup_cloud.sh` habitat block always failed (no pip wheels,
conda fallback hits wrong Python). RTX 5090 (sm_120) is incompatible with the older
PyTorch needed for Python 3.9 — must use A100/H100 (sm_80/sm_90).

### Changes
- `setup_habitat.sh` (NEW) — One-shot script: creates Python 3.9 conda env, installs
  habitat-sim via conda, PyTorch cu121, all deps, downloads test scenes, runs sanity check.
- `setup_cloud.sh` — Removed broken habitat install block, replaced with pointer to
  `setup_habitat.sh`.

## 2026-04-18 — Fix DINOv2 + Python 3.9 compatibility (habitat-sim)

### Why
habitat-sim only provides pre-built packages for Python 3.9. DINOv2's torch.hub code
uses PEP 604 type unions (`float | None`) which require Python 3.10+. This crashes
on the conda Python 3.9 environment needed for habitat-sim.

### Fix
`abm/vjepa_encoder.py` — Added `_patch_dinov2_for_py39()` that automatically adds
`from __future__ import annotations` to cached DINOv2 .py files before loading. This
makes all type annotations lazy-evaluated strings, so PEP 604 syntax works on 3.7+.
The patch is a no-op on Python 3.10+ and only modifies files containing `float | `.

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
