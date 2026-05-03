# Changelog

## 2026-05-02 — Run 37 results: HWM hierarchy peak=40% (2× flat CEM baseline)

### Result
- Run 37 `lewm_maniskill_hierarchy` on FetchPickAndPlace-v4: **peak=40%** success rate
- Run 35 flat CEM baseline: **peak=20%**
- Hierarchy doubles success — subgoal decomposition confirmed effective even with simple cosine scoring heuristic

### What changed
- `EXPERIMENTS.md`: updated Run 35 results (pending → 20%), added Run 37 entry with full architecture and results

---

## 2026-05-02 — Run 37 built: HWM-style hierarchy on FetchPickAndPlace-v4

### Why
Run 35 flat CEM peaks at 20% on FetchPickAndPlace-v4. The task has a natural two-phase structure (reach block → lift to goal) that a single H=8 flat horizon can't reliably capture. HWM-style hierarchy tests whether a high-level unconditional predictor scoring waypoints can double the baseline.

### What
- New file: `abm/loop_lewm_maniskill_run37.py` — HWM hierarchy loop; condition `lewm_maniskill_hierarchy`
- `HighLevelPredictor`: MLP z→z_next (no action), trained on (obs_t, obs_{t+STRIDE}) pairs
- `WaypointReplayBuffer`: stores stride-3 observation pairs for hi-pred training
- Subgoal selection: scores N_CANDIDATES=200 goal buffer entries via α·cos(hi_pred(z), z_cand) + (1-α)·cos(z_cand, z_goal)
- Low-level CEM: H=3, K=100, n_iters=10 (faster than Run 35's H=8 K=300 n_iters=30)
- `abm_experiment.py`: added `lewm_maniskill_hierarchy` to valid conditions

---

## 2026-05-01 — Run 35 built: LeWM + Continuous CEM on ManiSkill3 PickCube-v1

### Why
Crafter experiments (Runs 29-34) established that SIGReg + CEM fails on visually ambiguous long-chain tasks. ManiSkill3 PickCube-v1 addresses both failure conditions: visually distinct before/after states (cube on table vs cube at goal), short 200-step horizon, and richer signal available. First test of whether geometric planning transfers to continuous robot manipulation.

### What
- New file: `abm/cem_continuous.py` — ContinuousCEMPlanner: Gaussian CEM over H×a_dim continuous action sequences, with warm-start shift between MPC steps
- New file: `abm/loop_lewm_maniskill_run35.py` — full OBSERVE/ACT loop for ManiSkill3 PickCube-v1; condition `lewm_maniskill_pickcube`
- Env: PickCube-v1, obs_mode="state_dict+rgb", control_mode="arm_pd_ee_delta_pose" (6-dim continuous)
- SIGReg: M=1024, λ=0.1 (LeWM paper values, up from M=512, λ=0.05 in Crafter runs)
- CEM: K=300, elite=30, iters=30, H=8 (LeWM paper values for continuous control)
- Goal buffer: observations where cube height > 0.05m or info['success']=True
- Video recording: eval MP4s saved to media/maniskill_run35_eval_step{N}.mp4
- `abm_experiment.py`: added `lewm_maniskill_pickcube` to valid conditions; added `maniskill` to env choices

## 2026-04-29 — Run 31 built: Director-lite with longer horizon + cosine intrinsic reward

### Why
Run 30 diagnosed two problems: H_MANAGER=50 is too short for tier3 prerequisite chains (~150-200 steps), and sparse achievement rewards give REINFORCE near-zero signal in most subgoal periods (manager can't learn). Two targeted fixes: longer horizon covers full prerequisite chains; cosine progress intrinsic reward provides dense signal in every step.

### What
- New file: `abm/loop_mpc_crafter_run31.py` — condition `lewm_crafter_hierarchy_v2`
- H_MANAGER: 50 → 150 (covers wood→table→pickaxe chain length)
- MGR_BATCH: 32 → 16 (more frequent updates for longer horizon)
- INTRINSIC_COEF = 0.5: cosine progress reward added at every ACT step
- `prev_cos_sim[n_envs]`: tracks per-env cosine baseline, reset on new subgoal/episode end
- Total manager reward = achievement_reward + 0.5 * sum(max(0, Δcos_sim))
- `abm_experiment.py`: added `lewm_crafter_hierarchy_v2` to valid conditions

### How to run
```bash
cd /workspace/ose
git pull
python abm_experiment.py --loop-module abm.loop_mpc_crafter_run31 \
  --condition lewm_crafter_hierarchy_v2 --device cuda --env crafter \
  --steps 600000 --n-envs 8
```

---

## 2026-04-29 — Run 30 built: Director-lite hierarchy on Crafter pixels

### Why
Run 29 confirmed a hard ceiling at ~27% arithmetic score with tier3/tier4=0%. Flat L2 CEM with random goals cannot plan prerequisite chains (wood→table→pickaxe→iron). Run 30 adds a SubgoalManager (Director-style, arXiv 2206.04114): a small MLP trained with REINFORCE that maps z_cur → discrete code → codebook subgoal. Worker (CEM) plans toward the subgoal with cosine distance. Ordering expected to emerge from the manager learning which subgoal sequences unlock achievements.

### What
- New file: `abm/loop_mpc_crafter_run30.py` — condition `lewm_crafter_hierarchy`
- `SubgoalManager`: 3-layer MLP → Categorical(K=64) → codebook lookup; trained with REINFORCE
- `_build_codebook()`: sklearn MiniBatchKMeans on all OBSERVE replay z at OBSERVE→ACT transition
- `_manager_update()`: REINFORCE with normalized returns, entropy_coef=0.01, grad_clip=1.0
- CEM distance changed from "l2" → "cosine" (Director max-cosine similarity)
- Manager horizon H_MANAGER=50: worker gets 50 primitive steps per subgoal before manager re-selects
- Manager checkpoint saved alongside encoder+predictor
- `abm_experiment.py`: added `lewm_crafter_hierarchy` to valid conditions; fixed `success_rate` KeyError in `plot_learning_curves` (now falls back to `crafter_score`)
- wandb logging: scalars every 10k steps (score, pred_loss, sigreg_loss, mgr_loss, per-tier breakdown); video every 50k steps via `_record_episode()` → wandb.Video (T, C, H, W) at 10 fps; project `lewm-crafter` run name `run30-lewm_crafter_hierarchy`
- World model training (OBSERVE phase): unchanged from Run 29

### How to run
```bash
cd /workspace/ose
pip install timm crafter scikit-learn
python abm_experiment.py --loop-module abm.loop_mpc_crafter_run30 \
  --condition lewm_crafter_hierarchy --device cuda --env crafter \
  --steps 600000 --n-envs 8
```

---

## 2026-04-29 — Run 29 built: LeWM on Crafter pixels (same arch, first Crafter test)

### Why
Same architecture as Run 28 validated on Crafter. Validates that ViT-Tiny + online SIGReg + L2 CEM is not overfitted to DoorKey's simple grid structure. No manual stages — uses open-ended goal-conditioned exploration (random goals from replay, biased toward achievement-positive obs).

### What
- New file: `abm/loop_mpc_crafter_run29.py` — condition `lewm_crafter_pixels`
- `ViTTinyEncoder`: img_size=64 (Crafter native, 4×4=16 patches), z_dim=256
- `PixelReplayBuffer`: 30k × 64 × 64 × 3 ≈ 368 MB
- `GoalPixelBuffer`: flat deque (no bucketing) — 70% from achievement-positive obs, 30% from replay
- No scripted seeder, no manual stages — pure open-ended goal discovery
- Training: identical to Run 28 (MSE + SIGReg λ=0.05, grad clip 1.0)
- Eval metric: Crafter achievement score (fraction of 22 achievements unlocked)
- `abm_experiment.py`: added `lewm_crafter_pixels` to valid conditions

### How to run
```bash
cd /workspace/ose
pip install timm crafter
python abm_experiment.py --loop-module abm.loop_mpc_crafter_run29 \
  --condition lewm_crafter_pixels --device cuda --env crafter \
  --steps 600000 --n-envs 8
```

---

## 2026-04-29 — Run 28 built: LeWM on DoorKey pixels (ViT-Tiny + SIGReg + L2 CEM)

### Why
Run 27 confirmed the hard ceiling of 5-dim symbolic state. Even with exact goal positions, CEM H=3 cannot navigate stage-2 (goal counter 200→204 over 100k+ ACT steps). Root cause: adjacent grid cells differ by 0.2 in the symbolic space — indistinguishable after H predictor steps under compound error. The fix is a metrically meaningful latent space where SIGReg (arXiv 2603.19312) spreads representations so L2 distance creates real CEM signal. This architecture (ViT-Tiny encoder + online SIGReg) requires no symbolic features and scales unchanged to Crafter and beyond.

### What
- New file: `abm/loop_mpc_doorkey_run28.py` — condition `lewm_doorkey_pixels`
- `ViTTinyEncoder`: timm `vit_tiny_patch16_224`, img_size=48, projected to z_dim=256
- `PixelGoalBuffer`: stores raw pixel obs bucketed by (col, row) goal-cell position; encodes on-the-fly with frozen encoder at plan time → no episode-mixing bug, no stale embeddings
- `PixelReplayBuffer`: pre-allocated numpy array (50k × 48 × 48 × 3), ~346MB
- Training: `MSE(z_pred, sg(z_next)) + 0.05 * sigreg(z_t)` every 16 vectorised steps
- Reused unchanged: `Predictor` and `sigreg()` from `world_model.py`; `CEMPlanner(distance="l2")` from `cem_planner.py` (L2 path already implemented)
- OBSERVE+ACT: 150k OBSERVE trains encoder+predictor; encoder+predictor frozen at ACT start
- Three manual stages identical to R26/27 logic but now in pixel/latent space
- `abm_experiment.py`: added `lewm_doorkey_pixels` to valid conditions

### How to run
```bash
cd /workspace/ose
pip install timm
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_run28 \
  --condition lewm_doorkey_pixels --device cuda --env doorkey \
  --steps 300000 --n-envs 8
```

---

## 2026-04-28 — Run 27 built: fix stage-2 goal episode-mixing bug from Run 26

### Why
Run 26's stage-2 CEM sampled the goal latent from `goal_buf`, which stores terminal states from many past episodes. DoorKey randomises the goal cell position each episode reset, so `goal_buf` contains states with different (x, y) goal positions. Stages 0/1 are unaffected — `has_key=1` and `door_open=1` are binary flags independent of position. Stage 2 navigates to the exit tile — position IS the discriminating feature. Planning toward a mixed-episode average goal position produces approximately random actions; the agent cannot reliably complete stage 2.

### What
- New file: `abm/loop_mpc_doorkey_run27.py` — condition `symbolic_exact_goal_s2`
- Stage-2 goal now constructed from `_find_cell(uw, "goal")` at goal-refresh time: `[1., 1., gx/5, gy/5, 0.]` — exact position for the current episode, never stale
- Same fix applied in `_eval_run27`: reads goal from eval env directly instead of `goal_buf.sample(1)`
- `_pick()` helper in eval replaced with `_pick_s01()` (stages 0/1 only) to make the asymmetry explicit
- `abm_experiment.py`: added `symbolic_exact_goal_s2` to valid conditions
- Everything else identical to Run 26 (H=8 stages 0/1, H=3 stage 2, frozen predictor, two-phase EBM)

### How to run
```bash
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_run27 \
  --condition symbolic_exact_goal_s2 --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 40000
```

---

## 2026-04-28 — Run 26 built: short-horizon CEM H=3 for stage 2 (pure world model, HWM-inspired)

### Why
Run 24 proved the predictor is accurate at H=8 for stages 0/1 but used oracle BFS for stage 2 (not publishable). Run 25 adds PPO for stage 2 (hybrid). Run 26 keeps everything in the world-model framework — no BFS, no RL — using the HWM paper insight (arXiv 2604.03208, Figure 6): low-level predictors are most accurate at H≤4 due to compound error at longer horizons. DoorKey stage 2 is a short navigation (~4-5 cells). CEM at H=3 replanning every step stays in the accurate predictor regime while covering the full path through receding-horizon iteration.

### What
- New file: `abm/loop_mpc_doorkey_run26.py` — condition `symbolic_short_horizon_s2`
- `CEM_HORIZON_S2 = 3` (vs `CEM_HORIZON = 8` for stages 0/1)
- Two CEMPlanner instances sharing same frozen predictor and EBM: `mpc` (H=8) and `mpc_s2` (H=3)
- Stage 2 goal sampling from `goal_buf` (seeded with ~150 scripted successes, refreshed every 64 steps)
- `_eval_run26`: uses `mpc_s2.plan_single` for stage 2, goal from `goal_buf`
- Heartbeat log includes `s2_envs=N` count
- `abm_experiment.py`: added `symbolic_short_horizon_s2` to valid conditions

### How to run
```bash
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_run26 \
  --condition symbolic_short_horizon_s2 --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 40000
```

---

## 2026-04-28 — Run 25 built: PPO stage 2 replaces scripted BFS (publishable hybrid)

### Why
Run 24 beat 42% (peak=50%) using oracle BFS for stage 3. BFS is not a learned policy — a reviewer would ask "what does the world model contribute if you're using A* for the hard part?" Run 25 replaces BFS with a small PPO actor-critic (2-layer MLP, 64 hidden, 5-dim input) trained only on post-door stage-3 transitions. If PPO stage 3 sustains peak > 42%, the claim is clean: world model (CEM+EBM) handles multi-step subgoal discovery (key→door); RL (PPO) handles short-range navigation (door→exit). Combined system outperforms pure RL (42% baseline).

### What
- New file: `abm/loop_mpc_doorkey_run25.py` — condition `symbolic_ppo_stage3`
- `PPOActorCritic`: 2-layer MLP (64 hidden, tanh), input=5-dim (agent_x, agent_y, agent_dir, goal_x, goal_y), output=7 logits + 1 value
- `PPORollout`: flat buffer, collects stage-3 (s2_idx) transitions only; update every 512 stage-3 steps
- `_ppo_update`: 4 epochs, minibatch=64, clip=0.2, ent_coef=0.01, vf_coef=0.5, GAE γ=0.99 λ=0.95, step penalty=-0.005
- `_get_ppo_obs(uw)`: PPO obs includes goal_x, goal_y (not in world model state — needed because goal position is randomised per episode)
- Predictor: frozen at OBSERVE end (same as Run 24)
- EBM: hinge OBSERVE → softplus ACT (same as Run 24)
- `_eval_run25`: deterministic greedy PPO for stage 2 (argmax logits)
- `abm_experiment.py`: added `symbolic_ppo_stage3` to valid conditions

---

## 2026-04-28 — Run 24 final: peak=50% — 42% PPO baseline beaten ★

### Result
peak=50.0%, final=10% | 6932s | pred_ewa locked at 0.0098 all 160k ACT steps | goal 220→366 | steps_to_80=N/A.  
First run to exceed the 42% PPO baseline. Frozen predictor preserved door-opening rate throughout ACT (vs Run 23 where pred_ewa 0.011→0.006 caused late-ACT collapse). Goal grew to 366 — most sustained exploration of any run. Peak hit at step 100k (50%), secondary peaks at 45%/40%/35%. Success is noisy across eval windows (5%–50% oscillation at n_envs=16 granularity); underlying conversion rate ~0.9 goal/1k steps.

Limiting constraint: 8 of 16 env-pool layouts are "hard" — CEM H=8 can't plan the door→exit path. Scripted BFS solves the easy 8 cleanly. Remaining 8 need hierarchical planning or a learned stage-3 policy.

### What
- EXPERIMENTS.md: Run 24 full entry added, summary table updated with ★

---

## 2026-04-28 — Run 23 final: stage 3 bottleneck confirmed, pred_ewa degradation discovered

### Result
peak=25%, final=5% | 8559s | goal=242 (first run past 226 ceiling) | pred_ewa 0.011→0.006 during ACT | 87% stage-3 conversion when door opened | door-opening rate collapsed by late ACT.  
Stage 3 is the confirmed sole bottleneck. But predictor degrades during ACT as replay buffer fills with goal-directed data — door-opening rate drops from 1/1k steps to 1/10k by run end.

### What
- EXPERIMENTS.md: Run 23 full entry added, summary table updated

---

## 2026-04-28 — Run 21 final: H=12 confirmed worse than H=8

### Result
peak=5%, final=0% | killed early | goal stalled 200–205, never built meaningful stage-3 conversions.  
H=12 compound prediction errors destroy CEM plan quality. Each rollout step accumulates predictor error; by step 12, z_pred noise exceeds signal. Flat horizon scaling is the wrong axis — hierarchical planning is the principled fix.

### What
- EXPERIMENTS.md: Run 21 full entry added, summary table updated

---

## 2026-04-28 — Run 24 built: frozen predictor + scripted BFS stage 3

### Why
Run 23 confirmed stage 3 was the sole initial bottleneck (goal grew past 226, 87% stage-3 conversion). But pred_ewa degraded 0.011→0.006 during ACT as the replay buffer filled with goal-directed data. Door-opening rate dropped from 1/1k to 1/10k steps. Freezing the predictor at OBSERVE end keeps the CEM at peak quality throughout all 160k ACT steps. EBM continues training (softplus) independently.

### What
- New file: `abm/loop_mpc_doorkey_run24.py` — condition `symbolic_frozen_pred_stage3`
- Predictor trains during OBSERVE only — `if in_observe: pred_loss = _train_predictor(...)`
- EBM trains independently throughout (hinge OBSERVE → softplus ACT)
- CEM creation decoupled from predictor training
- Stage 3: scripted BFS (same as Run 23), everything else identical
- `abm_experiment.py`: added `symbolic_frozen_pred_stage3`

---

## 2026-04-28 — Run 23 built: scripted BFS oracle for stage 3 (diagnostic)

### Why
Three runs (15c/18/20) stall at goal≈224–226 regardless of EBM loss. Run 21 tests H=12. Run 23 answers the higher-level question first: is stage 3 the only bottleneck? CEM+EBM for stages 0/1 is unchanged. Stage 2 (door→exit) replaced with the existing BFS scripted policy. If this beats 42%, stage 3 is confirmed as sole ceiling and Run 21/22 are on the right path. If it still stalls, there's a hidden stage 0/1 problem.

### What
- New file: `abm/loop_mpc_doorkey_run23.py` — condition `symbolic_scripted_stage3`
- `_scripted_stage3_action(uw)`: BFS navigator to goal cell, only called when door is open
- ACT loop: stage 0/1 envs → `mpc.plan_batch` (CEM+EBM, H=8); stage 2 envs → `_scripted_stage3_action` per-env
- Eval function: `current_stage == 2` → scripted, else → `mpc.plan_single`
- EBM still trains on stage-3 data (softplus, same as Run 20) — same training, different action selection
- Single CEMPlanner (H=8) for stages 0/1 only
- `abm_experiment.py`: added `symbolic_scripted_stage3`

---

## 2026-04-28 — Run 21 built: H=12 for stage 2 (door→exit)

### Why
Runs 15c, 18, and 20 all stall at goal≈224–226 with healthy pred_ewa (0.013+). Three independent EBM loss functions hitting the same ceiling rules out saturation — the binding constraint is the CEM horizon. H=8 cannot navigate hard door→exit layouts that require >8 steps. Run 21 keeps the full Run 20 architecture (hinge→softplus two-phase EBM) and makes exactly one change: stage-2 planning uses H=12.

### What
- New file: `abm/loop_mpc_doorkey_run21.py` — condition `symbolic_horizon12_s2`
- Two `CEMPlanner` instances: `mpc_fast` (H=8, stages 0/1) and `mpc_deep` (H=12, stage 2)
- ACT loop splits envs by stage — `s01_idx` → `mpc_fast.plan_batch`, `s2_idx` → `mpc_deep.plan_batch`
- Both planners share the same predictor and receive `set_ebm(ebm)` on EBM activation
- Eval function uses `mpc_deep if current_stage == 2 else mpc_fast`
- `abm_experiment.py`: added `symbolic_horizon12_s2` to valid conditions
- Everything else identical to Run 20 (hinge OBSERVE → softplus ACT, same buffers/LRs/seeds)

---

## 2026-04-28 — Run 20 final: architectural ceiling confirmed at ~25%

### Result
peak=25.0%, final=0.0% | 8569s | pred_ewa record 0.0144 (step 112k) | goal 200→226 | stalled at 224–226 for last 85k ACT steps.  
Three independent runs (15c, 18, 20) all stall at goal≈224–226 regardless of EBM loss function. Saturation is ruled out — EBM and predictor both healthy. The architectural ceiling is the CEM horizon: H=8 cannot navigate the harder door→exit layouts that require >8 steps. Next: Run 21 increases H to 12 for stage 3.

### What
- EXPERIMENTS.md: Run 20 full entry added, summary table updated

---

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
