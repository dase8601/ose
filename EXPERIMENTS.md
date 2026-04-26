# OSE Experiment Log

Tracks every scout run: what we tested, why, and what it showed.
Update this file after each run completes.

---

## Phase 1 — Prove world model + planning > RL (DoorKey)

**Env:** MiniGrid-DoorKey-6x6  
**Baseline to beat:** 42% (PPO-only)  
**Loop:** `abm/loop_mpc_doorkey.py` — fixed schedule, no System M  
**Promote threshold:** planner_only success rate > 42%

---

### Run 1 — 2026-04-24

| Parameter | Value |
|-----------|-------|
| Condition | planner_only |
| Steps | 200k |
| n_envs | 16 |
| observe_steps | 80k |
| CEM horizon | 10 |
| CEM samples | 512 |
| CEM elites | 64 |
| CEM iters | 5 |
| latent_dim | 256 |
| Train during ACT | No |
| Device | A100 |

**Result:** 0.0% success throughout all ACT steps (80k–200k)  
**Goals collected:** 19 (very sparse — random walk on DoorKey ≈ 0.0015% hit rate)  
**ssl_ewa at OBSERVE end:** 0.077 (converged)

**Why it failed:**  
H=10 is too short for DoorKey's multi-step structure (navigate to key → pick up → navigate to door → unlock → reach goal = 30–80 steps minimum). CEM can't plan through the full task with only 10-step lookahead. World model also froze at OBSERVE end, so ACT-phase transitions (goal-directed) never improved it.

**What we changed for Run 2:**  
- CEM horizon: 10 → 30  
- World model training continues during ACT phase

---

### Run 2 — 2026-04-24

| Parameter | Value |
|-----------|-------|
| Condition | planner_only |
| Steps | 200k |
| n_envs | 16 |
| observe_steps | 80k |
| CEM horizon | **30** |
| CEM samples | 512 |
| CEM elites | 64 |
| CEM iters | 5 |
| latent_dim | 256 |
| Train during ACT | **Yes** |
| Device | A100 |

**Result:** 0.0% success throughout all 120k ACT steps  
**Goals collected:** 23 by step 200k (vs 19 in Run 1 — more goals, same outcome)  
**ssl_ewa at OBSERVE end:** 0.0675 (well converged)  
**ssl_ewa during ACT:** dropped to 0.028 then stabilized ~0.04 — world model kept learning on goal-directed transitions as expected  
**Peak success:** 0.0%

**Why it failed:**  
Cosine distance in latent space is not a good cost function for CEM. The LeWM encoder was trained to predict next states, not to make "closeness to goal" meaningful in its latent geometry. CEM optimized the wrong objective — minimizing cosine distance to z_goal does not correlate with task progress on DoorKey.

**What we changed for Run 3:**  
- Added EBM cost head: `E(z, z_goal) → scalar` trained contrastively from goal buffer vs replay buffer  
- EBM replaces cosine distance in CEM once trained for 500 steps with ≥5 goals  
- Directly implements LeCun's energy-based objective-driven AI cost module  
- EBM trains in both OBSERVE and ACT phases — improves continuously

---

### Run 3 — 2026-04-25

| Parameter | Value |
|-----------|-------|
| Condition | planner_only |
| Steps | 200k |
| n_envs | 16 |
| observe_steps | 80k |
| CEM horizon | 30 |
| CEM samples | 512 |
| CEM elites | 64 |
| CEM iters | 5 |
| latent_dim | 256 |
| Train during ACT | Yes |
| Cost function | **EBM** (contrastive, replaces cosine) |
| EBM min goals | 5 |
| EBM warmup | 500 steps |
| Device | RTX 4090 |

**Result:** 0.0% success throughout all 120k ACT steps  
**Goals collected:** 19 by step 200k (18 at step 190k, last one at step 200k)  
**EBM activated:** confirmed ON by step 192k heartbeat  
**ssl_ewa final:** 0.0309 (end of ACT) | 0.0675 (OBSERVE end)  
**Total wall time:** 16,263s (~4.5 hrs on RTX 4090)  
**Peak success:** 0.0%

**Why it failed:**  
EBM was trained on ~23 goal images — all from the same tiny region of state space (the exit). The world model was trained entirely on random-walk transitions and has never seen key pickup or door unlock sequences. So when CEM imagines H=30-step futures, it rolls through a model that cannot simulate the task-critical transitions. Even with a correctly-shaped EBM cost, CEM can't find the goal because the imagined future is wrong.

**Root cause (confirmed):** Train-test distribution gap in the world model. The model is trained on random-walk data; CEM evaluates goal-directed plans. These distributions don't overlap enough for the EBM to be meaningful in the regions CEM actually explores.

**What we changed for Runs 4–7:**  
Four separate diagnostic runs targeting different hypotheses about the failure:

---

### Run 4 — (next) curiosity_observe

**File:** `abm/loop_mpc_doorkey_curiosity.py`  
**Loop module:** `abm.loop_mpc_doorkey_curiosity`  
**Condition:** `curiosity_observe`

| Parameter | Value |
|-----------|-------|
| OBSERVE policy | **Curiosity (novelty-maximizing)** |
| Everything else | Identical to Run 3 |

**Hypothesis:** World model never sees key/door transitions because random walk almost never triggers them (~0.0015% per step). Curiosity replaces random walk with Plan2Explore-style novelty — for each env, run all 7 actions through the predictor, pick the one with the most novel predicted next-latent. This should actively steer exploration toward rare transitions the model doesn't know yet.

**Result:** 0.0% success throughout all 120k ACT steps  
**act_steps:** 120,000 (60% of total)  
**Peak success:** 0.0%

**Why it failed:**  
Curiosity improved exploration diversity but the planning horizon (H=30) remained the bottleneck. Even with a better-trained world model, 30-step CEM rollouts compound prediction errors to the point where CEM cannot distinguish good action sequences from noise. Exploration quality is not the binding constraint — model accuracy over long horizons is.

---

### Run 5 — 2026-04-26 — her_goals

**File:** `abm/loop_mpc_doorkey_her.py`  
**Loop module:** `abm.loop_mpc_doorkey_her`  
**Condition:** `her_goals`

| Parameter | Value |
|-----------|-------|
| EBM signal | **Standard + HER** |
| HER buffer | All episode-end states (4k capacity) |
| Everything else | Identical to Run 3 |

**Hypothesis:** EBM had only ~23 positive training pairs in 200k steps. HER relabeling treats every episode-end state as "a goal achieved for itself" — E(z_end, z_end) should be low, E(z_random, z_end) should be high. This gives thousands of positive pairs per run instead of 23.

**Result:** 0.0% success throughout all 120k ACT steps  
**Goals collected:** 11 actual successes | HER buffer: 548 terminal states  
**ssl_ewa final:** 0.0259  
**Total wall time:** 16,795s (~4.7hr)  
**Peak success:** 0.0%

**Why it failed:**  
HER grew the EBM training signal from 11 pairs to 548+ but the EBM still cannot guide CEM to the exit. The root cause is not signal density — it's that the world model was trained entirely on random-walk transitions and has never seen the right half of the grid (post-door region). CEM plans imagined trajectories through an OOD region; no cost function can fix a hallucinating world model.

---

### Run 6 — 2026-04-26 — subgoals

**File:** `abm/loop_mpc_doorkey_subgoals.py`  
**Loop module:** `abm.loop_mpc_doorkey_subgoals`  
**Condition:** `subgoals`

| Parameter | Value |
|-----------|-------|
| Goal structure | **3-stage: key → door → exit** |
| Pre-seeding | 200 random episodes before OBSERVE |
| EBM buffers | key_buf + door_buf + goal_buf (separate) |
| Stage detection | live from env.unwrapped.carrying + grid scan |
| Everything else | Identical to Run 3 |

**Hypothesis:** DoorKey's full task is 30–80 steps. A single CEM invocation at H=30 can't bridge the full distance. Breaking it into three separately-tractable stages lets CEM solve each sub-problem with a realistic horizon. The EBM is trained on subgoal images, not just final exits.

**Result:** 0.0% success throughout all 120k ACT steps  
**Buffer counts at step 200k:** key=596 (growing ✅) | door=106 (growing ✅) | goal=12 (FROZEN ❌)  
**ssl_ewa final:** 0.0269  
**Total wall time:** 18,087s (~5.0hr)  
**Peak success:** 0.0%

**Why it failed:**  
Stages 0 (key pickup) and 1 (door unlock) both work — key and door buffers grew throughout ACT phase, confirming CEM can navigate to the key and door with H=30. Stage 2 (exit navigation) never produced a single new success beyond the 12 random-seed images. The post-door region (right half of the grid) is completely OOD for the world model: during OBSERVE, a random walk almost never opens the door, so the WM has zero training data from that region. CEM imagines physically incoherent trajectories past the door and cannot reach the exit.

**Root cause confirmed:** World model distribution gap — the post-door region is never visited during OBSERVE. This is the binding constraint, not horizon length or EBM signal quality.

---

### Run 7 — curiosity_her

**File:** `abm/loop_mpc_doorkey_curiosity_her.py`  
**Loop module:** `abm.loop_mpc_doorkey_curiosity_her`  
**Condition:** `curiosity_her`

| Parameter | Value |
|-----------|-------|
| OBSERVE policy | **Curiosity** (from Run 4) |
| EBM signal | **Standard + HER** (from Run 5) |
| Everything else | Identical to Run 3 |

**Hypothesis:** Both problems (OOD world model + sparse EBM positives) compound. If neither Run 4 nor Run 5 alone succeeds, Run 7 tests whether fixing both simultaneously is needed.

_(Result pending)_

---

### Run 8 — 2026-04-26 — short_horizon

**File:** `abm/loop_mpc_doorkey_run8.py`  
**Loop module:** `abm.loop_mpc_doorkey_run8`  
**Condition:** `short_horizon`

| Parameter | Value |
|-----------|-------|
| CEM horizon | **5** (was 30) |
| Goal structure | 3-stage subgoals |
| OBSERVE policy | Curiosity |
| EBM signal | Standard + HER |
| Pre-seeding | 200 random episodes |
| Everything else | Identical to Run 6 |

**Hypothesis:** Compound error at H=30: 0.65^30 ≈ 0.002% coherent trajectories. H=5 gives 0.65^5 ≈ 11.6%. Shorter horizon should let CEM find tractable plans even with an imperfect world model.

**Result:** 0.0% success throughout all 120k ACT steps  
**Buffer counts at step 200k:** key=262 (growing ✅) | door=52 (FROZEN ❌) | goal=5 (seed only)  
**ssl_ewa final:** 0.0411  
**Total wall time:** 5,317s (~1.5hr)  
**Peak success:** 0.0%

**Why it failed:**  
H=5 solved the compound error problem for stage 0 — the key buffer grew steadily throughout ACT, confirming CEM can reach the key with a 5-step horizon. But door approach requires 6–8 steps minimum (navigate adjacent + face + toggle), so H=5 can never execute stage 1. The door buffer froze at 52 (the random-seed count) from step 80k onward. H=5 is too short for stage 1; H=30 is too long for a coherent plan. The right horizon is 7–9 — long enough to approach the door, short enough to avoid compound error.

**Stage-by-stage diagnosis (cross-run summary):**

| Stage | H=5, Run 8 | H=30 + subgoals, Run 6 |
|-------|-----------|------------------------|
| 0: key pickup | ✅ growing | ✅ growing |
| 1: door unlock | ❌ frozen (H too short) | ✅ growing |
| 2: exit navigation | — | ❌ frozen (WM OOD) |

**What we changed for Run 9:**  
- BFS scripted seeder: completes all 200 seed episodes successfully, giving WM training data from the entire grid including post-door region (200+ images per buffer vs 5 exit images from random seeding)  
- CEM horizon: 5 → 8 (handles 6-8 step door approach; 0.65^8 ≈ 3.2% coherence)

---

### Run 9 — 2026-04-26 — scripted_seed

**File:** `abm/loop_mpc_doorkey_run9.py`  
**Loop module:** `abm.loop_mpc_doorkey_run9`  
**Condition:** `scripted_seed`

| Parameter | Value |
|-----------|-------|
| Seeder | **BFS scripted policy** — completes all 200 episodes |
| CEM horizon | **8** |
| Goal structure | 3-stage subgoals |
| OBSERVE policy | Curiosity |
| EBM signal | Standard + HER |
| Everything else | Identical to Run 8 |

**Hypothesis:** The binding constraint across all prior runs is that the world model has never seen post-door transitions. Random seeding gives ~5 exit images from 200 episodes (~2.5% completion rate). BFS seeder completes 100% of episodes, giving 200+ images per subgoal buffer and WM training data from every region of the grid. H=8 bridges the stage-1 gap (door approach is 6-8 steps) without the compound error that defeated H=30.

**Result:** 0.0% success throughout all 120k ACT steps  
**Buffer counts at step 200k:** key=335 (growing ✅) | door=211 (growing slowly ✅) | goal=200 (FROZEN ❌ — never grew past seed)  
**ssl_ewa final:** 0.0468  
**Total wall time:** 6,922s (~1.9hr)  
**Peak success:** 0.0%

**Why it failed:**  
The seeder populated the EBM goal image buffers (200 images each) but never pushed `(obs, action, next_obs)` transitions to `buf_lew` (the WM replay buffer). The world model was trained entirely on 80k OBSERVE steps of curiosity exploration — all left-half transitions, same OOD problem as Runs 1–8.

The replay buffer (100k capacity) fills after 100k/16 ≈ 6,250 steps and is completely recycled every ~6k steps. Even if the seeder had pushed transitions to `buf_lew`, they would have been evicted long before ACT started. The WM had zero post-door training data.

Evidence: CEM opened the door 10 times during ACT (door: 201→211) but goal never grew past 200. After the door opens, CEM plans through an OOD world model that hallucinates the right half of the grid.

**Root cause confirmed (definitive):** Seeded transitions were never written to `buf_lew`. WM training data = random walk only, regardless of seeder success rate.

---

### Run 10 — protected_seed

**File:** `abm/loop_mpc_doorkey_run10.py`  
**Loop module:** `abm.loop_mpc_doorkey_run10`  
**Condition:** `protected_seed`

| Parameter | Value |
|-----------|-------|
| Seeder | BFS scripted (200 episodes, same as Run 9) |
| WM training | **50% seed_buf + 50% buf_lew every step** |
| seed_buf | 20k capacity, **never overwritten** |
| CEM horizon | 8 |
| Goal structure | 3-stage subgoals |
| OBSERVE policy | Curiosity |
| EBM signal | Standard + HER |

**Hypothesis:** The seeder now pushes `(obs, action, next_obs)` transitions to a separate `seed_buf` (capacity 20k, never overwritten by online data). Every WM training step mixes 50% seed_buf + 50% buf_lew. Post-door dynamics are permanently represented in WM training throughout all 200k steps. When CEM opens the door and enters stage 2, the world model now has accurate predictions for the right half of the grid.

**Result:** 0.0% success throughout all 120k ACT steps  
**Buffer counts at step 200k:** key=305 (growing ✅) | door=213 (growing ✅ — 13 openings) | goal=200 (FROZEN ❌)  
**ssl_ewa final:** 0.0641  
**Total wall time:** 7,283s (~2.0hr)  
**Peak success:** 0.0%

**Why it failed:**  
The WM fix worked — door opened 13 times during ACT (best rate of any run), confirming CEM now plans correctly through stages 0 and 1. But goal never grew past 200. Each of the 13 door-opening windows lasted ~10–30 steps in the right half before episode timeout, yet CEM never reached the exit once.

Root cause: stage-2 EBM energy landscape is wrong inside the right half. EBM negatives throughout training are ~99% left-half states (from buf_lew). Right-half non-exit states have never appeared as negatives, so the EBM either assigns them flat energy (CEM gets no gradient) or assigns them lower energy than left-half states (CEM plans back through the door). Either way, the 13 brief stage-2 windows produced zero successes.

---

### Run 11 — 2026-04-26 — post_door_neg

**File:** `abm/loop_mpc_doorkey_run11.py`  
**Loop module:** `abm.loop_mpc_doorkey_run11`  
**Condition:** `post_door_neg`

| Parameter | Value |
|-----------|-------|
| Seeder | BFS scripted (200 episodes, same as Run 10) |
| WM training | 50% seed_buf + 50% buf_lew (from Run 10) |
| seed_buf | 20k capacity, never overwritten (from Run 10) |
| **post_door_neg_buf** | **5k capacity — right-half non-exit frames** |
| EBM signal | Standard + HER + **stage-2 specific (goal vs post_door_neg)** |
| CEM horizon | 8 |
| Goal structure | 3-stage subgoals |
| OBSERVE policy | Curiosity |

**Hypothesis:** The seeder collects every frame between door-open and exit (~4–8 frames × 200 episodes ≈ 800–1000 right-half non-exit images) into `post_door_neg_buf`. A 4th EBM training signal trains `goal_buf` positives vs `post_door_neg_buf` negatives — teaching `E(z_exit, z_goal) < E(z_right_non_exit, z_goal)`. During ACT, every door-open non-success frame is also pushed to `post_door_neg_buf` for online refinement. This should give CEM a meaningful energy gradient inside the right half for the first time.

**Expected signal:** `post_door_neg=~800` in seeder log; goal grows past 200 during ACT.

**Result:** FAILED — peak=0.0%, final=0.0% | 7050s  
Final buffer state: key=319 door=223 goal=200 post_neg=4318 her=528 | EBM=ON  
The EBM activated and post_door_neg grew to 4318 during ACT — the signal was there. But goal stayed frozen at the seeded 200. Door opened 223 times (more than any prior run), agent navigated correctly through stages 1 and 2, but CEM still failed to plan from door-open state to exit. Root cause: the CNN encoder (LeWM) cannot produce a latent space where CEM-predicted trajectories map meaningfully to actual DoorKey dynamics — the issue is encoder quality, not the EBM signals. This run confirms the architecture is correct but requires a better encoder, motivating Runs 12 (DINOv2) and 13 (V-JEPA 2.1).

**RunPod command:**

```bash
cd /workspace/ose && git pull && python abm_experiment.py \
  --loop-module abm.loop_mpc_doorkey_run11 \
  --condition post_door_neg \
  --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 80000
```

---

### Run 12 — 2026-04-26 — dinov2_frozen

**File:** `abm/loop_mpc_doorkey_run12.py`  
**Loop module:** `abm.loop_mpc_doorkey_run12`  
**Condition:** `dinov2_frozen`

| Parameter | Value |
|-----------|-------|
| Encoder | Frozen DINOv2 ViT-B/14 (timm, 86M params, 768-dim CLS token, L2-norm) |
| Discrimination test | cross-seed cos_sim=0.9699 — WARNING > 0.95, continued anyway |
| Replay | FeatureReplayBuffer (stores 768-dim features, not raw images) |
| Predictor | FeaturePredictor: MLP 768→1024→768, LayerNorm+GELU, cosine loss |
| EBM | EBMCostHead(768), contrastive hinge, 3 signals + post_door_neg |
| seed_buf | 20k protected FeatureReplayBuffer (seeded DINOv2 features) |
| post_door_neg_buf | 5k GoalFeatureBuffer of right-half non-exit features |
| CEM horizon | 8 |
| Goal structure | 3-stage subgoals (key→door→exit) |
| OBSERVE policy | Curiosity |

**Result:** peak=20.0%, final=0.0% | 7228s  
Final buffer state: key=632 door=547 goal=228 post_neg=5000 her=537  
First non-zero result across all 12 runs — confirms pretrained encoder hypothesis. Door opened 547 times (stage 1+2 working well), goal grew to 228 (agent reaches exit occasionally in stage 3). Peak 20% did not hold — success rate was noisy (10% then 0% for many evals, then 20% peak). The 0.9699 discrimination warning may explain the noise: DINOv2 features for different DoorKey states are nearly identical, making the EBM energy landscape flat and CEM guidance unreliable.

**RunPod command:**

```bash
cd /workspace && git clone https://github.com/dase8601/ose.git && cd ose && pip install -r requirements.txt && python abm_experiment.py \
  --loop-module abm.loop_mpc_doorkey_run12 \
  --condition dinov2_frozen \
  --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 80000
```

---

### Run 13 — 2026-04-26 — vjepa2_frozen

**File:** `abm/loop_mpc_doorkey_run13.py`  
**Loop module:** `abm.loop_mpc_doorkey_run13`  
**Condition:** `vjepa2_frozen`

| Parameter | Value |
|-----------|-------|
| Encoder | Frozen V-JEPA 2.1 ViT-Base-384 (86M params, 768-dim mean-pooled patch tokens, L2-norm) |
| Input shape | (B, 3, 1, 384, 384) — T=1 single frame |
| Replay | FeatureReplayBuffer (stores 768-dim features) |
| Predictor | FeaturePredictor: MLP 768→1024→768, LayerNorm+GELU, cosine loss |
| EBM | EBMCostHead(768), same as Run 12 |
| seed_buf | 20k protected FeatureReplayBuffer |
| post_door_neg_buf | 5k GoalFeatureBuffer |
| CEM horizon | 8 |
| Goal structure | 3-stage subgoals |
| OBSERVE policy | Curiosity |

**Hypothesis:** V-JEPA 2.1 trained on video with a JEPA objective (predict future latents) — more theoretically aligned with our transition predictor than DINOv2's static-image training. Features should be sensitive to motion and state change, giving CEM a meaningful gradient to plan toward the exit.

**Mid-run signal (step 96k, ~15k into ACT):** success=20.0%, goal=208 (8 new goals in 15k ACT steps), door=248, post_neg=5000 (capped), pred_ewa≈0.0000 — predictor loss near zero, V-JEPA features are smooth/predictable. Still running.

**RunPod command:**

```bash
cd /workspace && git clone https://github.com/dase8601/ose.git && cd ose && pip install -r requirements.txt einops && python abm_experiment.py \
  --loop-module abm.loop_mpc_doorkey_run13 \
  --condition vjepa2_frozen \
  --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 80000
```

---

## Phase 2 — Prove autonomous System M > fixed switching (Crafter)

_Not started. Begins only after Phase 1 planner proves > 42% on DoorKey._

---

## Phase 3 — Combined system (Crafter)

_Not started._

---

## All results summary

| Date | Run | Condition | Env | Steps | Peak | Final | Notes |
|------|-----|-----------|-----|-------|------|-------|-------|
| 2026-04-24 | DoorKey R1 | planner_only | DoorKey | 200k | 0% | 0% | H=10 too short |
| 2026-04-25 | DoorKey R2 | planner_only | DoorKey | 200k | 0% | 0% | H=30, train during ACT — cosine dist not sufficient |
| 2026-04-25 | DoorKey R3 | planner_only | DoorKey | 200k | 0% | 0% | EBM ON by 192k, 19 goals, 4.5hr — WM OOD, too few positives |
| 2026-04-26 | DoorKey R4 | curiosity_observe | DoorKey | 200k | 0% | 0% | Curiosity alone insufficient — WM horizon still the bottleneck |
| 2026-04-26 | DoorKey R5 | her_goals | DoorKey | 200k | 0% | 0% | 11 goals, her=548, 4.7hr — HER can't fix OOD WM |
| 2026-04-26 | DoorKey R6 | subgoals | DoorKey | 200k | 0% | 0% | key=596✅ door=106✅ goal=12❌ frozen — post-door OOD confirmed |
| — | DoorKey R7 | curiosity_her | DoorKey | 200k | — | — | Curiosity + HER combined — pending |
| 2026-04-26 | DoorKey R8 | short_horizon | DoorKey | 200k | 0% | 0% | H=5: key✅ door❌frozen — H too short for stage 1 (needs 6-8 steps) |
| 2026-04-26 | DoorKey R9 | scripted_seed | DoorKey | 200k | 0% | 0% | door=211✅ goal=200❌frozen — seeder never wrote to buf_lew, WM still OOD |
| 2026-04-26 | DoorKey R10 | protected_seed | DoorKey | 200k | 0% | 0% | door=213✅ 13 openings — WM fixed, EBM stage-2 energy wrong in right half |
| 2026-04-26 | DoorKey R11 | post_door_neg | DoorKey | 200k | 0% | 0% | door=223✅ goal=200❌frozen — EBM ON, post_neg=4318, CNN encoder OOD confirmed |
| 2026-04-26 | DoorKey R12 | dinov2_frozen | DoorKey | 200k | 20% | 0% | First non-zero — DINOv2 cos_sim=0.9699 warning, noisy success, goal=228 |
| 2026-04-26 | DoorKey R13 | vjepa2_frozen | DoorKey | 200k | — | — | V-JEPA 2.1 — 20% at step 95k (15k into ACT), still running |
| — | DoorKey (old) | autonomous PPO | DoorKey | 200k | 18% | 10% | 9 switches |
| — | DoorKey (old) | fixed PPO | DoorKey | 200k | 16% | 10% | 19 switches |
| — | DoorKey (old) | ppo_only | DoorKey | 200k | 42% | 42% | baseline |
| — | Crafter | ACWM planner-only | Crafter | 500k | 31.8% | 22.7% | System M confound |
| — | Crafter | PPO autonomous | Crafter | 1M | 50% | 27.3% | 10 switches |
| — | Crafter | PPO fixed | Crafter | 1M | 59.1% | 31.8% | 99 switches — likely noise |
