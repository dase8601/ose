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

---

### Run 5 — (next) her_goals

**File:** `abm/loop_mpc_doorkey_her.py`  
**Loop module:** `abm.loop_mpc_doorkey_her`  
**Condition:** `her_goals`

| Parameter | Value |
|-----------|-------|
| EBM signal | **Standard + HER** |
| HER buffer | All episode-end states (4k capacity) |
| Everything else | Identical to Run 3 |

**Hypothesis:** EBM had only ~23 positive training pairs in 200k steps. HER relabeling treats every episode-end state as "a goal achieved for itself" — E(z_end, z_end) should be low, E(z_random, z_end) should be high. This gives thousands of positive pairs per run instead of 23.

---

### Run 6 — (next) subgoals

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

---

### Run 7 — (next) curiosity_her

**File:** `abm/loop_mpc_doorkey_curiosity_her.py`  
**Loop module:** `abm.loop_mpc_doorkey_curiosity_her`  
**Condition:** `curiosity_her`

| Parameter | Value |
|-----------|-------|
| OBSERVE policy | **Curiosity** (from Run 4) |
| EBM signal | **Standard + HER** (from Run 5) |
| Everything else | Identical to Run 3 |

**Hypothesis:** Both problems (OOD world model + sparse EBM positives) compound. If neither Run 4 nor Run 5 alone succeeds, Run 7 tests whether fixing both simultaneously is needed.

---

**RunPod commands (run in order):**

```bash
# Run 4
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_curiosity \
  --condition curiosity_observe --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 80000

# Run 5
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_her \
  --condition her_goals --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 80000

# Run 6
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_subgoals \
  --condition subgoals --device cuda --env doorkey \
  --steps 200000 --n-envs 16 --observe-steps 80000

# Run 7
python abm_experiment.py --loop-module abm.loop_mpc_doorkey_curiosity_her \
  --condition curiosity_her --device cuda --env doorkey \
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
| — | DoorKey R4 | curiosity_observe | DoorKey | 200k | — | — | Curiosity OBSERVE — pending |
| — | DoorKey R5 | her_goals | DoorKey | 200k | — | — | HER EBM signal — pending |
| — | DoorKey R6 | subgoals | DoorKey | 200k | — | — | 3-stage subgoals + seeding — pending |
| — | DoorKey R7 | curiosity_her | DoorKey | 200k | — | — | Curiosity + HER combined — pending |
| — | DoorKey (old) | autonomous PPO | DoorKey | 200k | 18% | 10% | 9 switches |
| — | DoorKey (old) | fixed PPO | DoorKey | 200k | 16% | 10% | 19 switches |
| — | DoorKey (old) | ppo_only | DoorKey | 200k | 42% | 42% | baseline |
| — | Crafter | ACWM planner-only | Crafter | 500k | 31.8% | 22.7% | System M confound |
| — | Crafter | PPO autonomous | Crafter | 1M | 50% | 27.3% | 10 switches |
| — | Crafter | PPO fixed | Crafter | 1M | 59.1% | 31.8% | 99 switches — likely noise |
