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

**Result:** _in progress_  
**Goals collected:** —  
**EBM activated at step:** —  
**Peak success:** —  
**Notes:** —

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
| 2026-04-25 | DoorKey R3 | planner_only | DoorKey | 200k | — | — | EBM cost head replaces cosine |
| — | DoorKey (old) | autonomous PPO | DoorKey | 200k | 18% | 10% | 9 switches |
| — | DoorKey (old) | fixed PPO | DoorKey | 200k | 16% | 10% | 19 switches |
| — | DoorKey (old) | ppo_only | DoorKey | 200k | 42% | 42% | baseline |
| — | Crafter | ACWM planner-only | Crafter | 500k | 31.8% | 22.7% | System M confound |
| — | Crafter | PPO autonomous | Crafter | 1M | 50% | 27.3% | 10 switches |
| — | Crafter | PPO fixed | Crafter | 1M | 59.1% | 31.8% | 99 switches — likely noise |
