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

**Result:** _in progress_  
**Goals collected:** —  
**ssl_ewa at OBSERVE end:** —  
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
| 2026-04-24 | DoorKey R2 | planner_only | DoorKey | 200k | — | — | H=30, train during ACT |
| — | DoorKey (old) | autonomous PPO | DoorKey | 200k | 18% | 10% | 9 switches |
| — | DoorKey (old) | fixed PPO | DoorKey | 200k | 16% | 10% | 19 switches |
| — | DoorKey (old) | ppo_only | DoorKey | 200k | 42% | 42% | baseline |
| — | Crafter | ACWM planner-only | Crafter | 500k | 31.8% | 22.7% | System M confound |
| — | Crafter | PPO autonomous | Crafter | 1M | 50% | 27.3% | 10 switches |
| — | Crafter | PPO fixed | Crafter | 1M | 59.1% | 31.8% | 99 switches — likely noise |
