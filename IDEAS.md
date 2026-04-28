# Research Ideas & Literature Context

_Last updated: 2026-04-28_

---

## Where This Project Sits in the Field

This project is a direct implementation attempt of LeCun's A-B-M architecture
(arXiv 2603.15381, March 2026) on a controlled toy environment (MiniGrid-DoorKey-6x6).
The LeWM paper (arXiv 2603.19312, March 2026) was published six weeks ago — we are
building on work that barely exists in the literature yet.

The core claim we are testing: **world model + CEM planning > pure RL (42% PPO baseline)**
on a short-horizon discrete task with sparse reward.

After 24 runs, the picture is:
- Stages 0/1 (key pickup, door opening): CEM+EBM works, confirmed by Run 23
- Stage 2 (door→exit): CEM fails due to compound prediction error at H=8 and H=8
- Predictor degrades during ACT as replay buffer distribution shifts
- Best result so far: peak=25% (Runs 18, 20, 23), never sustained above 42%

---

## Most Relevant Papers Found (2025-2026)

### 1. LeWorldModel — arXiv 2603.19312 (March 2026)
**Authors:** Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero

The official published paper behind "LeWM." End-to-end JEPA from raw pixels using
only two loss terms: prediction loss + Gaussian regularization. Reduces hyperparameter
count from 6 to 1. Achieves **48× faster planning** than DINO-WM while competitive on
control tasks. ~15M trainable parameters, single GPU.

**What it means for us:** This is the system we are implementing. The paper validates
the symbolic predictor + CEM approach as sound. Our 5-dim symbolic state is a
deliberately simplified version to understand failure modes before scaling.

---

### 2. Hierarchical Planning with Latent World Models — arXiv 2604.03208 (April 3, 2026)
**Authors:** Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria,
Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, Nicolas Ballas

Multi-scale hierarchical planning operating at multiple temporal scales in learned latent
dynamics. Directly reduces compounding prediction errors and exponential search space
growth of single-level planners.

**Results:** 70% success on real-world pick-and-place (0% for single-level flat planner),
4× less planning-time compute in simulation, zero-shot generalization to new environments.

**What it means for us:** This paper is the architectural answer to our H=8 and H=12
failures. Flat CEM at any fixed horizon accumulates prediction error exponentially. The
solution is not a longer or shorter H — it is two planning levels:
- **Low level:** short-horizon (H=3-4), precise, high-accuracy predictions for navigation
- **High level:** abstract long-horizon predictions at coarser temporal scale for subgoal sequencing

Run 21 showed H=12 is worse than H=8 — confirming that flat CEM cannot scale horizon
with an imperfect predictor. Hierarchical planning is the principled fix.

**Priority: High. This is the next major architectural idea.**

---

### 3. Value-guided Action Planning with JEPA — arXiv 2601.00844 (Dec 2025)
**Authors:** Matthieu Destrade, Oumayma Bounou, Quentin Le Lidec, Jean Ponce, Yann LeCun
**Venue:** World Modeling Workshop 2026 (Mila)

Shapes the representation space so that the goal-conditioned value function is approximated
by embedding distance. Enables better goal-directed planning without a separate EBM cost head.

**What it means for us:** Our EBM cost head is an approximation of the goal-conditioned
value function. This paper suggests training the predictor representation explicitly so
that cosine distance in latent space IS the value function — removing the need for a
separate EBM module entirely. Stages 0/1 and 2 would all use the same metric without
needing contrastive training.

**Priority: Medium. Interesting alternative to the EBM approach if EBM limitations persist.**

---

### 4. V-JEPA 2 — arXiv 2506.09985 (June 2025)
**Authors:** Mido Assran, Yann LeCun, Nicolas Ballas, and 30+ collaborators

1.2B-parameter world model trained on 1M+ hours of internet video. Action-conditioned
predictor fine-tuned on 62 hours of DROID robot data. Uses CEM planning at inference.
**80% success picking/placing novel objects in new environments** (vs 15% for Octo baseline).

**What it means for us:** V-JEPA 2 validates CEM as the right planning algorithm at scale.
The DoorKey failure (CEM + compound errors) is a scale and encoder quality problem, not a
fundamental CEM problem. V-JEPA 2's 1.2B parameters give it much more accurate per-step
predictions — compound errors stay small even at longer horizons.

---

### 5. DINO-WM — arXiv 2411.04983 (Nov 2024, revised Feb 2025)
**Authors:** Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

World models built on frozen DINOv2 patch features + MPC for zero-shot planning. No pixel
reconstruction. Outperforms prior SOTA on 6 environments including maze and object
manipulation.

**What it means for us:** Our Runs 12-16 attempted this approach (DINOv2/V-JEPA frozen
features + CEM). We found cos_sim≈0.999 between adjacent DoorKey frames — consistent with
DINO-WM requiring visual diversity that a 6×6 synthetic grid cannot provide. DINO-WM works
on real-world scenes, not toy grids.

---

### 6. CompACT / Planning in 8 Tokens — arXiv 2603.05438 (March 2026, CVPR 2026)
**Authors:** Dongwon Kim, Gawon Seo, Jinsung Lee, Minsu Cho, Suha Kwak

Reduces observations to 8 discrete tokens using frozen vision foundation models. World
models in this compact discrete space achieve **40× faster planning** with competitive
performance. Uses vector quantization with 8 discrete codes.

**What it means for us:** Our 5-dim symbolic state is analogous — compact discrete
representation. The difference is our symbolic state is hand-engineered (we know the 5
variables); CompACT learns the compact representation from pixels. If moving to visual
observations, CompACT-style tokenization is worth considering over full DINOv2 features.

---

### 7. DreamerV3 — arXiv 2301.04104 (Jan 2023)
**Authors:** Danijar Hafner, Timothy Lillicrap, et al.

General model-based RL across 150+ tasks with single hyperparameter configuration.
Categorical cross-entropy loss, discrete one-hot latent encoding, imagined rollouts for
policy improvement.

**What it means for us:** DreamerV3 solves the predictor degradation problem differently —
it trains the world model AND the policy jointly via imagined trajectories. The policy
adapts to what the world model can predict. Our approach trains the predictor separately
from the CEM planner, creating distribution mismatch during ACT. DreamerV3-style joint
training would naturally avoid the pred_ewa degradation seen in Run 23.

---

### 8. TD-MPC2 — arXiv 2310.16828 (ICLR 2025)

Latent implicit world models + policy-guided MPC. 300× parameter scaling. Outperforms
DreamerV3 on hard continuous control tasks. Uses policy guidance to focus CEM sampling
around good action sequences rather than uniform random sampling.

**What it means for us:** Policy-guided CEM (TD-MPC2 style) would dramatically improve
CEM efficiency. Instead of 512 uniform random action sequences, sample from a learned
policy distribution centered on good actions. This is particularly valuable for stage 2
(door→exit) where most random H=8 rollouts are uninformative.

---

## Concrete Architectural Ideas for Next Runs

### Idea A: Hierarchical CEM (most impactful, highest build cost)
Implement two-level planning based on arXiv 2604.03208:
- **High level (H=2, abstract):** Plan 2 abstract subgoals from current state
- **Low level (H=4, precise):** Execute each subgoal with 4-step accurate CEM

The high-level planner uses a slower-changing abstract representation (e.g., stage flags:
has_key, door_open). The low-level planner uses the full 5-dim symbolic state. This
directly addresses compound error growth without requiring a longer flat horizon.

Estimated build: 2-3 hours. High probability of unlocking stage 2.

### Idea B: Joint predictor-policy training (DreamerV3-style)
Train a small policy network alongside the predictor using imagined rollouts. The policy
provides a prior for CEM sampling (TD-MPC2 style). Prevents pred_ewa degradation because
the predictor trains on data generated by its own predictions, not just environment data.

Estimated build: 3-4 hours. Addresses the root cause of ACT degradation.

### Idea C: Value-shaped representation (arXiv 2601.00844 style)
Add a value head to the predictor that estimates distance-to-goal in latent space. Train
the predictor so that embedding distance approximates goal distance. Replace EBM cost head
with this value-shaped distance metric. Removes the EBM saturation/softplus complexity
entirely.

Estimated build: 1-2 hours. Clean alternative to the EBM approach.

### Idea D: Frozen predictor with larger seed buffer (Run 24 hypothesis, extended)
Run 24 tests frozen predictor + 200 seeded episodes. If it shows promise:
- Increase N_SEED_EPS from 200 → 1000 (more diverse scripted episodes to train from)
- Increase SEED_BUF_CAPACITY from 20k → 50k
- This gives the frozen predictor a richer OBSERVE training set

Low build cost, directly addresses the distribution shift problem.

### Idea E: PPO for stage 3 + CEM+EBM stages 0/1 (publishable version)
Replace scripted BFS stage 3 with a small PPO policy trained only on post-door transitions.
World model handles the hard planning (subgoal discovery: key, door). PPO handles the easy
navigation (door→exit). Consistent with LeCun's architecture: world model for high-level
planning, learned policy for execution.

This is the publishable result if it beats 42%.
Estimated build: 2-3 hours. Requires PPO actor-critic head + stage-2 only training.

---

## Current Experiment Status

| Run | Condition | Status | Peak | Key Finding |
|-----|-----------|--------|------|-------------|
| R20 | two-phase EBM | Done | 25% | pred_ewa 0.0144 record, ceiling at 226 |
| R21 | H=12 stage 2 | Done | 5% | Longer horizon = worse (compound error) |
| R23 | scripted BFS s2 | Running | 25% | Goal past 226 — stage 3 confirmed as bottleneck |
| R24 | frozen pred + BFS | Running | TBD | Testing if pred degradation causes ACT ceiling |

**Milestone not yet achieved:** beat 42% PPO baseline on DoorKey-6x6.

---

## Field Context Summary

The research confirms this project is timely and aligned with the frontier:
- LeWM published March 2026 (6 weeks ago)
- Hierarchical planning paper published April 3, 2026 (25 days ago)
- V-JEPA 2 validates CEM planning at scale (80% robot success)
- The specific failure modes found (compound error, EBM saturation, visual encoder
  insensitivity) are real problems the field is actively solving

The gap between our toy implementation (5-dim symbolic, DoorKey) and SOTA (1.2B V-JEPA 2,
real robots) is large — but that gap is exactly the right place to study failure modes.
Understanding why flat CEM fails at H=8 in a 6×6 grid is precisely the mechanistic
understanding needed before scaling.

---

## Path to a Publishable Result

**Minimum viable claim:** CEM+EBM (world model planning) handles multi-step subgoal
discovery (key→door) more efficiently than RL on DoorKey, while RL (PPO) handles
short-range navigation (door→exit). Combined system beats 42% PPO baseline.

**To achieve this:** Build Run 25 (PPO stage 3 + CEM+EBM stages 0/1). If it beats 42%,
the paper claim is: "World model planning outperforms RL for the hard subgoal discovery
problem; a learned policy handles the easy navigation step."

**Stronger claim (harder to achieve):** Pure CEM planning for all stages beats 42%.
Requires either hierarchical planning (Idea A) or value-shaped representations (Idea C).
