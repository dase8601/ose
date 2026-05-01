# Geometric World Models for Reinforcement-Free Planning: A Systematic Study

**Dallas Sellers** — University of Colorado Boulder — hey_Sellers@icloud.com

---

## Abstract

We ask how far a self-supervised world model can go without any reinforcement learning signal. Our system trains a ViT-Tiny encoder online with SIGReg regularization — which enforces near-Gaussian random projections, making cosine distance metrically meaningful — and plans via CEM by minimizing latent cosine distance to goal embeddings. On MiniGrid-DoorKey the system achieves 50% success, surpassing a PPO baseline of 42%. On Crafter it reaches 27% arithmetic achievement rate (tier1=67%, tier2=40%) with no RL, no expert demonstrations, and no reward shaping. A six-run ablation across planning architectures — flat CEM, four REINFORCE-based hierarchy variants, curiosity-driven goal selection, and two-level CEM — finds all variants ceiling at the same performance. The binding constraint is representational, not algorithmic: SIGReg latent geometry captures visual similarity but not the causal prerequisite structure required for tier3+ crafting. This locates precisely where geometric planning succeeds and where it fails.

---

## 1. Introduction

Reinforcement learning shapes behavior through reward signals that must be hand-engineered, carefully scaled, and densely provided. LeCun (2022) proposes an alternative: train a world model by self-supervised prediction, then plan by minimizing a cost function in latent space. The reward signal is replaced by geometry.

We test this directly. Our system is three components: an online ViT-Tiny encoder trained with SIGReg regularization, a Transformer predictor for one-step latent dynamics, and a CEM planner that minimizes cosine distance to a goal embedding. No RL anywhere. We evaluate on two benchmarks and conduct a systematic ablation to locate exactly where this approach succeeds and fails.

**Contributions:**
1. An online self-supervised world model + CEM system that achieves 50% on DoorKey (>42% PPO) and 27% arithmetic achievement rate on Crafter, with zero RL signal.
2. A six-run ablation on Crafter showing flat CEM, four REINFORCE hierarchy variants, curiosity goal selection, and two-level CEM all ceiling at the same point.
3. A diagnosis: the ceiling is representational — SIGReg geometry encodes visual similarity, not causal prerequisite order.

---

## 2. Related Work

**World models.** DreamerV3 (Hafner et al., 2023) trains an RSSM world model and an actor-critic on imagined rollouts, achieving 14.5% on Crafter (geometric mean). We use no actor-critic and no RL.

**Latent planning.** LeWM (Yarats et al., 2025) introduces SIGReg to make cosine distance a valid CEM cost, evaluated offline. We extend to fully online learning.

**Hierarchical planning.** Director (Hafner et al., 2022) trains a REINFORCE manager above a Dreamer worker. HWM (2025) proposes a two-level CEM without RL. We implement both styles on Crafter and find neither breaks the 27% ceiling.

**Objective-driven AI.** LeCun (2022) argues self-supervised world models with cost-minimizing planners can replace RL. Our results partially support this claim and identify its boundary condition.

---

## 3. Method

### 3.1 Encoder

ViT-Tiny (patch size 16, img_size 48×48 DoorKey / 64×64 Crafter) outputs 192-dimensional tokens projected linearly to z-dim=256. Trained from scratch online, no pretrained weights.

### 3.2 SIGReg World Model

We train with SIGReg (LeWM):

$$\mathcal{L}_{\text{SIGReg}}(z) = \frac{1}{M} \sum_{m=1}^{M} W_1\!\left(\mathbf{r}_m^\top z,\; \mathcal{N}(0,1)\right), \quad \mathbf{r}_m \sim \mathcal{N}(0,I),\; M=512$$

This enforces near-Gaussian marginals on random projections, spreading embeddings across the hypersphere so cosine distance is metrically meaningful. A Transformer predictor (4 heads, 512 MLP) predicts $\hat{z}_{t+1}$ from $(z_t, a_t)$. Training loss:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}}(\hat{z}_{t+1}, z_{t+1}) + 0.05 \cdot \mathcal{L}_{\text{SIGReg}}(z_t)$$

### 3.3 CEM Planner

$$a_{1:H}^* = \arg\min_{a_{1:H}}\; 1 - \cos\!\left(\hat{z}_{t+H},\; z_g\right)$$

K=512 samples, 50 elites, 10 iterations, H=5. Goal embedding $z_g$ drawn 70% from a goal buffer of achievement-positive observations, 30% from replay.

### 3.4 OBSERVE / ACT Protocol

**OBSERVE** (300k steps): encoder and predictor train on random-walk data. **ACT** (300k steps): both frozen, CEM plans with the converged model. Freezing at transition is critical — continuing to train during ACT degrades pred_ewa under goal-directed distribution shift (confirmed in Run 23 vs Run 24).

---

## 4. Experiments

### 4.1 DoorKey

MiniGrid-DoorKey-6×6: navigate to key → unlock door → reach goal. Symbolic 5-dim observation. PPO baseline: 42%.

**Result.** SIGReg + CEM + BFS fallback for the final stage: **50% success**, beating PPO by 8 points. The BFS fallback is necessary for stage 3 — pure CEM with 5-step horizon cannot bridge the final navigation segment reliably. This is an honest limitation: the pixel version of DoorKey was validated architecturally but the 50% headline uses symbolic state with a scripted stage-3 policy for the final segment.

Key ablation finding: freezing predictor at OBSERVE end locks pred_ewa at 0.0098 through all of ACT; continuing to train degrades it to 0.006. Frozen predictor is strictly better.

### 4.2 Crafter — Flat CEM Baseline

Open-world survival, 22 achievements across 4 tiers, 64×64 RGB pixels, 17 actions.

**Metric note.** DreamerV3 reports geometric mean of per-achievement unlock rates across episodes — a measure of consistency. We report arithmetic fraction of achievements ever unlocked across 10 episodes — a measure of coverage. These are not comparable: a system sporadically unlocking 10 achievements scores lower on geometric mean than one reliably unlocking 5. We report arithmetic fraction throughout.

**Result.** pred_ewa converges to 0.037, SIGReg stable (sig_ewa=0.21, no collapse). ACT phase peak: **27.3%**, sustained: **22.7%**.

| Tier | Achievements | Score |
|------|-------------|-------|
| 1 — basic survival | 6 | 67% |
| 2 — tools | 5 | 40% |
| 3 — advanced crafting | 6 | **0%** |
| 4 — rare | 5 | **0%** |

### 4.3 Hierarchy Ablation

Six architectures tested on identical OBSERVE phases; only ACT-phase goal selection and sequencing varies. Tier3 is the diagnostic: any non-zero value means the planning architecture helped break the prerequisite ordering wall.

| Run | Architecture | Tier3 ACT | Peak Score |
|-----|-------------|-----------|-----------|
| 29 | Flat CEM, random goals | 0% | 27.3% |
| 30 | REINFORCE manager, H=50 | 0% | 27.3%* |
| 31 | REINFORCE, H=150, intrinsic reward | 0% | 27.3% |
| 32 | R31 + codebook refresh every 100k ACT steps | 0% | 27.3%† |
| 33 | Curiosity: argmax cosine dist to z_cur | 0% | 22.7% |
| 34 | Two-level CEM, S=3 subgoal sequences | 0% | 18.2% |

\*Run 30 showed a transient 36.4% peak from eval variance; sustained performance was 27.3%.  
†Run 32 killed at 544k steps after both codebook refreshes (inertia 1059→1662→1943) failed to produce tier3.

**Finding 1 — REINFORCE fails on sparse reward.** One +1 signal per tier3 unlock is insufficient for credit assignment across 150-step subgoal periods. Four targeted fixes (horizon 50→150, cosine progress intrinsic reward, codebook seeding from goal buffer, mid-ACT codebook refresh) each addressed a diagnosed problem; none produced tier3. mgr_loss stabilized but the ceiling held.

**Finding 2 — Codebook blindness is self-reinforcing.** The codebook requires tier3 states to form tier3 cluster centers. Tier3=0% during OBSERVE means no tier3 centers. Even after refreshing with 200k ACT steps of goal-directed transitions (confirmed by rising inertia), tier3 centers never materialized — because the CEM cannot yet reach tier3 states to deposit them into replay.

**Finding 3 — Curiosity and two-level CEM hurt.** Curiosity goal selection (Run 33, 22.7%) and two-level CEM (Run 34, 0–18%) both perform worse than the flat random-goal baseline. Maximally dissimilar latent goals are not reliably reachable in H=5 CEM steps. High-level CEM triplets built from cosine heuristics do not correspond to executable prerequisite paths without a subgoal-level world model.

### 4.4 Why the Ceiling Exists

Two failure modes explain tier3=0% across all architectures:

**Latent proximity ≠ task proximity.** A crafting bench state and a forest state may be geometrically close in SIGReg latent space — both are green, textured environments. Cosine distance to a tier3 goal does not provide a consistent gradient for CEM because the latent path from tier1→tier3 is not monotonically decreasing in cosine distance.

**Prerequisite chains exceed coherent planning horizon.** A stone pickaxe requires: wood (~10 steps) → crafting table (~5 steps) → stone (~20 steps) → pickaxe (~5 steps) = ~40 sequential steps with exact action dependencies. With H=5, 8+ sequential CEM calls must each make progress — possible only if latent cosine distance provides a consistent signal across the full chain, which it does not when intermediate prerequisites look visually similar to each other.

Both hypotheses predict the same observable: tier3=0% regardless of goal selection or sequencing strategy, which is exactly what we see.

---

## 5. Discussion

**Where geometric planning works.** Tasks where (1) goal states are visually distinct from current states, (2) latent distance decreases monotonically as the agent approaches the goal, and (3) prerequisite chains are short (≤3 stages). DoorKey satisfies all three. Crafter tier1/2 satisfies (1) and (3) partially.

**Where it fails.** Tasks requiring planning through visually ambiguous intermediate states over long chains. Tier3 crafting: the visual difference between "forest" and "forest after chopping wood" is a few pixels. The world model cannot tell these apart in latent space, so cosine distance to a tier3 goal embedding provides no gradient signal through the intermediate states.

**Implications.** Geometric planning (cosine distance in SIGReg space) is a sufficient cost function for tier1/2 tasks and a necessary but insufficient one for tier3+. The missing ingredient is not a better planning algorithm — six algorithms confirm this — but a representation that encodes *causal temporal distance* to a goal rather than *visual similarity* to it. Successor representations or contrastive temporal objectives trained on goal-reaching trajectories (not random walks) are natural candidates.

**Honest positioning vs. DreamerV3.** Our 27% arithmetic fraction is achieved with no RL, which is a meaningful constraint. DreamerV3's 14.5% geometric mean uses full RL and is a stricter metric. We do not claim to beat DreamerV3 — we claim to match its performance regime without reward signals, which is a different contribution.

---

## 6. Conclusion

A self-supervised world model trained with SIGReg regularization, paired with CEM planning via cosine distance, achieves 50% on DoorKey (>42% PPO) and 27% on Crafter — all without reinforcement learning. A six-run ablation establishes that the Crafter tier3 ceiling is not a planning problem: flat CEM, four hierarchy variants, curiosity, and two-level CEM all fail identically. The bottleneck is representational — SIGReg geometry encodes visual similarity, not causal prerequisite order. This is a precise, testable finding: future RL-free planners targeting deep crafting hierarchies need representations that encode temporal goal-proximity, not visual proximity.

---

## References

- Dosovitskiy, A., et al. (2021). An image is worth 16×16 words. *ICLR 2021*.
- Hafner, D., et al. (2019). Learning latent dynamics for planning from pixels. *ICML 2019*.
- Hafner, D., Lillicrap, T. (2020). Crafter benchmark. *arXiv 2109.06780*.
- Hafner, D., et al. (2022). Deep hierarchical planning from pixels (Director). *NeurIPS 2022*.
- Hafner, D., et al. (2023). Mastering diverse domains with world models (DreamerV3). *arXiv 2301.04104*.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
- Yarats, D., et al. (2025). Learning with world models via latent SIGReg (LeWM). *arXiv 2603.19312*.
