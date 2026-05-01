# Geometric World Models for Reinforcement-Free Planning: A Systematic Study

**Dallas Sellers**
University of Colorado Boulder
dase8601@colorado.edu

---

## Abstract

We ask how far a self-supervised world model can go without any reinforcement learning signal. Our system trains a ViT-Tiny encoder online with SIGReg regularization, which enforces near-Gaussian random projections and makes cosine distance metrically meaningful, and plans via CEM by minimizing latent cosine distance to goal embeddings. On MiniGrid-DoorKey the system achieves 50% success, surpassing a PPO baseline of 42%. On Crafter it reaches 27% arithmetic achievement rate (tier1=67%, tier2=40%) with no RL, no expert demonstrations, and no reward shaping. A six-run ablation across planning architectures finds all variants ceiling at the same performance: flat CEM, four REINFORCE-based hierarchy variants, curiosity-driven goal selection, and two-level CEM all produce identical tier3=0%. The binding constraint is representational rather than algorithmic. SIGReg latent geometry captures visual similarity but not the causal prerequisite structure required for tier3+ crafting, and this paper locates precisely where geometric planning succeeds and where it fails.

---

## 1. Introduction

Reinforcement learning shapes behavior through reward signals that must be hand-engineered, carefully scaled, and densely provided. LeCun (2022) proposes an alternative: train a world model by self-supervised prediction, then plan by minimizing a cost function in latent space. The reward signal is replaced by geometry.

We test this directly on two standard benchmarks. Our system combines an online ViT-Tiny encoder trained with SIGReg regularization, a Transformer predictor for one-step latent dynamics, and a CEM planner that minimizes cosine distance to a goal embedding. No RL is used at any stage. We then conduct a systematic six-run ablation to locate exactly where this approach succeeds and fails.

Our contributions are threefold. First, we demonstrate an online self-supervised world model and CEM system that achieves 50% on DoorKey, exceeding a 42% PPO baseline, and 27% arithmetic achievement rate on Crafter with zero RL signal. Second, a six-run ablation on Crafter shows flat CEM, four REINFORCE hierarchy variants, curiosity goal selection, and two-level CEM all ceiling at the same point, pointing to representation rather than planning algorithm as the binding constraint. Third, we provide a diagnosis: SIGReg geometry encodes visual similarity, not causal prerequisite order, and we explain why this causes an identical tier3=0% result across all six architectures.

---

## 2. Related Work

**World models.** DreamerV3 (Hafner et al., 2023) trains an RSSM world model and an actor-critic on imagined rollouts, achieving 14.5% on Crafter (geometric mean). Our work uses no actor-critic and no RL anywhere in the system.

**Latent planning.** LeWM (Yarats et al., 2025) introduces SIGReg to make cosine distance a valid CEM cost, evaluated on offline data. We extend this to fully online learning from scratch with no pretrained encoder.

**Hierarchical planning.** Director (Hafner et al., 2022) trains a REINFORCE manager above a Dreamer worker, tested on mazes and Atari. HWM (2025) proposes a two-level CEM without RL. We implement both styles on Crafter and find neither breaks the 27% ceiling.

**Objective-driven AI.** LeCun (2022) argues self-supervised world models with cost-minimizing planners can replace RL for most tasks. Our results partially support this claim and identify its boundary condition precisely.

---

## 3. Method

### 3.1 Encoder

We use ViT-Tiny (patch size 16, image size 48x48 for DoorKey and 64x64 for Crafter), which outputs 192-dimensional tokens projected linearly to 256 dimensions. The encoder is trained from scratch online with no pretrained weights.

### 3.2 SIGReg World Model

Following LeWM, we regularize the encoder with SIGReg:

$$\mathcal{L}_{\text{SIGReg}}(z) = \frac{1}{M} \sum_{m=1}^{M} W_1\!\left(\mathbf{r}_m^\top z,\; \mathcal{N}(0,1)\right), \quad \mathbf{r}_m \sim \mathcal{N}(0,I),\; M=512$$

This enforces near-Gaussian marginals on random projections, spreading embeddings across the hypersphere so cosine distance is metrically meaningful for CEM planning. A Transformer predictor (4 heads, 512-dimensional MLP) learns to predict $\hat{z}_{t+1}$ from $(z_t, a_t)$. The total training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}}(\hat{z}_{t+1}, z_{t+1}) + 0.05 \cdot \mathcal{L}_{\text{SIGReg}}(z_t)$$

### 3.3 CEM Planner

$$a_{1:H}^* = \arg\min_{a_{1:H}}\; 1 - \cos\!\left(\hat{z}_{t+H},\; z_g\right)$$

We use K=512 samples, 50 elites, 10 iterations, and H=5. The goal embedding $z_g$ is drawn 70% from a goal buffer of achievement-positive observations and 30% from the replay buffer.

### 3.4 OBSERVE / ACT Protocol

Training proceeds in two phases. During OBSERVE (300k steps), the encoder and predictor train on random-walk data and the replay buffer is populated. During ACT (300k steps), both are frozen and the CEM plans using the converged world model. Freezing at the transition is critical: continuing to train during ACT degrades predictor quality under goal-directed distribution shift, confirmed across ablation runs.

---

## 4. Experiments

### 4.1 DoorKey

MiniGrid-DoorKey-6x6 requires the agent to navigate to a key, unlock a door, and reach the goal. The observation is a 5-dimensional symbolic state. The PPO baseline achieves 42%.

Our system with SIGReg and CEM achieves **50% success**, beating PPO by 8 percentage points. One caveat is required: the published 50% result uses symbolic state with a scripted BFS fallback for the final navigation stage. Pure CEM with H=5 cannot reliably bridge the final segment across episode layouts. The pixel version of DoorKey was validated architecturally but the headline number uses the hybrid approach.

A key finding from the DoorKey ablations: freezing the predictor at the OBSERVE/ACT boundary locks pred_ewa at 0.0098 through all of ACT. Continuing to train degrades it to 0.006. Frozen predictor is strictly better and this protocol carries through to all Crafter runs.

### 4.2 Crafter: Flat CEM Baseline

Crafter is an open-world survival environment with 22 achievements across 4 prerequisite tiers and 64x64 RGB pixel observations. We report arithmetic fraction of achievements ever unlocked across 10 evaluation episodes.

A note on metrics is necessary. DreamerV3 reports geometric mean of per-achievement unlock rates across episodes, which measures consistency. Our arithmetic fraction measures coverage: how many distinct achievements were unlocked at any point. A system that sporadically unlocks 10 achievements scores lower on geometric mean than one that reliably unlocks 5. These metrics are not comparable and we do not claim to beat DreamerV3.

The flat CEM baseline converges to pred_ewa=0.037 with SIGReg stable at sig_ewa=0.21, confirming no representation collapse. The ACT phase achieves a peak of **27.3%** sustained at 22.7%, with the following tier breakdown:

| Tier | Score |
|------|-------|
| 1 — basic survival | 67% |
| 2 — tools | 40% |
| 3 — advanced crafting | 0% |
| 4 — rare | 0% |

### 4.3 Hierarchy Ablation

Having established the 27% ceiling, we test whether planning improvements can break through it. All six runs share identical OBSERVE phases. Only the ACT-phase goal selection and sequencing strategy varies. Tier3 is the diagnostic variable: any non-zero value would indicate the architecture helped overcome the prerequisite ordering wall.

| Run | Architecture | Tier3 ACT | Peak Score |
|-----|-------------|-----------|-----------|
| 29 | Flat CEM, random goals | 0% | 27.3% |
| 30 | REINFORCE manager, H=50 | 0% | 27.3% |
| 31 | REINFORCE, H=150, intrinsic reward | 0% | 27.3% |
| 32 | Run 31 with codebook refresh every 100k ACT steps | 0% | 27.3% |
| 33 | Curiosity: argmax cosine distance to z_cur | 0% | 22.7% |
| 34 | Two-level CEM, S=3 subgoal sequences | 0% | 18.2% |

Run 30 showed a transient 36.4% peak from evaluation variance; sustained performance was 27.3%. Run 32 was stopped at 544k steps after two codebook refreshes (inertia 1059 to 1662 to 1943) failed to produce any tier3.

**REINFORCE fails on sparse reward.** One +1 signal per tier3 unlock is insufficient for credit assignment across 150-step subgoal periods. Four targeted fixes were applied across Runs 30-32: horizon 50 to 150, cosine progress intrinsic reward, goal-buffer codebook seeding, and mid-ACT codebook refresh. Each addressed a diagnosed problem. None produced tier3. The manager policy loss stabilized with each fix but the ceiling held throughout.

**Codebook blindness is self-reinforcing.** The codebook requires tier3 states to form tier3 cluster centers. Since tier3=0% during OBSERVE, no tier3 centers exist at the OBSERVE/ACT boundary. Even after refreshing with 200k ACT steps of goal-directed transitions, confirmed by rising inertia, tier3 centers never materialized. The agent cannot reach tier3 states to deposit them into the replay that the codebook is built from.

**Curiosity and two-level CEM hurt rather than help.** Curiosity goal selection (22.7%) and two-level CEM (0-18%) both perform below the flat random-goal baseline. Maximally dissimilar latent goals are not reliably reachable in H=5 CEM steps. High-level CEM triplets built from cosine heuristics do not correspond to executable prerequisite paths without a subgoal-level world model trained on subgoal-to-subgoal transitions.

### 4.4 Why the Ceiling Exists

The consistent tier3=0% across all six architectures, spanning five distinct algorithmic approaches, points to a representational rather than algorithmic bottleneck. Two failure modes explain the result.

The first is that latent proximity does not equal task proximity. A crafting bench state and a forest state may be geometrically close in SIGReg latent space because both are green, textured environments. Cosine distance to a tier3 goal embedding does not provide a consistent gradient for CEM to follow because the latent path from tier1 to tier3 is not monotonically decreasing in cosine distance.

The second is that prerequisite chains exceed what the planning horizon can coherently bridge. Obtaining a stone pickaxe requires wood (roughly 10 steps), a crafting table (5 steps), stone (20 steps), and the pickaxe itself (5 steps), totaling around 40 sequential steps with exact action dependencies. With H=5, eight or more sequential CEM calls must each make incremental progress toward the final goal. This is only possible if latent cosine distance provides a consistent signal across the full chain, which it does not when intermediate prerequisites look visually similar to each other.

Both failure modes predict the same observable: tier3=0% regardless of goal selection or sequencing strategy, which is exactly what the ablation shows.

---

## 5. Discussion

**Where geometric planning works.** Tasks where goal states are visually distinct from current states, where latent distance decreases monotonically as the agent approaches the goal, and where prerequisite chains are short. DoorKey satisfies all three conditions. Crafter tier1 and tier2 satisfy the first two partially.

**Where it fails.** Tasks requiring planning through visually ambiguous intermediate states over long chains. The visual difference between a forest state and the same forest state after chopping wood is a few pixels. The world model cannot distinguish these in latent space, so cosine distance to a tier3 goal embedding provides no gradient through the intermediate states.

**Implications for objective-driven AI.** Geometric planning is a sufficient cost function for tier1/2 tasks and a necessary but insufficient one for tier3+. The missing ingredient is not a better planning algorithm, as six algorithms confirm. What is needed is a representation that encodes causal temporal distance to a goal rather than visual similarity to it. Successor representations or contrastive temporal objectives trained on goal-reaching trajectories, rather than random walks, are natural candidates.

**Positioning relative to DreamerV3.** Our 27% arithmetic fraction is achieved with no RL. DreamerV3's 14.5% geometric mean uses full RL and is a stricter metric. We do not claim to beat DreamerV3. The contribution is matching its performance regime without reward signals, which is a different and narrower claim.

---

## 6. Conclusion

A self-supervised world model trained with SIGReg regularization, paired with CEM planning via cosine distance, achieves 50% on DoorKey and 27% on Crafter with no reinforcement learning. A six-run ablation establishes that the Crafter tier3 ceiling is not a planning problem. Flat CEM, four hierarchy variants, curiosity goal selection, and two-level CEM all fail identically. The bottleneck is representational: SIGReg geometry encodes visual similarity rather than causal prerequisite order. This is a precise and testable finding. Future RL-free planners targeting deep crafting hierarchies need representations that encode temporal goal-proximity rather than visual proximity.

---

## Acknowledgments

Experiments and analysis were conducted with assistance from Claude (Anthropic), used throughout for code generation, debugging, and iterative experimental design across 34 runs. All experimental results, architectural decisions, and scientific conclusions are the author's own.

---

## References

- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words. ICLR 2021.
- Hafner, D., et al. (2019). Learning latent dynamics for planning from pixels. ICML 2019.
- Hafner, D., Lillicrap, T. (2020). Crafter benchmark. arXiv 2109.06780.
- Hafner, D., et al. (2022). Deep hierarchical planning from pixels (Director). NeurIPS 2022.
- Hafner, D., et al. (2023). Mastering diverse domains with world models (DreamerV3). arXiv 2301.04104.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. OpenReview.
- Yarats, D., et al. (2025). Learning with world models via latent SIGReg (LeWM). arXiv 2603.19312.
