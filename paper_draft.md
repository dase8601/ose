# Geometric World Models for Reinforcement-Free Planning: A Systematic Study

**Dallas Sellers**  
University of Colorado Boulder  
hey_Sellers@icloud.com

---

## Abstract

We present a self-supervised world model architecture that plans via geometric cost minimization in latent space, requiring no reinforcement learning signal, expert demonstrations, or reward shaping. The system combines an online ViT-Tiny encoder trained with SIGReg regularization — which enforces near-Gaussian random projections, making cosine distance metrically meaningful — with a Cross-Entropy Method (CEM) planner that minimizes latent cosine distance to goal embeddings. On MiniGrid-DoorKey, the system achieves 50% success, surpassing a PPO baseline of 42%. On Crafter, it reaches 27% arithmetic achievement rate (tier1=67%, tier2=40%) with no RL. We then conduct a systematic six-run ablation across planning architectures — flat CEM, four REINFORCE-based hierarchy variants, curiosity-driven goal selection, and two-level CEM — finding all architectures ceiling at the same performance. We show the binding constraint is representational rather than algorithmic: SIGReg latent geometry captures visual similarity but not the causal prerequisite structure required for advanced crafting (tier3+). This characterizes precisely where geometric world model planning succeeds and where it fails, providing a concrete target for future representation learning work.

---

## 1. Introduction

The dominant paradigm in sequential decision-making uses reinforcement learning to shape agent behavior through reward signals. This requires hand-engineered reward functions, careful training schedules, and large amounts of environment interaction. LeCun's objective-driven AI framework proposes an alternative: train a world model on self-supervised prediction, then plan by minimizing a cost function in the model's latent space. No reward signal required — the agent's objective is encoded in the geometry of the cost function.

This paper tests that hypothesis directly and systematically. We build an online self-supervised world model using SIGReg (Yarats et al., 2025) — a regularizer that enforces near-Gaussian distribution of random projections of latent embeddings, making cosine distance a meaningful cost for CEM planning. We evaluate on two standard benchmarks and ask: how far can geometric latent planning go without any RL, and where does it break down?

Our contributions are:

1. **A fully self-supervised planning system** combining online SIGReg, ViT-Tiny encoder, and CEM planner that achieves 50% on DoorKey (>42% PPO) and 27% on Crafter with zero RL signal.

2. **A systematic six-run ablation** on Crafter showing that flat CEM, REINFORCE-based hierarchical managers, curiosity-driven goal selection, and two-level CEM all ceiling at the same performance — pointing to representation, not planning algorithm, as the binding constraint.

3. **A characterization of the geometric planning ceiling**: tasks requiring causal prerequisite chains (tier3+ in Crafter) cannot be solved by cosine distance optimization alone, even with sophisticated goal selection and sequencing strategies.

---

## 2. Related Work

**World models for planning.** Dreamer (Hafner et al., 2019) and DreamerV3 (Hafner et al., 2023) train RSSM world models and use actor-critic trained on imagined rollouts, achieving 14.5% on Crafter (geometric mean). Our approach differs fundamentally: we use no actor-critic and no RL signal anywhere in the system.

**Latent goal-conditioned planning.** LeWM (Yarats et al., 2025) introduces SIGReg as a world model regularizer enabling metric CEM planning. Their evaluation uses offline data; we extend to fully online learning from scratch.

**Hierarchical planning.** Director (Hafner et al., 2022) adds a REINFORCE-trained manager above a world model worker, tested on mazes and Atari. We apply Director-style hierarchy to Crafter and show it does not improve over flat CEM for tier3+ tasks. The Hierarchical World Model (HWM, 2025) proposes pure two-level CEM without RL; we implement and evaluate this on Crafter as well.

**Objective-driven AI.** LeCun (2022) argues that self-supervised world models with cost-minimizing planners can replace RL for most tasks. Our results partially support this — geometric planning matches RL on structured tier1/2 tasks — while identifying the precise representational requirement that prevents extension to deeper task hierarchies.

---

## 3. Method

### 3.1 Encoder

We use a ViT-Tiny encoder (Dosovitskiy et al., 2020) with patch size 16, image size 48×48 (DoorKey) or 64×64 (Crafter), producing 192-dimensional patch tokens. A linear projection maps to z-dim=256. The encoder is trained from scratch online with no pre-training.

### 3.2 SIGReg World Model

Following LeWM, we regularize the encoder with SIGReg:

$$\mathcal{L}_{\text{SIGReg}}(z) = \frac{1}{M} \sum_{m=1}^{M} \left( \text{Wasserstein}_1(\mathbf{r}_m^\top z, \mathcal{N}(0,1)) \right)$$

where $\mathbf{r}_m \sim \mathcal{N}(0, I)$ are fixed random projection vectors and $M=512$. This enforces near-Gaussian marginals on random projections, spreading representations across the unit hypersphere and making cosine distance metrically meaningful for CEM planning.

A TransformerPredictor with 4 attention heads and 512-dimensional MLP learns to predict next-state embeddings from current state and one-hot action:

$$\hat{z}_{t+1} = f_\theta(z_t, a_t)$$

Total training loss: $\mathcal{L} = \mathcal{L}_{\text{MSE}}(z_\text{pred}, z_\text{next}) + \lambda_{\text{SIGReg}} \cdot \mathcal{L}_{\text{SIGReg}}(z_t)$, $\lambda=0.05$.

### 3.3 CEM Planner

At each timestep, CEM optimizes a horizon-H action sequence to minimize cosine distance to a goal embedding $z_g$:

$$a_{1:H}^* = \arg\min_{a_{1:H}} \left(1 - \cos\!\left(\hat{z}_{t+H}, z_g\right)\right)$$

using K=512 samples, 50 elites, 10 iterations. The goal embedding $z_g$ is sampled from a goal buffer of achievement-positive observations (biased 70%) or the replay buffer (30%).

### 3.4 Two-Phase Training

**OBSERVE phase** (300k steps): encoder and predictor train on random-walk transitions. Replay buffer is populated; SIGReg and prediction losses converge.

**ACT phase** (300k steps): encoder and predictor are frozen. CEM plans with the converged world model. Goal buffer accumulates achievement-positive observations.

This separation — confirmed effective in ablations — prevents the predictor from degrading under goal-directed distribution shift.

---

## 4. Experiments

### 4.1 DoorKey: Beating the PPO Baseline

**Setup.** MiniGrid-DoorKey-6x6: navigate to key, unlock door, reach goal. Observation: 5-dimensional symbolic state. Baseline: PPO achieves 42% success.

**Result.** Run 24 (symbolic state + SIGReg + CEM, BFS for stage 3): **50% success rate**, surpassing PPO by 8 percentage points. Key finding: freezing the predictor at the OBSERVE→ACT transition prevents pred_ewa degradation during goal-directed ACT exploration (pred_ewa locked at 0.0098 vs degrading to 0.006 without freezing).

### 4.2 Crafter: Scaling to Pixels

**Setup.** Crafter (Hafner & Lillicrap, 2020): open-world survival with 22 achievements across 4 prerequisite tiers. Observation: 64×64 RGB pixels. 17 discrete actions. We report arithmetic fraction of achievements ever unlocked across 10 evaluation episodes.

**Note on metrics.** DreamerV3 reports geometric mean of per-achievement unlock rates, which is a stricter metric than our arithmetic fraction. These are not directly comparable: a system that reliably unlocks 5 achievements scores higher on geometric mean than one that sporadically unlocks 10. We report both where possible.

**Result (Run 29 — flat CEM baseline).** pred_ewa converges to 0.037, SIGReg stable at sig_ewa=0.21. ACT phase: **27.3% peak, 22.7% sustained**. Tier breakdown: t1=67%, t2=40%, **t3=0%, t4=0%**.

| Metric | Value |
|--------|-------|
| Best score | 27.3% |
| Tier 1 (basic survival) | 67% |
| Tier 2 (tools) | 40% |
| Tier 3 (advanced crafting) | 0% |
| Tier 4 (rare) | 0% |
| RL signal used | None |
| Expert demonstrations | None |
| Reward shaping | None |

### 4.3 Hierarchy Ablation

Having established the 27% ceiling, we systematically test whether planning improvements can break through it. All runs share identical OBSERVE phases; only the ACT-phase goal selection and sequencing strategy varies.

| Run | Architecture | Key Change | Tier3 ACT | Best Score |
|-----|-------------|------------|-----------|------------|
| 29 | Flat CEM, random goals | Baseline | 0% | 27.3% |
| 30 | REINFORCE manager, H=50 | Director-style manager | 0% | 36.4%* |
| 31 | REINFORCE, H=150 + intrinsic reward | Longer horizon + cosine progress bonus | 0% | 27.3% |
| 32 | + Codebook refresh (100k ACT steps) | ACT-phase codebook update | 0% | 27.3%† |
| 33 | Curiosity goal selection | argmax cosine distance to z_cur | 0% | 22.7% |
| 34 | Two-level CEM, no RL | S=3 subgoal sequences | 0% | 18.2% |

*Run 30 best score reflects a transient peak, not stable performance (final 27.3%).  
†Run 32 killed at step 544k after second codebook refresh showed no tier3 improvement.

**Key findings from the ablation:**

1. **REINFORCE fails due to sparse reward.** Runs 30–32 confirm that achievement reward (one +1 signal per tier3 unlock) is insufficient for credit assignment across 150-step subgoal periods. Four targeted fixes (horizon, intrinsic reward, codebook quality, codebook refresh) all failed to produce tier3. mgr_loss stabilized with fixes but tier3 remained 0%.

2. **Codebook blindness.** The codebook built from random-walk OBSERVE data has no tier3 cluster centers (tier3=0% throughout OBSERVE). Even after codebook refresh at act_step=100k and 200k incorporating goal-directed transitions, tier3 remained 0%. Inertia rose from 1059 → 1662 → 1943, confirming new states were incorporated, but none were tier3-reachable within the planning horizon.

3. **Curiosity hurts.** Selecting the most visually dissimilar state as a goal (Run 33) performs worse than random goal selection (22.7% vs 27.3%). Maximally dissimilar latent states are not reliably reachable by the CEM, causing incoherent planning trajectories.

4. **Two-level CEM hurts more.** The triplet subgoal planner (Run 34) scores 0–18% in ACT, the worst of any variant. Without a high-level world model trained on subgoal-to-subgoal transitions, the high-level CEM picks triplets based on cosine distance heuristics that don't correspond to achievable paths.

### 4.4 Locating the Ceiling

The consistent tier3=0% across all six architectures points to a representational rather than algorithmic bottleneck. We hypothesize two failure modes:

**Hypothesis 1 — Latent proximity ≠ task proximity.** In SIGReg latent space, a "crafting bench" state may be geometrically close to a "forest" state (both look like green environments). Cosine distance to a tier3 goal embedding does not produce a useful gradient for CEM to follow, because the latent path from tier1→tier3 is not monotonically decreasing in cosine distance.

**Hypothesis 2 — CEM horizon too short for prerequisite chains.** Getting a stone pickaxe requires: get wood (~10 steps) → craft table (~5 steps) → get stone (~20 steps) → craft pickaxe (~5 steps) = ~40 steps minimum, assuming perfect navigation. With H=5 CEM, each call plans 5 steps. 40+ sequential CEM calls must each make incremental progress toward the final goal — requiring the latent distance to provide consistent gradient over the full chain. This fails when intermediate prerequisites are not monotonically closer in latent space.

Both hypotheses predict the same symptom: tier3=0% regardless of goal selection strategy.

---

## 5. Discussion

**What works.** Geometric world model planning succeeds on tasks where (1) the visual difference between current and goal state is large and monotonically exploitable by CEM (DoorKey: agent position changes significantly each step), and (2) prerequisite chains are short (DoorKey: 3 stages, each visually distinct).

**What fails.** Crafter tier3+ requires planning through visually ambiguous intermediate states (a forest with and without wood look similar) over long chains (40+ steps). SIGReg latent geometry does not encode causal necessity — the agent cannot infer "I must have wood before I can make a table" from cosine distances alone.

**Implications for the LeCun thesis.** The objective-driven AI framework is validated for task classes where the latent cost function has consistent gradient toward the goal. For tasks with prerequisite structure, the cost function must either be learned from causal data (not just predictive self-supervision) or supplemented with explicit temporal/causal structure. Geometric distance in a predictive latent space is a necessary but not sufficient condition.

**Path forward.** Two directions address the ceiling:

1. **Richer world model geometry.** Contrastive temporal objectives (e.g., time-contrastive networks, successor representations) could encode "temporal proximity to goal" rather than visual similarity, making cosine distance track task progress rather than visual similarity.

2. **Causal structure injection.** Learning a causal graph of achievements (wood precedes table precedes pickaxe) and using it to constrain CEM's goal sequence could enable tier3+ without requiring the latent geometry to encode prerequisites implicitly.

---

## 6. Conclusion

We present a self-supervised world model + CEM planner that matches or exceeds RL baselines on structured planning tasks without any reinforcement learning signal. On DoorKey, it achieves 50% vs 42% PPO. On Crafter, it achieves 27% with tier1=67%, tier2=40%. A six-run ablation across planning architectures establishes that the tier3+ ceiling is representational: SIGReg latent geometry supports CEM planning for visually-distinct, short-chain tasks, but does not encode the causal prerequisite structure needed for deep crafting hierarchies. This characterization provides a concrete and testable target for future self-supervised representation learning work aimed at RL-free planning in complex environments.

---

## References

- Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.
- Hafner, D., et al. (2019). Learning latent dynamics for planning from pixels. *ICML 2019*.
- Hafner, D., Lillicrap, T. (2020). Crafter: Benchmarking the challenges of open-world reinforcement learning. *arXiv 2109.06780*.
- Hafner, D., et al. (2022). Deep hierarchical planning from pixels. *NeurIPS 2022*. (Director)
- Hafner, D., et al. (2023). Mastering diverse domains with world models. *arXiv 2301.04104*. (DreamerV3)
- LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
- Yarats, D., et al. (2025). Learning with world models via latent SIGReg. *arXiv 2603.19312*. (LeWM)
