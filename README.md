# Geometric World Models for Reinforcement-Free Planning

A systematic study of self-supervised world models for planning without reinforcement learning.

Online ViT-Tiny encoder trained with SIGReg regularization, a Transformer predictor for one-step latent dynamics, and a CEM planner that minimizes cosine distance to goal embeddings. No reward signals, no expert demonstrations, no policy gradient at any stage.

**Results:** 50% on MiniGrid-DoorKey (vs. 42% PPO baseline), 27% arithmetic achievement rate on Crafter with zero RL.

---

## Demo

Agent playing Crafter at evaluation step 100k (tier1=67%, tier2=40%, tier3=17%):

<video src="media/crafter_eval_step100k.mp4" controls width="480"></video>

---

## How It Works

The core idea: replace reward-shaping with geometry. SIGReg enforces near-Gaussian marginals on random projections of the encoder output, spreading embeddings across the hypersphere so cosine distance becomes a valid planning cost. CEM then optimizes action sequences by minimizing cosine distance from the predicted future latent to a goal embedding sampled from a goal buffer.

Training runs in two phases. During **OBSERVE** (300k steps), the encoder and predictor train on random-walk experience. During **ACT** (300k steps), both are frozen and the CEM planner uses the converged world model. Freezing at the transition is critical: continuing to train during ACT degrades predictor quality under goal-directed distribution shift.

```
Observation (pixels)
    → ViT-Tiny encoder + SIGReg → z_t (256-dim)
    → Transformer predictor      → z_{t+H} (predicted future)
    → CEM planner                → argmin cosine_distance(z_{t+H}, z_goal)
    → action sequence
```

---

## Results

### MiniGrid-DoorKey

| System | Success Rate |
|--------|-------------|
| PPO baseline | 42% |
| SIGReg + CEM (ours) | **50%** |

Uses symbolic state with BFS fallback for the final navigation stage. Pure pixel CEM was architecturally validated but the headline number uses the hybrid approach.

### Crafter — Flat CEM Baseline

| Tier | Score |
|------|-------|
| 1 — basic survival | 67% |
| 2 — tools | 40% |
| 3 — advanced crafting | 0% |
| 4 — rare | 0% |
| **Overall (arithmetic)** | **27%** |

No RL, no reward shaping, no demonstrations. `pred_ewa=0.037`, `sig_ewa=0.21` (stable representation).

Note on metrics: DreamerV3 reports geometric mean of per-achievement unlock rates across episodes (14.5%). Our 27% is arithmetic fraction of achievements ever unlocked (coverage). These are not comparable metrics, and we do not claim to beat DreamerV3.

### Crafter — Six-Run Hierarchy Ablation

All six runs share identical OBSERVE phases. Only the ACT-phase goal selection and sequencing varies. Tier3 is the diagnostic variable.

| Run | Architecture | Tier3 | Peak |
|-----|-------------|-------|------|
| 29 | Flat CEM, random goals | 0% | 27.3% |
| 30 | REINFORCE manager, H=50 | 0% | 27.3% |
| 31 | REINFORCE, H=150, intrinsic reward | 0% | 27.3% |
| 32 | Run 31 + codebook refresh every 100k ACT steps | 0% | 27.3% |
| 33 | Curiosity: argmax cosine distance to current state | 0% | 22.7% |
| 34 | Two-level CEM, S=3 subgoal sequences | 0% | 18.2% |

The consistent tier3=0% across five distinct algorithmic approaches points to the representation, not the planning algorithm, as the binding constraint. SIGReg encodes visual similarity; tier3 crafting requires causal prerequisite order.

---

## Architecture

### Encoder

ViT-Tiny (`vit_tiny_patch16_224`, `pretrained=False`), image size 64x64 for Crafter / 48x48 for DoorKey. Outputs 192-dim tokens projected to 256-dim via linear layer. Trained from scratch online with no pretrained weights.

### SIGReg Regularizer

```python
L_sigreg(z) = (1/M) * sum_m W1(r_m.T @ z, N(0,1))   # M=512 random projections
```

Wasserstein-1 distance between random projections and a standard Gaussian. Enforces hyperspherical spreading so cosine distance is metrically meaningful.

### Transformer Predictor

4 attention heads, 512-dim MLP. Predicts `z_{t+1}` from `(z_t, a_t)`. Training loss:

```
L = MSE(z_pred, z_next) + 0.05 * L_sigreg(z_t)
```

### CEM Planner

```
a* = argmin_{a_{1:H}} 1 - cos(z_{t+H}, z_goal)
```

K=512 samples, 50 elites, 10 iterations, H=5. Goal embedding drawn 70% from achievement-positive observations, 30% from replay buffer.

---

## Running Experiments

### Setup

```bash
git clone https://github.com/dase8601/ose.git
cd ose
pip install torch torchvision timm minigrid crafter gymnasium numpy
```

### Run a Crafter experiment

```bash
python abm_experiment.py \
  --loop-module abm.loop_mpc_crafter_run29 \
  --condition lewm_crafter_pixels \
  --device cuda \
  --env crafter \
  --steps 600000
```

### Run DoorKey

```bash
python abm_experiment.py \
  --loop-module abm.loop_mpc_doorkey_run28 \
  --condition lewm_doorkey_pixels \
  --device cuda \
  --env doorkey \
  --steps 300000
```

### Available conditions

| Condition | Module | Description |
|-----------|--------|-------------|
| `lewm_doorkey_pixels` | `abm.loop_mpc_doorkey_run28` | DoorKey pixels, ViT-Tiny + SIGReg |
| `lewm_crafter_pixels` | `abm.loop_mpc_crafter_run29` | Crafter flat CEM baseline |
| `lewm_crafter_hierarchy` | `abm.loop_mpc_crafter_run30` | REINFORCE manager H=50 |
| `lewm_crafter_hierarchy_v2` | `abm.loop_mpc_crafter_run31` | REINFORCE H=150 + intrinsic reward |
| `lewm_crafter_hierarchy_v3` | `abm.loop_mpc_crafter_run32` | Run 31 + codebook refresh |
| `lewm_crafter_curiosity` | `abm.loop_mpc_crafter_run33` | Curiosity goal selection |
| `lewm_crafter_twolevel` | `abm.loop_mpc_crafter_run34` | Two-level CEM |

---

## Code Structure

```
abm/
├── world_model.py              # ViT-Tiny encoder, TransformerPredictor, sigreg()
├── cem_planner.py              # CEM planner (cosine and L2 distance modes)
├── loop_mpc_doorkey_run28.py   # DoorKey pixel run
├── loop_mpc_crafter_run29.py   # Crafter flat CEM baseline
├── loop_mpc_crafter_run30.py   # REINFORCE manager (H=50)
├── loop_mpc_crafter_run31.py   # REINFORCE + intrinsic reward
├── loop_mpc_crafter_run32.py   # Codebook refresh ablation
├── loop_mpc_crafter_run33.py   # Curiosity goal selection
└── loop_mpc_crafter_run34.py   # Two-level CEM

abm_experiment.py               # Experiment runner / dispatcher
EXPERIMENTS.md                  # Full run log with parameters and results
paper_draft.md                  # Paper draft
```

---

## Why the Ceiling Exists

Tier3 requires approximately 40 sequential steps with exact action dependencies (e.g., collect wood → craft table → collect stone → craft pickaxe). Two failure modes interact.

**Latent proximity is not task proximity.** A crafting bench state and a forest state may be close in SIGReg space because both are textured green environments. Cosine distance to a tier3 goal does not provide a monotonically decreasing gradient across the full prerequisite chain.

**Prerequisite chains exceed H=5 CEM horizon.** Eight or more sequential CEM calls must each make incremental progress toward the final goal. This requires a consistent cosine distance gradient through visually ambiguous intermediates, which SIGReg does not provide.

The diagnosis: what is needed is not a better planning algorithm (six algorithms confirm this), but a representation that encodes causal temporal distance rather than visual similarity. Successor representations or contrastive temporal objectives trained on goal-reaching trajectories are natural next directions.

---

## Acknowledgments

Experiments and analysis were conducted with assistance from Claude (Anthropic), used throughout for code generation, debugging, and iterative experimental design across 34 runs. All experimental results, architectural decisions, and scientific conclusions are the author's own.

---

## References

- Dosovitskiy et al. (2021). An image is worth 16x16 words. ICLR 2021.
- Hafner et al. (2022). Deep hierarchical planning from pixels (Director). NeurIPS 2022.
- Hafner et al. (2023). Mastering diverse domains with world models (DreamerV3). arXiv 2301.04104.
- LeCun (2022). A path towards autonomous machine intelligence. OpenReview.
- Yarats et al. (2025). Learning with world models via latent SIGReg (LeWM). arXiv 2603.19312.
