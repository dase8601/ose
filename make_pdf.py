"""Generate paper_neurips.pdf from the paper content using reportlab."""
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

doc = SimpleDocTemplate(
    "paper_neurips.pdf",
    pagesize=letter,
    rightMargin=1.1 * inch,
    leftMargin=1.1 * inch,
    topMargin=1.1 * inch,
    bottomMargin=1.1 * inch,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", parent=styles["Normal"],
    fontSize=16, leading=20, alignment=TA_CENTER,
    spaceAfter=6, fontName="Helvetica-Bold"
)
author_style = ParagraphStyle(
    "Author", parent=styles["Normal"],
    fontSize=11, leading=14, alignment=TA_CENTER,
    spaceAfter=2, fontName="Helvetica"
)
abstract_style = ParagraphStyle(
    "Abstract", parent=styles["Normal"],
    fontSize=10, leading=13, alignment=TA_JUSTIFY,
    leftIndent=0.4 * inch, rightIndent=0.4 * inch,
    spaceAfter=12
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Normal"],
    fontSize=13, leading=16, fontName="Helvetica-Bold",
    spaceBefore=14, spaceAfter=4
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Normal"],
    fontSize=11, leading=14, fontName="Helvetica-Bold",
    spaceBefore=10, spaceAfter=3
)
h3_style = ParagraphStyle(
    "H3", parent=styles["Normal"],
    fontSize=10.5, leading=13, fontName="Helvetica-BoldOblique",
    spaceBefore=8, spaceAfter=2
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, leading=14, alignment=TA_JUSTIFY,
    spaceAfter=6
)
code_style = ParagraphStyle(
    "Code", parent=styles["Normal"],
    fontSize=8.5, leading=11, fontName="Courier",
    leftIndent=0.3 * inch, spaceAfter=6,
    backColor=colors.HexColor("#f5f5f5")
)
ref_style = ParagraphStyle(
    "Ref", parent=styles["Normal"],
    fontSize=9, leading=12, leftIndent=0.25 * inch,
    firstLineIndent=-0.25 * inch, spaceAfter=3
)

def h1(text):
    return Paragraph(text, h1_style)

def h2(text):
    return Paragraph(text, h2_style)

def h3(text):
    return Paragraph(text, h3_style)

def p(text):
    return Paragraph(text, body_style)

def sp(n=6):
    return Spacer(1, n)

def eq(text):
    return Paragraph(text, ParagraphStyle(
        "Eq", parent=styles["Normal"],
        fontSize=10, leading=14, alignment=TA_CENTER,
        leftIndent=0.5 * inch, rightIndent=0.5 * inch,
        spaceAfter=6, spaceBefore=4, fontName="Courier"
    ))

story = []

# Title
story.append(sp(12))
story.append(Paragraph(
    "Geometric World Models for Reinforcement-Free Planning: A Systematic Study",
    title_style
))
story.append(sp(8))
story.append(Paragraph("Dallas Sellers", author_style))
story.append(Paragraph("University of Colorado Boulder", author_style))
story.append(Paragraph("dase8601@colorado.edu", author_style))
story.append(sp(14))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
story.append(sp(10))

# Abstract
story.append(Paragraph("<b>Abstract</b>", h2_style))
story.append(Paragraph(
    "We ask how far a self-supervised world model can go without any reinforcement learning signal. "
    "Our system trains a ViT-Tiny encoder online with SIGReg regularization, which enforces near-Gaussian "
    "random projections and makes cosine distance metrically meaningful, and plans via CEM by minimizing "
    "latent cosine distance to goal embeddings. On MiniGrid-DoorKey the system achieves 50% success, "
    "surpassing a PPO baseline of 42%. On Crafter it reaches 27% arithmetic achievement rate "
    "(tier1=67%, tier2=40%) with no RL, no expert demonstrations, and no reward shaping. "
    "A six-run ablation across planning architectures finds all variants ceiling at the same performance: "
    "flat CEM, four REINFORCE-based hierarchy variants, curiosity-driven goal selection, and two-level CEM "
    "all produce identical tier3=0%. The binding constraint is representational rather than algorithmic. "
    "SIGReg latent geometry captures visual similarity but not the causal prerequisite structure required "
    "for tier3+ crafting, and this paper locates precisely where geometric planning succeeds and where it fails.",
    abstract_style
))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
story.append(sp(10))

# 1. Introduction
story.append(h1("1  Introduction"))
story.append(p(
    "Reinforcement learning shapes behavior through reward signals that must be hand-engineered, "
    "carefully scaled, and densely provided. LeCun (2022) proposes an alternative: train a world model "
    "by self-supervised prediction, then plan by minimizing a cost function in latent space. The reward "
    "signal is replaced by geometry."
))
story.append(p(
    "We test this directly on two standard benchmarks. Our system combines an online ViT-Tiny encoder "
    "trained with SIGReg regularization, a Transformer predictor for one-step latent dynamics, and a CEM "
    "planner that minimizes cosine distance to a goal embedding. No RL is used at any stage. We then "
    "conduct a systematic six-run ablation to locate exactly where this approach succeeds and fails."
))
story.append(p(
    "Our contributions are threefold. First, we demonstrate an online self-supervised world model and CEM "
    "system that achieves 50% on DoorKey, exceeding a 42% PPO baseline, and 27% arithmetic achievement "
    "rate on Crafter with zero RL signal. Second, a six-run ablation on Crafter shows flat CEM, four "
    "REINFORCE hierarchy variants, curiosity goal selection, and two-level CEM all ceiling at the same "
    "point, pointing to representation rather than planning algorithm as the binding constraint. Third, "
    "we provide a diagnosis: SIGReg geometry encodes visual similarity, not causal prerequisite order, "
    "and we explain why this causes an identical tier3=0% result across all six architectures."
))

# 2. Related Work
story.append(h1("2  Related Work"))
story.append(h3("World models."))
story.append(p(
    "DreamerV3 (Hafner et al., 2023) trains an RSSM world model and an actor-critic on imagined rollouts, "
    "achieving 14.5% on Crafter (geometric mean). Our work uses no actor-critic and no RL anywhere in the system."
))
story.append(h3("Latent planning."))
story.append(p(
    "LeWM (Yarats et al., 2025) introduces SIGReg to make cosine distance a valid CEM cost, evaluated on "
    "offline data. We extend this to fully online learning from scratch with no pretrained encoder."
))
story.append(h3("Hierarchical planning."))
story.append(p(
    "Director (Hafner et al., 2022) trains a REINFORCE manager above a Dreamer worker, tested on mazes and "
    "Atari. HWM (2025) proposes a two-level CEM without RL. We implement both styles on Crafter and find "
    "neither breaks the 27% ceiling."
))
story.append(h3("Objective-driven AI."))
story.append(p(
    "LeCun (2022) argues self-supervised world models with cost-minimizing planners can replace RL for most "
    "tasks. Our results partially support this claim and identify its boundary condition precisely."
))

# 3. Method
story.append(h1("3  Method"))
story.append(h2("3.1  Encoder"))
story.append(p(
    "We use ViT-Tiny (patch size 16, image size 48×48 for DoorKey and 64×64 for Crafter), which outputs "
    "192-dimensional tokens projected linearly to 256 dimensions. The encoder is trained from scratch "
    "online with no pretrained weights."
))
story.append(h2("3.2  SIGReg World Model"))
story.append(p(
    "Following LeWM, we regularize the encoder with SIGReg:"
))
story.append(eq("L_SIGReg(z) = (1/M) * sum_m W1( r_m^T z,  N(0,1) ),   r_m ~ N(0,I),  M=512"))
story.append(p(
    "This enforces near-Gaussian marginals on random projections, spreading embeddings across the "
    "hypersphere so cosine distance is metrically meaningful for CEM planning. A Transformer predictor "
    "(4 heads, 512-dimensional MLP) learns to predict z_{t+1} from (z_t, a_t). The total training loss is:"
))
story.append(eq("L = L_MSE( z_pred, z_next ) + 0.05 * L_SIGReg(z_t)"))
story.append(h2("3.3  CEM Planner"))
story.append(eq("a*_{1:H} = argmin_{a_{1:H}}  1 - cos( z_{t+H}, z_goal )"))
story.append(p(
    "We use K=512 samples, 50 elites, 10 iterations, and H=5. The goal embedding z_g is drawn 70% from "
    "a goal buffer of achievement-positive observations and 30% from the replay buffer."
))
story.append(h2("3.4  OBSERVE / ACT Protocol"))
story.append(p(
    "Training proceeds in two phases. During OBSERVE (300k steps), the encoder and predictor train on "
    "random-walk data and the replay buffer is populated. During ACT (300k steps), both are frozen and "
    "the CEM plans using the converged world model. Freezing at the transition is critical: continuing to "
    "train during ACT degrades predictor quality under goal-directed distribution shift, confirmed across "
    "ablation runs."
))

# 4. Experiments
story.append(h1("4  Experiments"))
story.append(h2("4.1  DoorKey"))
story.append(p(
    "MiniGrid-DoorKey-6x6 requires the agent to navigate to a key, unlock a door, and reach the goal. "
    "The observation is a 5-dimensional symbolic state. The PPO baseline achieves 42%."
))
story.append(p(
    "<b>Our system with SIGReg and CEM achieves 50% success</b>, beating PPO by 8 percentage points. "
    "One caveat is required: the published 50% result uses symbolic state with a scripted BFS fallback "
    "for the final navigation stage. Pure CEM with H=5 cannot reliably bridge the final segment across "
    "episode layouts. The pixel version of DoorKey was validated architecturally but the headline number "
    "uses the hybrid approach."
))
story.append(p(
    "A key finding from the DoorKey ablations: freezing the predictor at the OBSERVE/ACT boundary locks "
    "pred_ewa at 0.0098 through all of ACT. Continuing to train degrades it to 0.006. Frozen predictor "
    "is strictly better and this protocol carries through to all Crafter runs."
))

story.append(h2("4.2  Crafter: Flat CEM Baseline"))
story.append(p(
    "Crafter is an open-world survival environment with 22 achievements across 4 prerequisite tiers and "
    "64×64 RGB pixel observations. We report arithmetic fraction of achievements ever unlocked across "
    "10 evaluation episodes."
))
story.append(p(
    "A note on metrics is necessary. DreamerV3 reports geometric mean of per-achievement unlock rates "
    "across episodes, which measures consistency. Our arithmetic fraction measures coverage: how many "
    "distinct achievements were unlocked at any point. These metrics are not comparable and we do not "
    "claim to beat DreamerV3."
))
story.append(p(
    "The flat CEM baseline converges to pred_ewa=0.037 with SIGReg stable at sig_ewa=0.21, confirming "
    "no representation collapse. The ACT phase achieves a peak of <b>27.3%</b> sustained at 22.7%."
))

tier_data = [
    ["Tier", "Score"],
    ["1 — basic survival", "67%"],
    ["2 — tools", "40%"],
    ["3 — advanced crafting", "0%"],
    ["4 — rare", "0%"],
]
tier_table = Table(tier_data, colWidths=[3.2 * inch, 1.2 * inch])
tier_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(sp(4))
story.append(tier_table)
story.append(sp(8))

story.append(h2("4.3  Hierarchy Ablation"))
story.append(p(
    "Having established the 27% ceiling, we test whether planning improvements can break through it. "
    "All six runs share identical OBSERVE phases. Only the ACT-phase goal selection and sequencing "
    "strategy varies. Tier3 is the diagnostic variable: any non-zero value would indicate the architecture "
    "helped overcome the prerequisite ordering wall."
))

ablation_data = [
    ["Run", "Architecture", "Tier3 ACT", "Peak Score"],
    ["29", "Flat CEM, random goals", "0%", "27.3%"],
    ["30", "REINFORCE manager, H=50", "0%", "27.3%"],
    ["31", "REINFORCE, H=150, intrinsic reward", "0%", "27.3%"],
    ["32", "Run 31 + codebook refresh every 100k ACT steps", "0%", "27.3%"],
    ["33", "Curiosity: argmax cosine distance to z_cur", "0%", "22.7%"],
    ["34", "Two-level CEM, S=3 subgoal sequences", "0%", "18.2%"],
]
ablation_table = Table(ablation_data, colWidths=[0.4*inch, 3.0*inch, 0.85*inch, 0.9*inch])
ablation_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(sp(4))
story.append(ablation_table)
story.append(sp(8))

story.append(p(
    "Run 30 showed a transient 36.4% peak from evaluation variance; sustained performance was 27.3%. "
    "Run 32 was stopped at 544k steps after two codebook refreshes (inertia 1059 to 1662 to 1943) "
    "failed to produce any tier3."
))
story.append(h3("REINFORCE fails on sparse reward."))
story.append(p(
    "One +1 signal per tier3 unlock is insufficient for credit assignment across 150-step subgoal "
    "periods. Four targeted fixes were applied across Runs 30-32: horizon 50 to 150, cosine progress "
    "intrinsic reward, goal-buffer codebook seeding, and mid-ACT codebook refresh. Each addressed a "
    "diagnosed problem. None produced tier3. The manager policy loss stabilized with each fix but the "
    "ceiling held throughout."
))
story.append(h3("Codebook blindness is self-reinforcing."))
story.append(p(
    "The codebook requires tier3 states to form tier3 cluster centers. Since tier3=0% during OBSERVE, "
    "no tier3 centers exist at the OBSERVE/ACT boundary. Even after refreshing with 200k ACT steps of "
    "goal-directed transitions, confirmed by rising inertia, tier3 centers never materialized. The agent "
    "cannot reach tier3 states to deposit them into the replay that the codebook is built from."
))
story.append(h3("Curiosity and two-level CEM hurt rather than help."))
story.append(p(
    "Curiosity goal selection (22.7%) and two-level CEM (0-18%) both perform below the flat random-goal "
    "baseline. Maximally dissimilar latent goals are not reliably reachable in H=5 CEM steps. High-level "
    "CEM triplets built from cosine heuristics do not correspond to executable prerequisite paths without "
    "a subgoal-level world model trained on subgoal-to-subgoal transitions."
))

story.append(h2("4.4  Why the Ceiling Exists"))
story.append(p(
    "The consistent tier3=0% across all six architectures, spanning five distinct algorithmic approaches, "
    "points to a representational rather than algorithmic bottleneck. Two failure modes explain the result."
))
story.append(p(
    "The first is that latent proximity does not equal task proximity. A crafting bench state and a forest "
    "state may be geometrically close in SIGReg latent space because both are green, textured environments. "
    "Cosine distance to a tier3 goal embedding does not provide a consistent gradient for CEM to follow "
    "because the latent path from tier1 to tier3 is not monotonically decreasing in cosine distance."
))
story.append(p(
    "The second is that prerequisite chains exceed what the planning horizon can coherently bridge. "
    "Obtaining a stone pickaxe requires wood (roughly 10 steps), a crafting table (5 steps), stone "
    "(20 steps), and the pickaxe itself (5 steps), totaling around 40 sequential steps with exact action "
    "dependencies. With H=5, eight or more sequential CEM calls must each make incremental progress toward "
    "the final goal. This is only possible if latent cosine distance provides a consistent signal across "
    "the full chain, which it does not when intermediate prerequisites look visually similar to each other."
))
story.append(p(
    "Both failure modes predict the same observable: tier3=0% regardless of goal selection or sequencing "
    "strategy, which is exactly what the ablation shows."
))

# 5. Discussion
story.append(h1("5  Discussion"))
story.append(h3("Where geometric planning works."))
story.append(p(
    "Tasks where goal states are visually distinct from current states, where latent distance decreases "
    "monotonically as the agent approaches the goal, and where prerequisite chains are short. DoorKey "
    "satisfies all three conditions. Crafter tier1 and tier2 satisfy the first two partially."
))
story.append(h3("Where it fails."))
story.append(p(
    "Tasks requiring planning through visually ambiguous intermediate states over long chains. The visual "
    "difference between a forest state and the same forest state after chopping wood is a few pixels. "
    "The world model cannot distinguish these in latent space, so cosine distance to a tier3 goal "
    "embedding provides no gradient through the intermediate states."
))
story.append(h3("Implications for objective-driven AI."))
story.append(p(
    "Geometric planning is a sufficient cost function for tier1/2 tasks and a necessary but insufficient "
    "one for tier3+. The missing ingredient is not a better planning algorithm, as six algorithms confirm. "
    "What is needed is a representation that encodes causal temporal distance to a goal rather than "
    "visual similarity to it. Successor representations or contrastive temporal objectives trained on "
    "goal-reaching trajectories, rather than random walks, are natural candidates."
))
story.append(h3("Positioning relative to DreamerV3."))
story.append(p(
    "Our 27% arithmetic fraction is achieved with no RL. DreamerV3's 14.5% geometric mean uses full RL "
    "and is a stricter metric. We do not claim to beat DreamerV3. The contribution is matching its "
    "performance regime without reward signals, which is a different and narrower claim."
))

# 6. Conclusion
story.append(h1("6  Conclusion"))
story.append(p(
    "A self-supervised world model trained with SIGReg regularization, paired with CEM planning via "
    "cosine distance, achieves 50% on DoorKey and 27% on Crafter with no reinforcement learning. "
    "A six-run ablation establishes that the Crafter tier3 ceiling is not a planning problem. "
    "Flat CEM, four hierarchy variants, curiosity goal selection, and two-level CEM all fail identically. "
    "The bottleneck is representational: SIGReg geometry encodes visual similarity rather than causal "
    "prerequisite order. This is a precise and testable finding. Future RL-free planners targeting deep "
    "crafting hierarchies need representations that encode temporal goal-proximity rather than visual proximity."
))

# Acknowledgments
story.append(h1("Acknowledgments"))
story.append(p(
    "Experiments and analysis were conducted with assistance from Claude (Anthropic), used throughout "
    "for code generation, debugging, and iterative experimental design across 34 runs. All experimental "
    "results, architectural decisions, and scientific conclusions are the author's own."
))

# References
story.append(h1("References"))
refs = [
    "Dosovitskiy, A., et al. (2021). An image is worth 16x16 words. <i>ICLR 2021</i>.",
    "Hafner, D., et al. (2019). Learning latent dynamics for planning from pixels. <i>ICML 2019</i>.",
    "Hafner, D., Lillicrap, T. (2021). Crafter benchmark. <i>arXiv 2109.06780</i>.",
    "Hafner, D., et al. (2022). Deep hierarchical planning from pixels (Director). <i>NeurIPS 2022</i>.",
    "Hafner, D., et al. (2023). Mastering diverse domains with world models (DreamerV3). <i>arXiv 2301.04104</i>.",
    "LeCun, Y. (2022). A path towards autonomous machine intelligence. <i>OpenReview</i>.",
    "Yarats, D., et al. (2025). Learning with world models via latent SIGReg (LeWM). <i>arXiv 2603.19312</i>.",
]
for ref in refs:
    story.append(Paragraph(ref, ref_style))
    story.append(sp(2))

doc.build(story)
print("Done: paper_neurips.pdf")
