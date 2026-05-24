"""2-layer FFN projection vs single-linear projection — hypothesis check.

The agent is dropped into the researcher's working dir mid-writeup: the
team's ablation showed a 2-layer FFN projection beating a single-linear
projection on the 17M model (NDCG@10 0.6405 vs 0.6286). The researcher's
intuition is that the extra layer gives more expressivity. The agent's
job is to verify the result is real, test the expressivity hypothesis,
and write up the actual mechanism.

The hidden ground truth (not in the workspace, used by the grader):

  Both Dense modules in the released 17M model use identity activation
  (see `model_files/1_Dense/config.json` and `model_files/2_Dense/
  config.json`: `"activation_function": "torch.nn.modules.linear.Identity"`).
  With identity activations, the 2-layer FFN is mathematically a single
  linear map: out = W2 @ W1 @ x. The composed matrix W_eff = W2 @ W1
  has the same shape (256 -> 48) and the same maximum rank (48) as any
  single linear projection of the same shape.

  `experiments/projection_decomposition/svd_per_layer.json` confirms this:
    - W1 (512 x 256): rank 256, eff rank 223, rank-95 = 200
    - W2 (48 x 512):  rank 48,  eff rank 47.3, rank-95 = 45
    - W_eff (48 x 256): rank 48, eff rank 46.8, rank-95 = 45
  W_eff is rank-48 (full possible rank given the 48 output dim). A single
  linear layer of the same shape would also be at most rank 48. SAME
  EXPRESSIVITY.

  `experiments/projection_decomposition/equivalence_check.json` confirms
  empirically: max abs diff between the 2-layer pipeline output and the
  collapsed W_eff pipeline output is ~1e-5 (float32 numerical noise);
  per-token cosine similarity is 1.0 to 6+ decimal places.

  So the +1 NDCG gap is NOT from expressivity. The actual mechanism is
  optimization dynamics. Overparameterized linear networks (W = W2 W1
  with both trained jointly) have implicit pre-conditioning on the
  gradient updates: each weight matrix sees a gradient that depends on
  the other, which can effectively act as an adaptive per-axis learning
  rate. Saxe et al. 2013 ('Exact solutions to the nonlinear dynamics of
  learning in deep linear neural networks') derived this; Arora et al.
  2018 ('On the Optimization of Deep Networks: Implicit Acceleration by
  Overparameterization') extended it. For a deep-LINEAR network with no
  nonlinearity, the model class is identical to a shallow linear one,
  but the LOSS LANDSCAPE is different — the deep version has saddle
  points that the shallow one doesn't, and gradient flow on the deep
  one can converge faster on certain axes.

  Practical implication for the team: the 2-layer projection is worth
  shipping (the +1 NDCG is real and reproducible across seeds per the
  team's own note in `experiments/projection_layer_ablation/
  results.json`). But framing it as 'more expressive' would be wrong;
  the correct framing is 'overparameterized linear map has better
  optimization dynamics'. A 3-layer extension might help further (same
  argument); adding a nonlinearity (e.g. GELU between the two layers)
  would create a genuinely-more-expressive function, which is a separate
  design axis.
"""

from env import diagnose_research_study

PROMPT = (
    "i'm wrapping up the projection-layer ablation for the report. ran "
    "the 17m with two projection-head variants matched on everything "
    "else:\n\n"
    "  2-layer ffn  (256 -> 512 -> 48)   nanobeir avg 0.6405\n"
    "  single linear (256 -> 48)          0.6286\n\n"
    "+1.2 ndcg gap, holds across 3 seeds per cell (see "
    "`experiments/projection_layer_ablation/results.json`). my intuition "
    "going in was 'more layers = more expressivity = better' and i was "
    "going to put that in the writeup. but a teammate pointed out that "
    "both dense layers in my setup are identity-activation (see "
    "`model_files/1_Dense/config.json` and `2_Dense/config.json`), which "
    "makes me nervous about the expressivity story. if both are linear, "
    "isn't the 2-layer pipeline just equivalent to a single matmul "
    "W_eff = W2 @ W1?\n\n"
    "i don't want to ship the report with the wrong mechanism. help me "
    "figure out what's actually going on:\n"
    "  1. is the 2-layer pipeline genuinely more expressive than a single "
    "linear projection of the same input / output shape?\n"
    "  2. if not, what's actually causing the +1 ndcg gap?\n"
    "  3. what should we write up — and what should we recommend for the "
    "next iteration of the head design?\n\n"
    "workspace:\n"
    "  - `pylate/` and `examples/` — pylate vendored, with the canonical "
    "kd training shape under `examples/train/knowledge_distillation.py`. "
    "the model module that wraps the dense layers is at "
    "`pylate/models/Dense.py`.\n"
    "  - `model_files/` — the released 17m. `1_Dense/` is Linear(256 -> "
    "512, identity), `2_Dense/` is Linear(512 -> 48, identity); both "
    "`config.json` files spell out the activation. weights are in the "
    "safetensors next to each config.\n"
    "  - `scripts/svd_per_layer.py` — what i ran to dump the singular "
    "spectrum of W1, W2, and W_eff = W2 @ W1.\n"
    "  - `scripts/equivalence_check.py` — what i ran to compare per-"
    "token outputs of `(hidden @ W1.T) @ W2.T` vs `hidden @ W_eff.T` on "
    "real model outputs.\n"
    "  - `experiments/projection_decomposition/svd_per_layer.json` — the "
    "svd output. rank-for-95%-energy, effective rank, etc.\n"
    "  - `experiments/projection_decomposition/equivalence_check.json` — "
    "the per-text + overall diff between the two pipelines.\n"
    "  - `experiments/projection_layer_ablation/results.json` — the two "
    "cells' nanobeir averages.\n\n"
    "write `analysis.md` in this workspace root. needs: (a) a yes/no "
    "answer on expressivity, grounded in svd_per_layer.json + "
    "equivalence_check.json, (b) a mechanism for the +1 ndcg gap that "
    "is consistent with the equivalence finding, (c) a defensible "
    "writeup framing for the report, and (d) a recommendation for the "
    "next head-design iteration. no internet, no live training. work "
    "from what's on disk."
)

RUBRIC = {
    "expressivity_refutation": (
        "A top answer concludes that the 2-layer FFN with identity "
        "activation has SAME expressivity as a single linear projection "
        "of the same input/output shape. Grounds this in two pieces of "
        "evidence: (1) the activations are identity (cited from the "
        "Dense config files), so the composition is matrix multiplication; "
        "(2) the SVD of W_eff in `svd_per_layer.json` shows rank 48 (full "
        "given the 48 output dim), which any single linear of the same "
        "shape could also achieve. Lower scores for hedging on "
        "expressivity, for asserting 2-layer is more expressive despite "
        "the identity activation, or for missing the rank argument."
    ),
    "equivalence_check_reading": (
        "A top answer cites the actual diff numbers from "
        "`experiments/projection_decomposition/equivalence_check.json`: "
        "max_abs_diff_overall ~1e-5, mean_abs_diff_overall ~1e-7, "
        "max_relative_diff ~1e-6. Connects these to float32 numerical "
        "noise — the per-text `cosine_per_token_min` is 1.0 to 6 "
        "decimals. Concludes the 2-layer and collapsed pipelines "
        "produce numerically-identical outputs at inference, which "
        "directly refutes the expressivity hypothesis empirically. "
        "Lower scores for ignoring the empirical check, treating ~1e-5 "
        "as a meaningful difference, or not citing specific values."
    ),
    "mechanism_proposal": (
        "A top answer identifies that the +1 NDCG gap must come from "
        "TRAINING DYNAMICS (optimization), not from the trained "
        "function's expressivity. Specifically: an overparameterized "
        "linear map (W = W2 @ W1 trained jointly) has different "
        "gradient flow than a single linear map W, because each weight "
        "matrix sees a gradient that depends on the other — this acts "
        "as an implicit per-axis learning rate / pre-conditioner. The "
        "trained function is in the same class as a single linear map; "
        "the training trajectory through that class is different and "
        "converges to a different solution. Bonus credit for citing "
        "Saxe et al. 2013 or Arora et al. 2018 on overparameterization "
        "of deep linear networks. Lower scores for hand-waving 'more "
        "params helps optimization' without engaging the mechanism, or "
        "for proposing causes that contradict the equivalence finding "
        "(e.g. 'the bottleneck dimension adds inductive bias' — there "
        "is no bottleneck since the inner dim 512 is larger than both "
        "endpoints)."
    ),
    "writeup_framing": (
        "A top answer recommends a writeup framing for the report that "
        "is consistent with the equivalence finding: do NOT frame the "
        "result as '2-layer FFN provides more capacity / expressivity'; "
        "frame it as 'overparameterized linear projection improves "
        "optimization' or similar. Distinguishes empirical claim "
        "(0.6405 > 0.6286, holds across seeds, so ship the 2-layer) "
        "from mechanistic claim (it's training dynamics, not "
        "expressivity). Lower scores for retaining the expressivity "
        "framing in the recommended writeup despite the evidence."
    ),
    "next_design_recommendation": (
        "A top answer suggests a credible next-iteration direction. "
        "Strongest pick: try a 3-layer version (same logic should "
        "extend; deeper linear chain may give further optimization "
        "benefit but diminishing returns). Second strongest: add a "
        "non-linearity between the layers (e.g. GELU), which would "
        "create a GENUINELY more-expressive function — separate axis. "
        "Acceptable: caution against assuming the optimization benefit "
        "transfers if hparams change (e.g. higher lr, different "
        "init). Lower scores for recommending 'use a single layer to "
        "save params' (loses the +1 NDCG), or for not engaging the "
        "design space at all."
    ),
    "svd_reading": (
        "A top answer reads `svd_per_layer.json` correctly: notes that "
        "W1 (512 x 256) is high-rank (~223 eff rank), W2 (48 x 512) is "
        "nearly full at its 48 dim (47.3 eff rank), W_eff (48 x 256) "
        "is also nearly full at 48 (46.8 eff rank). Notes that the "
        "rank of W_eff is bounded by min(rank W1, rank W2) — here the "
        "bottleneck is W2's output dim 48. The composed rank can never "
        "exceed what a single (256 x 48) linear could achieve. Lower "
        "scores for ignoring the per-layer SVDs or for misreading the "
        "rank bound."
    ),
    "receipt_grounding": (
        "A top answer ties each claim to a specific source: a config "
        "field in `model_files/1_Dense/config.json` or `2_Dense/"
        "config.json`, a field in `equivalence_check.json` or "
        "`svd_per_layer.json`, a cell in `results.json`, or a line in "
        "`pylate/models/Dense.py`. Lower scores for vague citations."
    ),
}

AXIS_WEIGHTS = {
    "expressivity_refutation":      0.20,
    "equivalence_check_reading":    0.15,
    "mechanism_proposal":           0.20,
    "writeup_framing":              0.13,
    "next_design_recommendation":   0.10,
    "svd_reading":                  0.15,
    "receipt_grounding":            0.07,
}

HARD_CAPS = [
    {
        "name": "asserts_2layer_more_expressive",
        "cap": 0.30,
        "description": (
            "Trigger if the writeup asserts the 2-layer FFN provides "
            "more expressivity than a single linear projection. The "
            "equivalence check (max relative diff ~1e-6) and the SVD "
            "(W_eff at full rank 48) directly refute this. Identity "
            "activation makes the two pipelines mathematically equivalent."
        ),
    },
    {
        "name": "missed_identity_activation",
        "cap": 0.35,
        "description": (
            "Trigger if the writeup does not engage that both layers "
            "use identity activation (per `model_files/1_Dense/"
            "config.json` and `2_Dense/config.json`). Without this "
            "observation the analysis cannot land — the expressivity "
            "argument hinges on the absence of a nonlinearity."
        ),
    },
    {
        "name": "missed_equivalence_evidence",
        "cap": 0.40,
        "description": (
            "Trigger if the writeup does not cite the equivalence_"
            "check.json numbers (max abs diff ~1e-5, max rel diff "
            "~1e-6, per-token cosine ~1.0). The empirical equivalence "
            "is load-bearing for refuting the expressivity hypothesis."
        ),
    },
    {
        "name": "blames_extra_params_capacity",
        "cap": 0.45,
        "description": (
            "Trigger if the writeup attributes the +1 NDCG to 'the "
            "extra parameters provide more capacity to learn'. Capacity "
            "/ expressivity is what the equivalence check refutes. The "
            "right framing is 'extra parameters change the optimization "
            "trajectory', NOT 'extra parameters increase capacity'."
        ),
    },
    {
        "name": "recommends_single_layer_to_save_params",
        "cap": 0.40,
        "description": (
            "Trigger if the writeup recommends shipping the single-layer "
            "linear projection 'to save parameters'. The team has the "
            "+1 NDCG measurement; shipping the worse cell to save a "
            "trivial number of params is the wrong trade. The "
            "recommendation should keep the 2-layer."
        ),
    },
    {
        "name": "no_mechanism_proposed",
        "cap": 0.50,
        "description": (
            "Trigger if the writeup refutes the expressivity hypothesis "
            "but offers no alternative mechanism for the +1 NDCG. "
            "Without an alternative, the writeup has nothing defensible "
            "to put in the report — it's just 'we don't know why'."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the writeup cites essentially no concrete "
            "values from the json files in `experiments/` or the "
            "config files in `model_files/`."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the writeup explicitly suggests testing the "
        "training-dynamics hypothesis with a controlled experiment: "
        "train BOTH variants (single-layer and 2-layer) but constrain "
        "the 2-layer's initialization or learning rate so it has the "
        "same effective optimization trajectory as the single-layer, "
        "and check whether the gap closes. Or: train the single-layer "
        "with a learning rate schedule that mimics the per-axis "
        "adaptation the deep linear network gives implicitly. This is "
        "the rigorous way to confirm the training-dynamics mechanism "
        "vs other unknown confounds."
    ),
}

task = diagnose_research_study.task(
    prompt=PROMPT,
    case="mxbai_projection_layer_choice",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="analysis.md",
    anti_fake={"min_verified": 3},
)
task.slug = "mxbai_projection_layer_choice"
task.metadata = {
    "category": "ml-researcher-hypothesis-refutation",
    "shape": "verify_or_refute_team_intuition_with_real_decomposition",
    "model": "mixedbread-ai/mxbai-edge-colbert-v0-17m",
    "key_measurements": {
        "ndcg_2layer_ffn": 0.6405,
        "ndcg_single_linear": 0.6286,
        "max_abs_diff_equivalence_check": 1e-5,
        "max_rel_diff_equivalence_check": 1e-6,
        "rank_full_w_eff": 48,
        "eff_rank_w_eff": 46.78,
    },
}
