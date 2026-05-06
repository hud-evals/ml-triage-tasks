"""
LogitLens evaluation runner for architecture comparison.

Implements the Engram-style analysis:
- Compute KL(p_final || p_layer) for each layer
- Generate comparison plots across architectures
- Save structured results for publication

Usage:
    python -m nmoe.research.physics.eval_logitlens \
        --output /data/physics/logitlens_eval \
        --variants baseline engram ple_ngrammer canon canon+engram canon+ple \
        --task ngram_polysemy \
        --steps 2000 \
        --dim 512
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nmoe.research.physics.arch_ablations import (
    AblationTransformer,
    ModelCfg,
    RunCfg,
    Variant,
    _build_split,
    _evaluate,
    _evaluate_logitlens,
    _train_variant,
)
from nmoe.research.physics.viz import (
    LogitLensResult,
    GateProfileResult,
    TaskComparisonResult,
    plot_logitlens_curves,
    plot_logitlens_heatmap,
    plot_gate_profiles,
    plot_task_comparison,
)


# Predefined variant configurations for easy comparison
VARIANT_PRESETS = {
    "baseline": Variant(width="fixed", residual="vanilla", memory="none", attn="global"),
    "engram": Variant(width="fixed", residual="vanilla", memory="engram", attn="global"),
    "ple_ngrammer": Variant(width="fixed", residual="vanilla", memory="ple_ngrammer", attn="global"),
    "canon": Variant(width="fixed", residual="vanilla", memory="none", attn="global", precond="canon"),
    "canon+engram": Variant(width="fixed", residual="vanilla", memory="engram", attn="global", precond="canon"),
    "canon+ple": Variant(width="fixed", residual="vanilla", memory="ple_ngrammer", attn="global", precond="canon"),
    "mhc": Variant(width="fixed", residual="mhc", memory="none", attn="global"),
    "mhc+engram": Variant(width="fixed", residual="mhc", memory="engram", attn="global"),
    "altup": Variant(width="fixed", residual="altup", memory="none", attn="global"),
    "altup+engram": Variant(width="fixed", residual="altup", memory="engram", attn="global"),
    # Local attention variants
    "canon+engram+local": Variant(width="fixed", residual="vanilla", memory="engram", attn="local", attn_window=64, precond="canon"),
    "canon+ple+local": Variant(width="fixed", residual="vanilla", memory="ple_ngrammer", attn="local", attn_window=64, precond="canon"),
}


@dataclass
class EvalResult:
    """Full evaluation result for one variant."""
    variant_name: str
    variant: dict
    task_acc: dict[str, float]
    logitlens_kl: list[float]
    gate_profile: list[float] | None  # Only for gated memory
    final_loss: float
    params: int


def run_logitlens_eval(
    *,
    output_dir: Path,
    variants: list[str],
    task: str,
    dim: int,
    n_layers: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, EvalResult]:
    """
    Run LogitLens evaluation for a set of variants.

    Returns dict mapping variant name -> EvalResult
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model config
    model_cfg = ModelCfg(
        dim=dim,
        hidden=dim * 4,
        hidden_small=dim,
        n_layers=n_layers,
        n_heads=max(4, dim // 64),
    )

    # Build run config
    run_cfg = RunCfg(
        output=output_dir,
        steps=steps,
        seed=seed,
        tasks=(f"{task}:1.0:n_symbols=512,n_steps=128",),
        model=model_cfg,
        batch_size=32 if dim <= 512 else (16 if dim <= 2048 else 4),
    )

    # Build data splits
    print(f"Building data splits for task: {task}")
    train_split = _build_split(
        tasks=run_cfg.tasks,
        n=run_cfg.n_train,
        seq_len=run_cfg.seq_len,
        seed=seed,
    )
    valid_split = _build_split(
        tasks=run_cfg.tasks,
        n=run_cfg.n_valid,
        seq_len=run_cfg.seq_len,
        seed=seed + 1_000_000,
    )

    results: dict[str, EvalResult] = {}

    for var_name in variants:
        print(f"\n{'='*60}")
        print(f"Evaluating: {var_name}")
        print(f"{'='*60}")

        if var_name not in VARIANT_PRESETS:
            print(f"  WARNING: Unknown variant '{var_name}', skipping")
            continue

        variant = VARIANT_PRESETS[var_name]

        # Train the model
        train_result = _train_variant(
            cfg=run_cfg,
            variant=variant,
            train=train_split,
            valid=valid_split,
            device=device,
        )

        # Load trained model for LogitLens eval
        model = AblationTransformer(variant=variant, cfg=model_cfg).to(device)
        # Note: In a real setup, we'd save/load checkpoints. For now, we re-evaluate
        # the model that was just trained (it's still in memory).

        # Actually, _train_variant doesn't return the model, so we need to re-create
        # and the trained weights are lost. For proper eval, we need to modify this.
        # For now, let's just use the final metrics from training.

        # Run LogitLens evaluation on the final model
        # We need to re-create and train to get LogitLens - this is a TODO
        # For now, extract what we can from training logs

        # Get final validation metrics
        final_valid = train_result.get("final", {}).get("valid", {})

        # Extract gate profile if available
        gate_profile = None
        if variant.memory == "engram":
            # Parse from training log
            log_path = Path(train_result["train_log"])
            if log_path.exists():
                with open(log_path) as f:
                    lines = f.readlines()
                    if lines:
                        last_line = json.loads(lines[-1])
                        gate_profile = []
                        for i in range(n_layers):
                            key = f"mem.layer{i}.gate_mean"
                            if key in last_line.get("stats", {}):
                                gate_profile.append(last_line["stats"][key])
                            elif key in last_line:
                                gate_profile.append(last_line[key])

        results[var_name] = EvalResult(
            variant_name=var_name,
            variant=asdict(variant),
            task_acc={task: final_valid.get("answer_token_acc", 0.0)},
            logitlens_kl=[],  # TODO: Need to implement proper LogitLens collection during training
            gate_profile=gate_profile,
            final_loss=final_valid.get("loss", float("inf")),
            params=0,  # TODO: Count params
        )

        print(f"  Final accuracy: {final_valid.get('answer_token_acc', 0):.3f}")
        print(f"  Final loss: {final_valid.get('loss', float('inf')):.3f}")
        if gate_profile:
            print(f"  Gate profile: {[f'{g:.3f}' for g in gate_profile]}")

    return results


def run_logitlens_eval_with_collection(
    *,
    output_dir: Path,
    variants: list[str],
    task: str,
    dim: int,
    n_layers: int,
    steps: int,
    seed: int,
    device: torch.device,
    eval_every: int = 500,
) -> dict[str, EvalResult]:
    """
    Run LogitLens evaluation with proper hidden state collection.

    This version collects LogitLens KL at evaluation checkpoints during training.
    """
    import numpy as np
    from nmoe.research.physics.data.generators import EOS

    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelCfg(
        dim=dim,
        hidden=dim * 4,
        hidden_small=dim,
        n_layers=n_layers,
        n_heads=max(4, dim // 64),
    )

    run_cfg = RunCfg(
        output=output_dir,
        steps=steps,
        seed=seed,
        tasks=(f"{task}:1.0:n_symbols=512,n_steps=128",),
        model=model_cfg,
        batch_size=32 if dim <= 512 else (16 if dim <= 2048 else 4),
        eval_every=eval_every,
    )

    train_split = _build_split(
        tasks=run_cfg.tasks,
        n=run_cfg.n_train,
        seq_len=run_cfg.seq_len,
        seed=seed,
    )
    valid_split = _build_split(
        tasks=run_cfg.tasks,
        n=run_cfg.n_valid,
        seq_len=run_cfg.seq_len,
        seed=seed + 1_000_000,
    )

    results: dict[str, EvalResult] = {}

    for var_name in variants:
        print(f"\n{'='*60}")
        print(f"Evaluating: {var_name}")
        print(f"{'='*60}")

        if var_name not in VARIANT_PRESETS:
            print(f"  WARNING: Unknown variant '{var_name}', skipping")
            continue

        variant = VARIANT_PRESETS[var_name]

        # Create and train model with LogitLens collection
        model = AblationTransformer(variant=variant, cfg=model_cfg).to(device)
        model.train()

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(run_cfg.lr),
            weight_decay=float(run_cfg.weight_decay),
            betas=(0.9, 0.95),
        )

        rng = np.random.default_rng(seed)
        n_train = train_split.tokens.size(0)

        final_logitlens_kl = []
        final_gate_profile = []
        final_acc = 0.0
        final_loss = float("inf")

        for step in range(1, steps + 1):
            # Sample batch
            idx = rng.integers(0, n_train, size=run_cfg.batch_size)
            batch = train_split.tokens[idx].to(device)
            x_in = batch[:, :-1]
            y = batch[:, 1:]

            # Forward pass
            opt.zero_grad()
            logits, stats = model(x_in, collect_stats=True, collect_hiddens=False)

            # Compute loss (answer-only)
            from nmoe.research.physics.arch_ablations import _answer_mask_from_input, ANSWER_START
            mask = _answer_mask_from_input(x_in, eos_token_id=int(EOS)) & (y != int(EOS))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.masked_fill(~mask, -100).view(-1),
                ignore_index=-100,
            )
            loss.backward()
            opt.step()

            # Evaluation with LogitLens
            if step % eval_every == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    # Run LogitLens evaluation
                    logitlens_kl = _evaluate_logitlens(
                        model=model,
                        split=valid_split,
                        batch_size=run_cfg.batch_size,
                        loss_mode="answer_only",
                        device=device,
                    )

                    # Run standard evaluation
                    eval_metrics = _evaluate(
                        model=model,
                        split=valid_split,
                        batch_size=run_cfg.batch_size,
                        loss_mode="answer_only",
                        device=device,
                    )

                final_logitlens_kl = logitlens_kl
                final_acc = eval_metrics["answer_token_acc"]
                final_loss = eval_metrics["loss"]

                # Extract gate profile
                if variant.memory == "engram":
                    final_gate_profile = [
                        stats.get(f"mem.layer{i}.gate_mean", 0.0)
                        for i in range(n_layers)
                    ]

                print(f"  Step {step}: acc={final_acc:.3f}, loss={final_loss:.3f}")
                print(f"    LogitLens KL: {[f'{kl:.2f}' for kl in logitlens_kl]}")

                model.train()

        # Save result
        params = sum(p.numel() for p in model.parameters())
        results[var_name] = EvalResult(
            variant_name=var_name,
            variant=asdict(variant),
            task_acc={task: final_acc},
            logitlens_kl=final_logitlens_kl,
            gate_profile=final_gate_profile if final_gate_profile else None,
            final_loss=final_loss,
            params=params,
        )

        # Clean up
        del model, opt
        torch.cuda.empty_cache()

    return results


def generate_plots(
    results: dict[str, EvalResult],
    output_dir: Path,
    task: str,
) -> None:
    """Generate all visualization plots from evaluation results."""

    # LogitLens curves
    logitlens_results = [
        LogitLensResult(name=r.variant_name, kl_by_layer=r.logitlens_kl)
        for r in results.values()
        if r.logitlens_kl
    ]
    if logitlens_results:
        plot_logitlens_curves(
            logitlens_results,
            title=f"LogitLens: Prediction Convergence ({task})",
            output_path=output_dir / "logitlens_curves.pdf",
        )
        plot_logitlens_heatmap(
            logitlens_results,
            title=f"LogitLens KL Heatmap ({task})",
            output_path=output_dir / "logitlens_heatmap.pdf",
        )

    # Gate profiles (for Engram variants)
    gate_results = [
        GateProfileResult(name=r.variant_name, gate_by_layer=r.gate_profile)
        for r in results.values()
        if r.gate_profile
    ]
    if gate_results:
        plot_gate_profiles(
            gate_results,
            title="Gate Activation by Layer",
            output_path=output_dir / "gate_profiles.pdf",
        )

    # Task comparison
    task_results = [
        TaskComparisonResult(name=r.variant_name, task_acc=r.task_acc)
        for r in results.values()
    ]
    if task_results:
        plot_task_comparison(
            task_results,
            title=f"Task Accuracy Comparison ({task})",
            output_path=output_dir / "task_comparison.pdf",
        )


def main():
    parser = argparse.ArgumentParser(description="LogitLens architecture evaluation")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--variants", type=str, nargs="+", default=["baseline", "engram", "ple_ngrammer", "canon", "canon+engram", "canon+ple"],
                       help="Variants to evaluate")
    parser.add_argument("--task", type=str, default="ngram_polysemy", help="Task to evaluate on")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate LogitLens every N steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run evaluation with LogitLens collection
    results = run_logitlens_eval_with_collection(
        output_dir=args.output,
        variants=args.variants,
        task=args.task,
        dim=args.dim,
        n_layers=args.n_layers,
        steps=args.steps,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
    )

    # Save results
    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {k: asdict(v) for k, v in results.items()},
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # Generate plots
    generate_plots(results, args.output, args.task)
    print(f"Plots saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {r.task_acc.get(args.task, 0):.3f}")
        print(f"  Loss: {r.final_loss:.3f}")
        if r.logitlens_kl:
            print(f"  LogitLens KL (L0→L{len(r.logitlens_kl)-1}): {r.logitlens_kl[0]:.2f} → {r.logitlens_kl[-1]:.2f}")
        if r.gate_profile:
            print(f"  Gate (L0→L{len(r.gate_profile)-1}): {r.gate_profile[0]:.3f} → {r.gate_profile[-1]:.3f}")


if __name__ == "__main__":
    main()
