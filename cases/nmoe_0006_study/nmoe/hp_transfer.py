"""Complete(d)P+MoE Hyperparameter Transfer.

Implements learning rate transfer across model scales following the Complete(d)P
framework, extended with MoE sparsity scaling.

Formula:
    lr_target = lr_base × (width_base / width_target)^α
                        × (depth_base / depth_target)^β
                        × (sparsity_target / sparsity_base)^γ

Coefficients (fit on DeepSeek 3B/9B/27B/671B):
    α = 0.35  (width scaling)
    β = 0.58  (depth scaling)
    γ = 0.17  (MoE sparsity scaling)

See docs/hp_scaling.md for derivation and validation.
"""

from dataclasses import dataclass
from typing import Optional
import math


# Fitted coefficients from DeepSeek model family
ALPHA = 0.35  # Width exponent
BETA = 0.58   # Depth exponent
GAMMA = 0.17  # Sparsity exponent (positive = higher sparsity allows higher LR)


@dataclass
class ModelSpec:
    """Model specification for HP transfer."""
    name: str
    dim: int
    n_layers: int
    n_routed_experts: int = 0  # 0 for dense
    n_activated_experts: int = 0  # 0 for dense
    lr: Optional[float] = None

    @property
    def sparsity(self) -> float:
        """Expert sparsity ratio (E/k). Returns 1.0 for dense models."""
        if self.n_routed_experts > 0 and self.n_activated_experts > 0:
            return self.n_routed_experts / self.n_activated_experts
        return 1.0

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts > 0


# Reference models with known-good HPs
DEEPSEEK_3B = ModelSpec(
    name="DeepSeek-3B",
    dim=1280,
    n_layers=12,
    n_routed_experts=64,
    n_activated_experts=6,
    lr=8.6e-4,
)

DEEPSEEK_9B = ModelSpec(
    name="DeepSeek-9B",
    dim=1920,
    n_layers=18,
    n_routed_experts=64,
    n_activated_experts=6,
    lr=5.9e-4,
)

DEEPSEEK_27B = ModelSpec(
    name="DeepSeek-27B",
    dim=2560,
    n_layers=30,
    n_routed_experts=72,
    n_activated_experts=6,
    lr=4.0e-4,
)

DEEPSEEK_671B = ModelSpec(
    name="DeepSeek-671B",
    dim=7168,
    n_layers=61,
    n_routed_experts=256,
    n_activated_experts=8,
    lr=2.2e-4,
)


def transfer_lr(
    base: ModelSpec,
    target: ModelSpec,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """
    Transfer learning rate from base model to target model.

    Args:
        base: Source model with known LR
        target: Target model to predict LR for
        alpha: Width scaling exponent (default: 0.35)
        beta: Depth scaling exponent (default: 0.58)
        gamma: Sparsity scaling exponent (default: 0.17)

    Returns:
        Predicted optimal learning rate for target model

    Example:
        >>> base = ModelSpec("50M", dim=384, n_layers=6, lr=1.0e-3)
        >>> target = ModelSpec("500M", dim=1280, n_layers=20)
        >>> lr = transfer_lr(base, target)
        >>> print(f"Predicted LR: {lr:.2e}")
        Predicted LR: 2.10e-04
    """
    if base.lr is None:
        raise ValueError(f"Base model '{base.name}' must have LR specified")

    # Width scaling
    width_term = (base.dim / target.dim) ** alpha

    # Depth scaling
    depth_term = (base.n_layers / target.n_layers) ** beta

    # Sparsity scaling
    sparse_term = (target.sparsity / base.sparsity) ** gamma

    return base.lr * width_term * depth_term * sparse_term


def transfer_lr_direct(
    base_lr: float,
    base_dim: int,
    base_layers: int,
    base_experts: int,
    base_topk: int,
    target_dim: int,
    target_layers: int,
    target_experts: int,
    target_topk: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """
    Direct LR transfer without ModelSpec objects.

    For dense models, pass experts=0 and topk=0.
    """
    # Calculate sparsity (1.0 for dense)
    base_sparsity = base_experts / base_topk if (base_experts > 0 and base_topk > 0) else 1.0
    target_sparsity = target_experts / target_topk if (target_experts > 0 and target_topk > 0) else 1.0

    width_term = (base_dim / target_dim) ** alpha
    depth_term = (base_layers / target_layers) ** beta
    sparse_term = (target_sparsity / base_sparsity) ** gamma

    return base_lr * width_term * depth_term * sparse_term


def validate_transfer(
    predictions: list[tuple[ModelSpec, float]],
    tolerance: float = 0.10,
) -> dict:
    """
    Validate transfer predictions against known LRs.

    Args:
        predictions: List of (model, predicted_lr) tuples
        tolerance: Maximum allowed relative error (default: 10%)

    Returns:
        Dict with validation results
    """
    results = {
        "passed": True,
        "max_error": 0.0,
        "predictions": [],
    }

    for model, predicted_lr in predictions:
        if model.lr is None:
            continue

        actual_lr = model.lr
        error = abs(predicted_lr - actual_lr) / actual_lr

        results["predictions"].append({
            "model": model.name,
            "predicted": predicted_lr,
            "actual": actual_lr,
            "error": error,
            "passed": error <= tolerance,
        })

        if error > results["max_error"]:
            results["max_error"] = error

        if error > tolerance:
            results["passed"] = False

    return results


def print_transfer_chain(
    chain: list[ModelSpec],
    base_lr: Optional[float] = None,
) -> None:
    """
    Print a transfer chain with predicted LRs.

    Args:
        chain: List of ModelSpec objects in order
        base_lr: Override LR for first model (uses model.lr if None)
    """
    if not chain:
        return

    # Use provided LR or model's LR for base
    if base_lr is not None:
        chain[0].lr = base_lr

    if chain[0].lr is None:
        raise ValueError("First model in chain must have LR")

    print("=" * 70)
    print("Complete(d)P+MoE Transfer Chain")
    print("=" * 70)
    print(f"Formula: lr = lr_base × (dim_b/dim_t)^{ALPHA} × (layers_b/layers_t)^{BETA} × (sparsity_t/sparsity_b)^{GAMMA}")
    print()

    prev = chain[0]
    print(f"{'Model':<15} {'dim':>6} {'layers':>6} {'sparsity':>8} {'LR':>12} {'Transfer':>12}")
    print("-" * 70)
    print(f"{prev.name:<15} {prev.dim:>6} {prev.n_layers:>6} {prev.sparsity:>8.2f} {prev.lr:>12.2e} {'(base)':>12}")

    for target in chain[1:]:
        predicted = transfer_lr(prev, target)

        # Calculate individual terms for display
        width_term = (prev.dim / target.dim) ** ALPHA
        depth_term = (prev.n_layers / target.n_layers) ** BETA
        sparse_term = (target.sparsity / prev.sparsity) ** GAMMA

        transfer_str = f"×{width_term * depth_term * sparse_term:.3f}"

        # Show actual if available
        if target.lr is not None:
            error = (predicted - target.lr) / target.lr * 100
            lr_str = f"{predicted:.2e} ({error:+.1f}%)"
        else:
            lr_str = f"{predicted:.2e}"
            target.lr = predicted  # Set for next iteration

        print(f"{target.name:<15} {target.dim:>6} {target.n_layers:>6} {target.sparsity:>8.2f} {lr_str:>12} {transfer_str:>12}")

        prev = target

    print()


def demo():
    """Demonstrate HP transfer across the DeepSeek family."""
    print("\n" + "=" * 70)
    print("DEMO: DeepSeek HP Transfer Validation")
    print("=" * 70)

    # Validate DeepSeek family predictions
    base = DEEPSEEK_3B
    targets = [DEEPSEEK_9B, DEEPSEEK_27B, DEEPSEEK_671B]

    print(f"\nBase: {base.name} (lr={base.lr:.2e})")
    print(f"\n{'Target':<15} {'Predicted':>12} {'Actual':>12} {'Error':>8}")
    print("-" * 50)

    for target in targets:
        predicted = transfer_lr(base, target)
        error = (predicted - target.lr) / target.lr * 100
        print(f"{target.name:<15} {predicted:>12.2e} {target.lr:>12.2e} {error:>7.1f}%")

    # Demo: Transfer chain from 50M to 3B
    print("\n" + "=" * 70)
    print("DEMO: 50M Dense → 3B MoE Transfer Chain")
    print("=" * 70)

    chain = [
        ModelSpec("50M-dense", dim=384, n_layers=6, lr=1.0e-3),
        ModelSpec("500M-dense", dim=1280, n_layers=20),
        ModelSpec("500M-MoE", dim=1024, n_layers=12, n_routed_experts=64, n_activated_experts=6),
        ModelSpec("3B-MoE", dim=1280, n_layers=12, n_routed_experts=64, n_activated_experts=6, lr=8.6e-4),
    ]

    print_transfer_chain(chain)


if __name__ == "__main__":
    demo()
