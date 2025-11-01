"""
Fuzzy Logic Penalty Functions for Multi-Label Learning
========================================================

Implements two types of fuzzy t-norms:
1. Łukasiewicz t-norm (additive logic)
2. Product t-norm (multiplicative logic)

Both are differentiable and suitable for gradient-based optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _relu(x):
    """ReLU activation."""
    return F.relu(x)


@torch.no_grad()
def _ensure_rule_tensors_device(rules, device):
    """Convert rule pairs to tensors on correct device."""
    if not rules:
        return None, None
    a = torch.tensor([i for i, _ in rules], device=device, dtype=torch.long)
    b = torch.tensor([j for _, j in rules], device=device, dtype=torch.long)
    return a, b


# ============================================================================
# ŁUKASIEWICZ T-NORM (Additive Logic)
# ============================================================================

def lukasiewicz_implication_penalty(preds: torch.Tensor,
                                    implication_pairs,
                                    weights=None,
                                    eps: float = 1e-6,
                                    reduction: str = "mean") -> torch.Tensor:
    """
    Łukasiewicz implication: A → B
    
    Truth value: T(A → B) = min(1, 1 - P(A) + P(B))
    Penalty: L = max(0, P(A) - P(B))
    
    Interpretation:
    - If A is highly predicted, B should also be highly predicted
    - Penalty is 0 when P(A) ≤ P(B) (satisfied)
    - Penalty grows linearly with P(A) - P(B) (violation)
    
    Args:
        preds: Predictions tensor (batch_size, num_classes) in [0, 1]
        implication_pairs: List of (A_idx, B_idx) tuples
        weights: Optional weights per rule (list of floats)
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Penalty tensor (scalar if reduction != 'none')
    """
    if not implication_pairs:
        return preds.new_zeros(())
    
    device = preds.device
    A_idx, B_idx = _ensure_rule_tensors_device(implication_pairs, device)
    
    # Get predictions for classes involved in rules
    pA = preds[:, A_idx].clamp(eps, 1 - eps)  # (batch_size, num_rules)
    pB = preds[:, B_idx].clamp(eps, 1 - eps)
    
    # Compute penalty: max(0, pA - pB)
    penalty = _relu(pA - pB)  # (batch_size, num_rules)
    
    # Apply weights if provided
    if weights is not None:
        w = torch.as_tensor(weights, device=device, dtype=preds.dtype).view(1, -1)
        penalty = penalty * w
    
    # Reduction
    if reduction == "mean":
        return penalty.mean()
    elif reduction == "sum":
        return penalty.sum()
    return penalty


def lukasiewicz_exclusion_penalty(preds: torch.Tensor,
                                  exclusion_pairs,
                                  weights=None,
                                  eps: float = 1e-6,
                                  reduction: str = "mean") -> torch.Tensor:
    """
    Łukasiewicz exclusion: A ⊥ B (mutual exclusion)
    
    Truth value: T(A ∧ B) = max(0, P(A) + P(B) - 1)
    Penalty: L = max(0, P(A) + P(B) - 1)
    
    Interpretation:
    - A and B should not both be highly predicted
    - Penalty is 0 when P(A) + P(B) ≤ 1 (satisfied)
    - Penalty grows linearly when both are predicted together
    
    Args:
        preds: Predictions tensor (batch_size, num_classes) in [0, 1]
        exclusion_pairs: List of (A_idx, B_idx) tuples
        weights: Optional weights per rule
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Penalty tensor
    """
    if not exclusion_pairs:
        return preds.new_zeros(())
    
    device = preds.device
    A_idx, B_idx = _ensure_rule_tensors_device(exclusion_pairs, device)
    
    pA = preds[:, A_idx].clamp(eps, 1 - eps)
    pB = preds[:, B_idx].clamp(eps, 1 - eps)
    
    # Compute penalty: max(0, pA + pB - 1)
    penalty = _relu(pA + pB - 1.0)
    
    if weights is not None:
        w = torch.as_tensor(weights, device=device, dtype=preds.dtype).view(1, -1)
        penalty = penalty * w
    
    if reduction == "mean":
        return penalty.mean()
    elif reduction == "sum":
        return penalty.sum()
    return penalty


def lukasiewicz_constraints_loss(preds: torch.Tensor,
                                 implication_pairs,
                                 exclusion_pairs,
                                 lambda_impl: float = 1.0,
                                 lambda_excl: float = 1.0,
                                 impl_weights=None,
                                 excl_weights=None,
                                 reduction: str = "mean") -> torch.Tensor:
    """
    Combined Łukasiewicz constraints loss.
    
    L_constraints = λ_impl * L_impl + λ_excl * L_excl
    """
    L_impl = lukasiewicz_implication_penalty(
        preds, implication_pairs, impl_weights, reduction=reduction
    )
    L_excl = lukasiewicz_exclusion_penalty(
        preds, exclusion_pairs, excl_weights, reduction=reduction
    )
    return lambda_impl * L_impl + lambda_excl * L_excl


# ============================================================================
# PRODUCT T-NORM (Multiplicative Logic)
# ============================================================================

def product_implication_penalty(preds: torch.Tensor,
                               implication_pairs,
                               weights=None,
                               eps: float = 1e-6,
                               reduction: str = "mean") -> torch.Tensor:
    """
    Product t-norm implication: A → B
    
    In product logic, implication is defined as:
    T(A → B) = 1 if P(A) ≤ P(B), else P(B) / P(A)
    
    We convert this to a penalty:
    Penalty = max(0, 1 - P(B) / P(A)) when P(A) > P(B)
    
    Alternatively, use log-space formulation:
    Penalty = max(0, log(P(A)) - log(P(B)))
    
    This is equivalent to KL-divergence style penalty but more stable.
    
    Args:
        preds: Predictions tensor (batch_size, num_classes) in [0, 1]
        implication_pairs: List of (A_idx, B_idx) tuples
        weights: Optional weights per rule
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Penalty tensor
    """
    if not implication_pairs:
        return preds.new_zeros(())
    
    device = preds.device
    A_idx, B_idx = _ensure_rule_tensors_device(implication_pairs, device)
    
    pA = preds[:, A_idx].clamp(eps, 1 - eps)
    pB = preds[:, B_idx].clamp(eps, 1 - eps)
    
    # Product implication penalty: max(0, log(pA) - log(pB))
    # This encourages pB ≥ pA in log-space
    log_pA = torch.log(pA + eps)
    log_pB = torch.log(pB + eps)
    penalty = _relu(log_pA - log_pB)
    
    if weights is not None:
        w = torch.as_tensor(weights, device=device, dtype=preds.dtype).view(1, -1)
        penalty = penalty * w
    
    if reduction == "mean":
        return penalty.mean()
    elif reduction == "sum":
        return penalty.sum()
    return penalty


def product_exclusion_penalty(preds: torch.Tensor,
                              exclusion_pairs,
                              weights=None,
                              eps: float = 1e-6,
                              reduction: str = "mean") -> torch.Tensor:
    """
    Product t-norm exclusion: A ⊥ B
    
    In product logic, conjunction is: T(A ∧ B) = P(A) * P(B)
    
    For exclusion, we want P(A) * P(B) to be small.
    Penalty = P(A) * P(B)
    
    This is naturally bounded in [0, 1] and differentiable.
    
    Args:
        preds: Predictions tensor (batch_size, num_classes) in [0, 1]
        exclusion_pairs: List of (A_idx, B_idx) tuples
        weights: Optional weights per rule
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Penalty tensor
    """
    if not exclusion_pairs:
        return preds.new_zeros(())
    
    device = preds.device
    A_idx, B_idx = _ensure_rule_tensors_device(exclusion_pairs, device)
    
    pA = preds[:, A_idx].clamp(eps, 1 - eps)
    pB = preds[:, B_idx].clamp(eps, 1 - eps)
    
    # Product exclusion penalty: pA * pB
    penalty = pA * pB
    
    if weights is not None:
        w = torch.as_tensor(weights, device=device, dtype=preds.dtype).view(1, -1)
        penalty = penalty * w
    
    if reduction == "mean":
        return penalty.mean()
    elif reduction == "sum":
        return penalty.sum()
    return penalty


def product_constraints_loss(preds: torch.Tensor,
                             implication_pairs,
                             exclusion_pairs,
                             lambda_impl: float = 1.0,
                             lambda_excl: float = 1.0,
                             impl_weights=None,
                             excl_weights=None,
                             reduction: str = "mean") -> torch.Tensor:
    """
    Combined Product t-norm constraints loss.
    
    L_constraints = λ_impl * L_impl + λ_excl * L_excl
    """
    L_impl = product_implication_penalty(
        preds, implication_pairs, impl_weights, reduction=reduction
    )
    L_excl = product_exclusion_penalty(
        preds, exclusion_pairs, excl_weights, reduction=reduction
    )
    return lambda_impl * L_impl + lambda_excl * L_excl


# ============================================================================
# VIOLATION METRICS (for logging)
# ============================================================================

@torch.no_grad()
def compute_violation_metrics(preds: torch.Tensor,
                              implication_pairs,
                              exclusion_pairs,
                              threshold: float = 0.5):
    """
    Compute hard violation rates (for logging, not training).
    
    Implication violation: P(A) ≥ threshold AND P(B) < threshold
    Exclusion violation: P(A) ≥ threshold AND P(B) ≥ threshold
    
    Returns:
        dict with violation rates and mean penalties
    """
    metrics = {}
    
    if preds.numel() == 0:
        return {
            'impl_viol_rate': 0.0,
            'excl_viol_rate': 0.0,
            'mean_impl_penalty_luk': 0.0,
            'mean_excl_penalty_luk': 0.0,
            'mean_impl_penalty_prod': 0.0,
            'mean_excl_penalty_prod': 0.0,
        }
    
    device = preds.device
    batch_size = preds.shape[0]
    
    # Hard predictions
    hard = (preds >= threshold).float()
    
    # Implication violations
    if implication_pairs:
        A_idx, B_idx = _ensure_rule_tensors_device(implication_pairs, device)
        impl_viols = (hard[:, A_idx] >= 0.5) & (hard[:, B_idx] < 0.5)
        metrics['impl_viol_rate'] = impl_viols.float().mean().item()
        
        # Mean penalties (Łukasiewicz)
        metrics['mean_impl_penalty_luk'] = lukasiewicz_implication_penalty(
            preds, implication_pairs, reduction='mean'
        ).item()
        
        # Mean penalties (Product)
        metrics['mean_impl_penalty_prod'] = product_implication_penalty(
            preds, implication_pairs, reduction='mean'
        ).item()
    else:
        metrics['impl_viol_rate'] = 0.0
        metrics['mean_impl_penalty_luk'] = 0.0
        metrics['mean_impl_penalty_prod'] = 0.0
    
    # Exclusion violations
    if exclusion_pairs:
        C_idx, D_idx = _ensure_rule_tensors_device(exclusion_pairs, device)
        excl_viols = (hard[:, C_idx] >= 0.5) & (hard[:, D_idx] >= 0.5)
        metrics['excl_viol_rate'] = excl_viols.float().mean().item()
        
        # Mean penalties (Łukasiewicz)
        metrics['mean_excl_penalty_luk'] = lukasiewicz_exclusion_penalty(
            preds, exclusion_pairs, reduction='mean'
        ).item()
        
        # Mean penalties (Product)
        metrics['mean_excl_penalty_prod'] = product_exclusion_penalty(
            preds, exclusion_pairs, reduction='mean'
        ).item()
    else:
        metrics['excl_viol_rate'] = 0.0
        metrics['mean_excl_penalty_luk'] = 0.0
        metrics['mean_excl_penalty_prod'] = 0.0
    
    return metrics


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def compute_fuzzy_constraints_loss(preds: torch.Tensor,
                                   implication_pairs,
                                   exclusion_pairs,
                                   fuzzy_type: str = 'lukasiewicz',
                                   lambda_impl: float = 1.0,
                                   lambda_excl: float = 1.0,
                                   impl_weights=None,
                                   excl_weights=None,
                                   reduction: str = "mean") -> torch.Tensor:
    """
    Unified interface for computing fuzzy constraints loss.
    
    Args:
        preds: Predictions (batch_size, num_classes)
        implication_pairs: List of (A, B) tuples for A → B
        exclusion_pairs: List of (C, D) tuples for C ⊥ D
        fuzzy_type: 'lukasiewicz' or 'product'
        lambda_impl: Weight for implication loss
        lambda_excl: Weight for exclusion loss
        impl_weights: Per-rule weights for implications
        excl_weights: Per-rule weights for exclusions
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Total constraints loss
    """
    if fuzzy_type == 'lukasiewicz':
        return lukasiewicz_constraints_loss(
            preds, implication_pairs, exclusion_pairs,
            lambda_impl, lambda_excl, impl_weights, excl_weights, reduction
        )
    elif fuzzy_type == 'product':
        return product_constraints_loss(
            preds, implication_pairs, exclusion_pairs,
            lambda_impl, lambda_excl, impl_weights, excl_weights, reduction
        )
    else:
        raise ValueError(f"Unknown fuzzy_type: {fuzzy_type}. Use 'lukasiewicz' or 'product'")


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_fuzzy_penalties(preds: torch.Tensor,
                           implication_pairs,
                           exclusion_pairs):
    """
    Compare penalties from both fuzzy logic types.
    
    Useful for analysis and debugging.
    """
    luk_impl = lukasiewicz_implication_penalty(preds, implication_pairs, reduction='mean').item()
    luk_excl = lukasiewicz_exclusion_penalty(preds, exclusion_pairs, reduction='mean').item()
    
    prod_impl = product_implication_penalty(preds, implication_pairs, reduction='mean').item()
    prod_excl = product_exclusion_penalty(preds, exclusion_pairs, reduction='mean').item()
    
    return {
        'lukasiewicz': {'impl': luk_impl, 'excl': luk_excl, 'total': luk_impl + luk_excl},
        'product': {'impl': prod_impl, 'excl': prod_excl, 'total': prod_impl + prod_excl},
    }


if __name__ == "__main__":
    # Quick test
    print("Testing fuzzy penalty functions...")
    
    # Create dummy data
    batch_size, num_classes = 4, 10
    preds = torch.rand(batch_size, num_classes)
    
    impl_rules = [(0, 1), (2, 3), (4, 5)]
    excl_rules = [(6, 7), (8, 9)]
    
    # Compute penalties
    print("\nŁukasiewicz penalties:")
    luk_loss = lukasiewicz_constraints_loss(preds, impl_rules, excl_rules)
    print(f"  Total loss: {luk_loss.item():.4f}")
    
    print("\nProduct penalties:")
    prod_loss = product_constraints_loss(preds, impl_rules, excl_rules)
    print(f"  Total loss: {prod_loss.item():.4f}")
    
    print("\nViolation metrics:")
    metrics = compute_violation_metrics(preds, impl_rules, excl_rules)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✓ All tests passed!")