"""
Independence loss (L_ind) for causal disentanglement

Uses HSIC (Hilbert-Schmidt Independence Criterion) between causal and non-causal
representations from CausalDisentanglementClassifier:
"causal_features" and "non_causal_features", each [B, *] with the same batch size B.

    L_ind = HSIC(causal_features, non_causal_features).

Feature trailing dimensions d_c and d_nc need not match; each side is flattened to
a 2D matrix [B, D] inside the estimator, and separate RBF kernels are used per branch.

Minimizing L_ind encourages statistical independence between the two branches (non-linear
dependence measured via RBF kernels). Uses an unbiased HSIC estimator with
Gaussian RBF kernels (median heuristic for bandwidth).

HSIC(X, Y) = 0 if and only if X and Y are independent (under general conditions).

Public API: independence_loss only.
"""
from typing import Union

import torch


def _gaussian_rbf_kernel(
    X: torch.Tensor,
    sigma_sq: Union[float, torch.Tensor],
    mask_diagonal: bool = True,
) -> torch.Tensor:
    """
    Compute Gaussian RBF kernel Gram matrix.

    K_ij = exp(-||x_i - x_j||^2 / (2*sigma^2)).
    Diagonal entries set to zero when mask_diagonal=True for unbiased HSIC.

    Args:
        X (torch.Tensor): Feature matrix, shape [B, D] (branch features).
        sigma_sq (Union[float, torch.Tensor]): Bandwidth variance (sigma^2);
            median heuristic or given.
        mask_diagonal (bool): If True, set diagonal to 0. Defaults to True.

    Returns:
        torch.Tensor: Gram matrix, shape [B, B].
    """
    xx = X @ X.T
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    dxx = rx.T + rx - xx * 2
    dxx = torch.clamp(dxx, min=0.0)

    sigma_sq_value = sigma_sq.item() if isinstance(
        sigma_sq, torch.Tensor) else float(sigma_sq)
    K = torch.exp(-0.5 * dxx / (sigma_sq_value + 1e-10))

    if mask_diagonal:
        b = X.shape[0]
        diag_zero = 1.0 - torch.eye(b, device=X.device, dtype=X.dtype)
        K = K * diag_zero

    return K


def _median_heuristic(X: torch.Tensor) -> torch.Tensor:
    """
    Compute kernel bandwidth via median heuristic.

    sigma^2 = median of pairwise squared distances (excluding diagonal).
    Used when sigma < 0 in HSIC computation.

    Args:
        X (torch.Tensor): Feature matrix, shape [B, D].

    Returns:
        torch.Tensor: Scalar sigma_sq (variance) for RBF kernel.
    """
    xx = X @ X.T
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    dxx = rx.T + rx - xx * 2
    dxx = torch.clamp(dxx, min=0.0)

    b = X.shape[0]
    off_diag_mask = ~torch.eye(b, dtype=torch.bool, device=X.device)
    valid_dxx = dxx[off_diag_mask]
    # quantile() requires float/double; pairwise distances follow X.dtype (may be
    # half or bfloat16 under autocast), so cast before quantile.
    sigma_sq = torch.quantile(valid_dxx.float(), 0.5).clamp(min=1e-10)
    return sigma_sq


def _hsic_unbiased(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: float = -1.0,
    sigma_y: float = -1.0,
) -> torch.Tensor:
    """
    Unbiased HSIC between paired rows (same batch index = same sample).

    X and Y are causal / non-causal features ("causal_features" / "non_causal_features"
    from CausalDisentanglementClassifier). Shapes [B, d_c] and [B, d_nc] with d_c != d_nc
    are valid; both are flattened to [B, -1] here.

    Args:
        X (torch.Tensor): Causal branch, shape [B, *], flattened to [B, -1].
        Y (torch.Tensor): Non-causal branch, same batch size B as X.
        sigma_x (float): RBF bandwidth for X. If < 0, median heuristic.
        sigma_y (float): RBF bandwidth for Y. If < 0, median heuristic.

    Returns:
        torch.Tensor: Scalar HSIC(X, Y) (non-negative).
    """
    X = X.flatten(start_dim=1)
    Y = Y.flatten(start_dim=1)
    b = X.shape[0]

    if b <= 3:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    vx = _median_heuristic(X) if sigma_x < 0 else sigma_x ** 2
    vy = _median_heuristic(Y) if sigma_y < 0 else sigma_y ** 2

    K = _gaussian_rbf_kernel(X, vx, mask_diagonal=True)
    L = _gaussian_rbf_kernel(Y, vy, mask_diagonal=True)

    # Gram matrix product (not KL divergence); unbiased HSIC uses trace(K @ L).
    K_L = K @ L
    trace = K_L.trace()
    second_term = K.sum() * L.sum() / ((b - 1) * (b - 2))
    third_term = K_L.sum() / (b - 2)

    hsic_value = trace + second_term - third_term * 2.0
    hsic_value = hsic_value / (b * (b - 3))
    hsic_value = torch.clamp(hsic_value, min=0.0)
    return hsic_value


def independence_loss(
    causal_features: torch.Tensor,
    non_causal_features: torch.Tensor,
    sigma_x: float = -1.0,
    sigma_y: float = -1.0,
) -> torch.Tensor:
    """
    Independence loss L_ind = HSIC(causal_features, non_causal_features).

    Aligns with CausalDisentanglementClassifier outputs: same batch size B per row;
    last dimensions for the two branches may differ and are flattened before HSIC.

    Args:
        causal_features (torch.Tensor): Causal branch, shape [B, d_c] or [B, *].
        non_causal_features (torch.Tensor): Non-causal branch, shape [B, d_nc] or [B, *].
        sigma_x (float): RBF bandwidth for causal features. -1 => median heuristic.
        sigma_y (float): RBF bandwidth for non-causal features. -1 => median heuristic.

    Returns:
        torch.Tensor: Scalar L_ind.
    """
    return _hsic_unbiased(
        causal_features, non_causal_features, sigma_x, sigma_y
    )
