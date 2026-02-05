import logging
from typing import Any, Tuple
import torch

from .utils import compute_lambda, shrink_l1_lq, calculate_errorcoefficient
from .selection import find_representatives, remove_representatives

logger = logging.getLogger(__name__)


def admm_main(
    Y: torch.Tensor,
    alpha: float = 5.0,
    q: int | float = 2,
    thr: float = 1e-7,
    maxIter: int = 5000,
    verbose: bool = True,
    logging_enabled: bool = False,
) -> Tuple[torch.Tensor, Tuple[float, float], dict[str, list[float]] | None]:
    """
    ADMM for finding sparse representation with or without affine constraints.

    Parameters:
    - Y: DxN data matrix of N data points in D-dimensional space (torch tensor).
    - alpha: regularization parameter.
    - q: norm for L1/Lq minimization.
    - thr: stopping threshold for coefficient error ||Z - C||.
    - maxIter: maximum number of ADMM iterations.
    - verbose: bool, if True, print iteration errors.
    - logging_enabled: bool, if True, track and return convergence history

    Returns:
    - Z2: NxN sparse coefficient matrix.
    - Err: final error(s).
    """
    _, N = Y.shape
    Y = Y.double()

    # Setting penalty parameters
    mu = alpha * 1 / compute_lambda(Y)
    rho = alpha

    P = Y.T @ Y

    V = torch.inverse(
        mu * P
        + rho * torch.eye(N, device=Y.device, dtype=Y.dtype)
        + rho * torch.ones((N, N), device=Y.device, dtype=Y.dtype)
    )
    Z_previous = torch.zeros((N, N), device=Y.device, dtype=Y.dtype)
    gamma1 = torch.zeros((N, N), device=Y.device, dtype=Y.dtype)
    gamma2 = torch.zeros(N, device=Y.device, dtype=Y.dtype)

    err1 = 10 * thr
    err2 = 10 * thr
    i = 1

    if logging_enabled:
        logs = {
            "primal_residual": [],
            "dual_residual": [],
            "affine_constraint_error": [],
        }

    while (err1 > thr or err2 > thr) and i < maxIter:
        # Update C
        C = V @ (
            mu * P
            + rho * (Z_previous - gamma1 / rho)
            + rho * torch.ones((N, N), device=Y.device, dtype=Y.dtype)
            + gamma2.unsqueeze(1).repeat(1, N)
        )

        # Update C using the proximal operator
        Z_current = shrink_l1_lq(C + gamma1 / rho, 1 / rho, q)

        # Update Lagrange multipliers
        gamma1 += rho * (C - Z_current)
        gamma2 += rho * (
            torch.ones(N, device=Y.device, dtype=Y.dtype) - torch.sum(C, dim=0)
        )

        # Compute errors
        err1 = calculate_errorcoefficient(C, Z_current)
        err2 = calculate_errorcoefficient(
            torch.sum(C, dim=0), torch.ones(N, device=Y.device, dtype=Y.dtype)
        )

        if logging_enabled:
            dual_res = rho * calculate_errorcoefficient(Z_current, Z_previous)

            logs["primal_residual"].append(err1.item())
            logs["dual_residual"].append(dual_res.item())
            logs["affine_constraint_error"].append(err2.item())

        Z_previous = Z_current
        i += 1

        if verbose and i % 100 == 0:
            logger.info(
                f"Iteration {i}, || Z - C || = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}"
            )

    Err = (err1, err2)
    if verbose:
        logger.info(
            f"Terminating ADMM at iteration {i:5d}, "
            "||Z - C|| = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}."
        )

    if logging_enabled:
        return Z_current, Err, logs
    else:
        return Z_current, Err


def sparse_modeling_representative_selection(
    Y: torch.Tensor,
    alpha: float = 5.0,
    r: int = 0,
    verbose: bool = True,
    max_iterations: int = 5000,
    logging_enabled: bool = False,
) -> Tuple[list[int], torch.Tensor, Any]:
    """
    Sparse Modeling Representative Selection (SMRS) function.

    Parameters:
    - Y: DxN data matrix of N data points in D-dimensional space (torch tensor).
    - alpha: regularization parameter, typically in [2, 50].
    - verbose: if True, prints information during iterations.
    - max_iterations: maximum number of ADMM iterations.
    - logging_enabled: enables logging for convergence testing.

    Returns:
    - representative_indices: indices of selected representative points.
    - C: NxN sparse coefficient matrix.
    """
    if not verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    # Force Y to be double precision
    Y = Y.double()
    q = 2
    thr = 1e-7
    max_iterations = max_iterations
    threshold_selection = 0.99  # threshold for find_representatives
    threshold_pruning = 0.95  # threshold for remove_representatives default = 0.95
    Y.shape[1]

    # Center the data matrix by subtracting the mean of each feature
    Y = Y - torch.mean(Y, dim=1, keepdim=True).double()

    # Compute the sparse coefficient matrix C using ADMM
    if logging_enabled:
        C, _, logs = admm_main(
            Y,
            alpha=alpha,
            q=q,
            thr=thr,
            maxIter=max_iterations,
            verbose=verbose,
            logging_enabled=logging_enabled,
        )
    else:
        C, _ = admm_main(
            Y,
            alpha=alpha,
            q=q,
            thr=thr,
            maxIter=max_iterations,
            verbose=verbose,
            logging_enabled=logging_enabled,
        )
    C = C.double()

    # Select representatives based on sparsity structure in C
    selected_indices = find_representatives(C, threshold_selection, q)
    representative_indices = remove_representatives(
        selected_indices, Y, threshold_pruning
    )

    if logging_enabled:
        return representative_indices, C, logs
    return representative_indices, C
