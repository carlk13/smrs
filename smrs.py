# Standard imports
from pathlib import Path
import numpy as np

# Torch imports
import torch


def compute_lambda(Y, affine=False):
    """
    Computes the regularization parameter lambda for the L1/Lq minimization.

    Parameters:
    - Y: torch tensor of shape (D, N), the data matrix with N data points in D dimensions.
    - affine: boolean, whether to use the affine constraint.

    Returns:
    - lambda_param: regularization parameter for the optimization problem.
    """

    Y = Y.double()
    _, N = Y.shape
    T = torch.zeros(N, device=Y.device, dtype=Y.dtype)

    if not affine:
        # Compute lambda without affine constraint
        for i in range(N):
            yi = Y[:, i]
            T[i] = torch.norm(torch.matmul(yi, Y))
        lambda_param = torch.max(T)
    else:
        # Compute lambda with affine constraint
        y_mean = torch.mean(Y, dim=1, keepdim=True).to(Y.dtype)
        ones_matrix = torch.ones((1, N), device=Y.device, dtype=Y.dtype)  # Same dtype
        for i in range(N):
            yi = Y[:, i].unsqueeze(0)  # Ensure (1, D) instead of (D,)
            affine_term = y_mean @ ones_matrix - Y  # Convert to same dtype
            T[i] = torch.norm(torch.matmul(yi, affine_term))  # No dtype mismatch
        lambda_param = torch.max(T)

    return lambda_param


def shrink_l1_lq(Z1, lambda_param, q=2):
    """
    Applies L1/Lq shrinkage for sparsity.

    Parameters:
    - Z1: torch tensor of shape (N, N), the coefficient matrix.
    - lambda_param: regularization parameter for shrinkage.
    - q: norm type (1 for L1, 2 for L1/L2, and float('inf') for L1/Linf).

    Returns:
    - Z2: torch tensor of shape (N, N), the shrunk matrix.
    """
    Z1 = Z1.double()
    D, N = Z1.shape
    if not isinstance(lambda_param, torch.Tensor):
        lambda_param = torch.tensor(lambda_param, dtype=Z1.dtype, device=Z1.device)
    else:
        lambda_param = lambda_param.double()

    if q == 1:
        Z2 = torch.maximum(
            torch.abs(Z1) - lambda_param, torch.zeros_like(Z1)
        ) * torch.sign(Z1)

    elif q == 2:
        # Rowwise shrinkage for L1/L2 norm
        row_norms = torch.norm(Z1, dim=1)
        r = torch.maximum(row_norms - lambda_param, torch.zeros_like(row_norms))
        r = r / (r + lambda_param)
        Z2 = r.unsqueeze(1) * Z1

    elif q == float("inf"):
        # Rowwise shrinkage for L1/Linf norm
        Z2 = torch.stack([shrink_l2_linf(row, lambda_param) for row in Z1])
    else:
        raise ValueError("Unsupported norm type. Use q=1, 2, or float('inf')")

    return Z2


def shrink_l2_linf(y, tau):
    """
    Minimizes 0.5 * ||x - y||_2^2 + tau * ||x||_inf

    Parameters:
    - y: torch tensor, the vector to shrink.
    - tau: regularization parameter

    Returns:
    - x: torch tensor, the shrunk vector.
    """
    y = y.double()
    x = y.clone()

    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, dtype=torch.float64, device=y.device)
    else:
        tau = tau.double()

    # Sort y by absolute values in descending order
    y_abs = torch.abs(y)
    y_sorted, indices_sorted = torch.sort(y_abs, descending=True)

    # Calculate cumulative sum for threshold check
    arange_tensor = torch.arange(1, len(y), device=y.device, dtype=torch.float64)
    # Calculate cumulative sum for threshold check
    cumulative_sum = (
        torch.cumsum(y_sorted[:-1], dim=0) / arange_tensor - tau / arange_tensor
    )

    # Find the cutoff index
    if cutoff_index < len(y):
        x[indices_sorted[:cutoff_index]] = torch.sign(
            y[indices_sorted[:cutoff_index]]
        ) * max(zbar - tau / cutoff_index, y_sorted[cutoff_index].item())
    else:
        x[indices_sorted[:cutoff_index]] = torch.sign(
            y[indices_sorted[:cutoff_index]]
        ) * max(zbar - tau / cutoff_index, 0)

    # Find the cutoff index
    d = cumulative_sum > y_sorted[1:]
    if not torch.any(d):
        cutoff_index = len(y)
    else:
        cutoff_index = torch.where(d)[0][0].item() + 1  # Convert to Python int

    zbar = torch.mean(y_sorted[:cutoff_index])
    # Compute the shrinkage threshold
    if cutoff_index < len(y):
        cutoff_val = torch.tensor(
            y_sorted[cutoff_index].item(), dtype=torch.float64, device=y.device
        )
        value = torch.maximum(zbar - tau / cutoff_index, cutoff_val)
    else:
        value = torch.maximum(
            zbar - tau / cutoff_index,
            torch.tensor(0.0, dtype=torch.float64, device=y.device),
        )

    x[indices_sorted[:cutoff_index]] = (
        torch.sign(y[indices_sorted[:cutoff_index]]) * value
    )

    return x


def calculate_errorcoefficient(Z, C):
    """
    Compute the normalized average absolute error between matrices Z and C for ADMM.

    Parameters:
    - Z: Current tensor in ADMM iteration.
    - C: Target tensor in ADMM iteration.

    Returns:
    - average_absolute_error: Normalized average absolute error between Z and C.

    Raises:
    ValueError:
        If Z and C have different shapes or unsupported dimensions (only 1D and 2D).
    """
    Z = Z.double()
    C = C.double()

    # Check for shape compatibility
    if Z.shape != C.shape:
        raise ValueError("Z and C must have the same shape.")

    # Determine the number of elements for normalization
    if Z.ndim == 1:  # Vector case
        num_elements = Z.shape[0]
    elif Z.ndim == 2:  # Matrix case
        num_elements = Z.shape[0] * Z.shape[1]
    else:
        raise ValueError("Unsupported tensor dimensionality.")

    # Calculate the average absolute error
    average_absolute_error = torch.sum(torch.abs(Z - C)) / num_elements

    return average_absolute_error


def admm_main(Y, affine=False, alpha=5, q=2, thr=1e-7, maxIter=5000, verbose=True):
    """
    ADMM for finding sparse representation with or without affine constraints.

    Parameters:
    - Y: DxN data matrix of N data points in D-dimensional space (torch tensor).
    - affine: bool, whether to enforce affine constraint.  default, = False because we assume subspaces are linear
    - alpha: regularization parameter.
    - q: norm for L1/Lq minimization.
    - thr: stopping threshold for coefficient error ||Z - C||.
    - maxIter: maximum number of ADMM iterations.
    - verbose: bool, if True, print iteration errors.

    Returns:
    - Z2: NxN sparse coefficient matrix.
    - Err: final error(s).
    """
    _, N = Y.shape
    Y = Y.double()

    # Setting penalty parameters
    mu = alpha * 1 / compute_lambda(Y, affine)
    rho = alpha

    P = Y.T @ Y

    if not affine:
        # --- NOT AFFINE CASE ---
        V = torch.inverse(mu * P + rho * torch.eye(N, device=Y.device, dtype=Y.dtype))
        Z1 = torch.zeros((N, N), device=Y.device, dtype=Y.dtype)
        gamma1 = torch.zeros((N, N), device=Y.device, dtype=Y.dtype)

        err1 = 10 * thr
        i = 1
        old_reps = list()

        # ADMM iterations for NOT AFFINE case
        while err1 > thr and i < maxIter:
            # Update C
            C = V @ (mu * P + rho * Z1 - gamma1)

            # Update Z using the proximal operator
            Z2 = shrink_l1_lq(C + gamma1 / rho, 1 / rho, q)

            # Update Lagrange multiplier
            gamma1 += rho * (C - Z2)

            # Compute error (coefficient error)
            err1 = calculate_errorcoefficient(C, Z2)

            Z1 = Z2
            i += 1

            if verbose and i % 100 == 0:
                print(f"Iteration {i}, || Z - C || = {err1:.5e}")
                threshold_selection = 0.99  # threshold for find_representatives
                threshold_pruning = 0.95  # threshold for remove_representatives
                selected_indices = find_representatives(Z2, threshold_selection, q)
                representative_indices = remove_representatives(
                    selected_indices, Y, threshold_pruning
                )
                print("-" * 80)
                print("Representative Indices:")
                print(representative_indices)
                print("-" * 80)
                if old_reps == representative_indices:
                    if verbose:
                        print("-" * 80)
                        print(
                            f"Terminating ADMM at iteration {i:5d}, \n ||Z - C|| = {err1:.5e}."
                        )
                        top_part = Z1[:5, :5]
                        print("Top part of the tensor:")
                        print(top_part)
                        print("-" * 80)
                    return Z2, err1
                old_reps = representative_indices

        Err = err1
        if verbose:
            print("-" * 80)
            print(f"Terminating ADMM at iteration {i:5d}, \n ||Z - C|| = {err1:.5e}.")
            top_part = Z1[:5, :5]
            print("Top part of the tensor:")
            print(top_part)
            print("-" * 80)
        return Z2, Err

    else:
        # --- AFFINE CASE ---
        V = torch.inverse(
            mu * P
            + rho * torch.eye(N, device=Y.device, dtype=Y.dtype)
            + rho * torch.ones((N, N), device=Y.device, dtype=Y.dtype)
        )
        Z1 = torch.zeros((N, N), device=Y.device, dtype=Y.dtype)
        gamma1 = torch.zeros((N, N), device=Y.device, dtype=Y.dtype)
        gamma2 = torch.zeros(N, device=Y.device, dtype=Y.dtype)

        err1 = 10 * thr
        err2 = 10 * thr
        i = 1
        old_reps = list()

        # ADMM iterations for AFFINE case
        while (err1 > thr or err2 > thr) and i < maxIter:
            # Update C
            C = V @ (
                mu * P
                + rho * (Z1 - gamma1 / rho)
                + rho * torch.ones((N, N), device=Y.device, dtype=Y.dtype)
                + gamma2.unsqueeze(1).repeat(1, N)
            )

            # Update C using the proximal operator
            Z2 = shrink_l1_lq(C + gamma1 / rho, 1 / rho, q)

            # Update Lagrange multipliers
            gamma1 += rho * (C - Z2)
            gamma2 += rho * (
                torch.ones(N, device=Y.device, dtype=Y.dtype) - torch.sum(C, dim=0)
            )

            # Compute errors
            err1 = calculate_errorcoefficient(C, Z2)
            err2 = calculate_errorcoefficient(
                torch.sum(C, dim=0), torch.ones(N, device=Y.device, dtype=Y.dtype)
            )

            Z1 = Z2
            i += 1

            if verbose and i % 100 == 0:
                print(
                    f"Iteration {i}, || Z - C || = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}"
                )
                threshold_selection = 0.99  # threshold for find_representatives
                threshold_pruning = 0.95  # threshold for remove_representatives
                selected_indices = find_representatives(Z2, threshold_selection, q)
                representative_indices = remove_representatives(
                    selected_indices, Y, threshold_pruning
                )
                print("-" * 80)
                print("Representative Indices:")
                print(representative_indices)
                print("-" * 80)
                if old_reps == representative_indices:
                    if verbose:
                        print("-" * 80)
                        print(
                            f"Terminating ADMM at iteration {i:5d}, \n ||Z - C|| = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}."
                        )
                        top_part = Z1[:5, :5]
                        print("Top part of the tensor:")
                        print(top_part)
                        print("-" * 80)
                    return Z2, (err1, err2)
                old_reps = representative_indices

        Err = (err1, err2)
        if verbose:
            print("-" * 80)
            print(
                f"Terminating ADMM at iteration {i:5d}, \n ||Z - C|| = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}."
            )
            top_part = Z1[:5, :5]
            print("Top part of the tensor:")
            print(top_part)
            print("-" * 80)
        return Z2, Err


def find_representatives(C, thr=0.99, q=2):
    """
    Identifies indices of nonzero rows in the coefficient matrix based on their norms
    or row-sparsity index (RSI).

    Parameters:
    - C: NxN coefficient matrix (torch tensor).
    - thr: Threshold for selecting rows based on cumulative norm.
    - q: Norm type (1 for L1, 2 for L2, float('inf') for Linf).

    Returns:
    - selected_indices: List of indices of selected representatives.
    """
    N, _ = C.shape
    C = C.double()

    row_norms = torch.norm(
        C, dim=1, p=q
    ).double()  # Compute the q-norm for each row in C
    non_outlier_indices = torch.arange(N, device=C.device, dtype=torch.long)

    # Sort norms in descending order and get indices
    sorted_norms, sorted_indices = torch.sort(row_norms, descending=True)

    # Determine the cut-off index where cumulative norm exceeds threshold
    cumulative_sum = 0
    total_norm_sum = torch.sum(sorted_norms).double()

    for j in range(len(sorted_norms)):
        cumulative_sum += sorted_norms[j]
        if cumulative_sum / total_norm_sum > thr:
            break
    else:
        # If the loop didn't break, set j to the last index
        j = len(sorted_norms) - 1

    # Indices of rows selected as representatives
    selected_indices = non_outlier_indices[sorted_indices[: j + 1]]

    return selected_indices


def remove_representatives(sInd, Y, thr=0.95):
    """
    Removes redundant representatives based on pairwise distances.

    Parameters:
    - sInd: indices of initial representative candidates.
    - Y: DxN data matrix (torch tensor).
    - thr: similarity threshold for pruning representatives.

    Returns:
    - pruned_representative_indices: pruned list of representative indices.
    """
    Y = Y.double()
    Ys = Y[:, sInd].double()
    Ns = Ys.shape[1]  # Number of columns

    # Compute pairwise Euclidean distances
    distances = torch.zeros((Ns, Ns), device=Y.device, dtype=Y.dtype)
    for i in range(Ns - 1):
        for j in range(i + 1, Ns):
            distances[i, j] = torch.norm(Ys[:, i] - Ys[:, j]).double()

    # Make distances matrix symmetric
    distances = distances + distances.T

    # Sort indices and distances by descending order for each column
    sorted_indices = torch.argsort(-distances, dim=0)
    sorted_distances = torch.gather(distances, 0, sorted_indices)

    # Initialize pruning index list
    pruned_indices = list(range(Ns))
    for i in range(Ns):
        if i in pruned_indices:
            cumulative_sum = 0
            t = 0
            total_distance_sum = torch.sum(sorted_distances[:, i]).double()

            # Use presorted distances to determine redundancy
            while (
                cumulative_sum <= thr * total_distance_sum
                and t < sorted_distances.shape[0]
            ):
                cumulative_sum += sorted_distances[t, i]
                t += 1

            redundant_indices = sorted_indices[t:, i]
            redundant_indices = [idx.item() for idx in redundant_indices if idx > i]
            pruned_indices = [
                idx for idx in pruned_indices if idx not in redundant_indices
            ]

    # Map pruned indices back to original representative indices
    pruned_representative_indices = [sInd[idx] for idx in pruned_indices]

    return pruned_representative_indices


def sparse_modeling_representative_selection(
    Y, alpha=5, r=0, verbose=True, delta=0.16, max_iterations=5000
):
    """
    Sparse Modeling Representative Selection (SMRS) function.

    Parameters:
    - Y: DxN data matrix of N data points in D-dimensional space (torch tensor).
    - alpha: regularization parameter, typically in [2, 50].
    - r: target dimensionality for optional projection, enter 0 to use original data.
    - verbose: if True, prints information during iterations.
    - delta: threshold for row sparsity index.
    - max_iterations: maximum number of ADMM iterations.

    Returns:
    - representative_indices: indices of selected representative points.
    - C: NxN sparse coefficient matrix.
    """
    # Force Y to be double precision
    Y = Y.double()
    q = 2
    affine = True
    thr = 1e-7
    max_iterations = max_iterations
    threshold_selection = 0.99  # threshold for find_representatives
    threshold_pruning = 0.95  # threshold for remove_representatives default = 0.95
    Y.shape[1]

    # Center the data matrix by subtracting the mean of each feature
    Y = Y - torch.mean(Y, dim=1, keepdim=True).double()

    # Optional dimensionality reduction using SVD if r is specified
    if r >= 1:
        # Use NumPy's SVD for dimensionality reduction
        Y_np = Y.cpu().numpy()
        _, S, Vt = np.linalg.svd(Y_np, full_matrices=False)
        r = min(r, Vt.shape[0])
        Y = torch.tensor(
            (S[:r, np.newaxis] * Vt[:r, :]).T, device=Y.device, dtype=Y.dtype
        )

    # Compute the sparse coefficient matrix C using ADMM
    C, _ = admm_main(
        Y,
        affine=affine,
        alpha=alpha,
        q=q,
        thr=thr,
        maxIter=max_iterations,
        verbose=verbose,
    )
    C = C.double()

    # Select representatives based on sparsity structure in C
    selected_indices = find_representatives(C, threshold_selection, q)
    representative_indices = remove_representatives(
        selected_indices, Y, threshold_pruning
    )

    return representative_indices, C
