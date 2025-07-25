# Torch imports
import torch


def compute_lambda(Y):
    """
    Computes the regularization parameter lambda for the L1/Lq minimization.

    Parameters:
    - Y: torch tensor of shape (D, N), the data matrix with N data points in D dimensions.

    Returns:
    - lambda_param: regularization parameter for the optimization problem.
    """

    Y = Y.double()
    _, N = Y.shape
    T = torch.zeros(N, device=Y.device, dtype=Y.dtype)

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
    - Z1: torch tensor of shape (N, N)
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
    elif tau.dtype != torch.float64:
        tau = tau.double()

    # Sort y by absolute values in descending order
    y_abs = torch.abs(y)
    y_sorted, indices_sorted = torch.sort(y_abs, descending=True)

    # Handle the trivial case of a single-element tensor
    if len(y) <= 1:
        zbar = y_sorted[0]
        value = torch.maximum(
            zbar - tau, torch.tensor(0.0, dtype=torch.float64, device=y.device)
        )
        x[0] = torch.sign(y[0]) * value
        return x

    # Calculate cumulative sum for threshold check
    arange_tensor = torch.arange(1, len(y), device=y.device, dtype=torch.float64)
    cumulative_sum = (torch.cumsum(y_sorted[:-1], dim=0) / arange_tensor) - (
        tau / arange_tensor
    )

    # Find the cutoff index
    d = cumulative_sum > y_sorted[1:]
    if not torch.any(d):
        cutoff_index = len(y)
    else:
        cutoff_index = torch.where(d)[0][0].item() + 1

    # Calculate the mean of the absolute values up to the cutoff
    zbar = torch.mean(y_sorted[:cutoff_index])

    # Compute the shrinkage threshold
    if cutoff_index < len(y):
        # Compare with the next largest absolute value
        threshold = y_sorted[cutoff_index]
        value = torch.maximum(zbar - tau / cutoff_index, threshold)
    else:
        # Compare with zero
        value = torch.maximum(
            zbar - tau / cutoff_index,
            torch.tensor(0.0, dtype=torch.float64, device=y.device),
        )

    # Apply the shrinkage to the first part of the vector
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


def admm_main(
    Y, alpha=5, q=2, thr=1e-7, maxIter=5000, verbose=True, logging: bool = False
):
    """
    ADMM for finding sparse representation with or without affine constraints.

    Parameters:
    - Y: DxN data matrix of N data points in D-dimensional space (torch tensor).
    - alpha: regularization parameter.
    - q: norm for L1/Lq minimization.
    - thr: stopping threshold for coefficient error ||Z - C||.
    - maxIter: maximum number of ADMM iterations.
    - verbose: bool, if True, print iteration errors.
    - logging: bool, if True, track and return convergence history

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

    if logging:
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

        if logging:
            dual_res = rho * calculate_errorcoefficient(Z_current, Z_previous)

            logs["primal_residual"].append(err1.item())
            logs["dual_residual"].append(dual_res.item())
            logs["affine_constraint_error"].append(err2.item())

        Z_previous = Z_current
        i += 1

        if verbose and i % 100 == 0:
            print(
                f"Iteration {i}, || Z - C || = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}"
            )

    Err = (err1, err2)
    if verbose:
        print("-" * 80)
        print(
            f"Terminating ADMM at iteration {i:5d}, \n ||Z - C|| = {err1:.5e}, ||1 - C^T 1|| = {err2:.5e}."
        )
        print("-" * 80)

    if logging:
        return Z_current, Err, logs
    else:
        return Z_current, Err


def find_representatives(C, thr=0.99, q=2):
    """
    Identifies indices of nonzero rows in the coefficient matrix based on their norms.

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
    Y, alpha=5, r=0, verbose=True, max_iterations=5000, logging=False
):
    """
    Sparse Modeling Representative Selection (SMRS) function.

    Parameters:
    - Y: DxN data matrix of N data points in D-dimensional space (torch tensor).
    - alpha: regularization parameter, typically in [2, 50].
    - verbose: if True, prints information during iterations.
    - max_iterations: maximum number of ADMM iterations.
    - logging: enables logging for convergence testing.

    Returns:
    - representative_indices: indices of selected representative points.
    - C: NxN sparse coefficient matrix.
    """
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
    if logging:
        C, _, logs = admm_main(
            Y,
            alpha=alpha,
            q=q,
            thr=thr,
            maxIter=max_iterations,
            verbose=verbose,
            logging=logging,
        )
    else:
        C, _ = admm_main(
            Y,
            alpha=alpha,
            q=q,
            thr=thr,
            maxIter=max_iterations,
            verbose=verbose,
            logging=logging,
        )
    C = C.double()

    # Select representatives based on sparsity structure in C
    selected_indices = find_representatives(C, threshold_selection, q)
    representative_indices = remove_representatives(
        selected_indices, Y, threshold_pruning
    )

    if logging:
        return representative_indices, C, logs
    else:
        return representative_indices, C
