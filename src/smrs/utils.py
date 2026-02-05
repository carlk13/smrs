import torch


def compute_lambda(Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the regularization parameter lambda for the L1/Lq minimization.

    Args:
        Y (torch.Tensor): Data matrix of shape (D, N).

    Returns:
        torch.Tensor: The computed scalar lambda parameter.
    """
    Y = Y.double()
    _, N = Y.shape
    T = torch.zeros(N, device=Y.device, dtype=Y.dtype)

    y_mean = torch.mean(Y, dim=1, keepdim=True).to(Y.dtype)
    ones_matrix = torch.ones((1, N), device=Y.device, dtype=Y.dtype)

    for i in range(N):
        yi = Y[:, i].unsqueeze(0)
        affine_term = y_mean @ ones_matrix - Y
        T[i] = torch.norm(torch.matmul(yi, affine_term))

    lambda_param = torch.max(T)
    return lambda_param


def shrink_l1_lq(
    Z1: torch.Tensor, lambda_param: float | torch.Tensor, q: int | float = 2
) -> torch.Tensor:
    """
    Applies L1/Lq shrinkage for sparsity.

    Args:
        Z1 (torch.Tensor): Input matrix of shape (N, N).
        lambda_param (float | torch.Tensor): Regularization parameter.
        q (int | float): Norm type (1, 2, or float('inf')).

    Returns:
        torch.Tensor: The shrunk matrix Z2.
    """
    Z1 = Z1.double()

    if not isinstance(lambda_param, torch.Tensor):
        lambda_param = torch.tensor(lambda_param, dtype=Z1.dtype, device=Z1.device)
    else:
        lambda_param = lambda_param.double()

    if q == 1:
        Z2 = torch.maximum(
            torch.abs(Z1) - lambda_param, torch.zeros_like(Z1)
        ) * torch.sign(Z1)
    elif q == 2:
        row_norms = torch.norm(Z1, dim=1)
        r = torch.maximum(row_norms - lambda_param, torch.zeros_like(row_norms))
        # Avoid division by zero
        r = r / (r + lambda_param)
        Z2 = r.unsqueeze(1) * Z1
    elif q == float("inf"):
        Z2 = torch.stack([shrink_l2_linf(row, lambda_param) for row in Z1])
    else:
        raise ValueError("Unsupported norm type. Use q=1, 2, or float('inf')")

    return Z2


def shrink_l2_linf(y: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """
    Minimizes 0.5 * ||x - y||_2^2 + tau * ||x||_inf.
    """
    y = y.double()
    x = y.clone()

    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, dtype=torch.float64, device=y.device)
    elif tau.dtype != torch.float64:
        tau = tau.double()

    y_abs = torch.abs(y)
    y_sorted, indices_sorted = torch.sort(y_abs, descending=True)

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


def calculate_errorcoefficient(Z: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Compute the normalized average absolute error between matrices Z and C."""
    Z = Z.double()
    C = C.double()

    if Z.shape != C.shape:
        raise ValueError(
            f"Z and C must have the same shape. Got {Z.shape} and {C.shape}"
        )

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
