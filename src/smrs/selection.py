import torch

def find_representatives(C: torch.Tensor, thr: float = 0.99, q: int | float = 2) -> torch.Tensor:
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


def remove_representatives(sInd: torch.Tensor, Y: torch.Tensor, thr: float=0.95) -> list[int]:
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
