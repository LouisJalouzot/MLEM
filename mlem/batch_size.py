import warnings

import numpy as np
import torch
from tqdm.auto import tqdm

from .pairwise_dataloader import PairwiseDataloader


def batch_corrcoef(
    x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """Compute the Pearson correlation coefficient between two batches of vectors.

    Args:
        x (torch.Tensor): The first input tensor.
        y (torch.Tensor): The second input tensor.
        dim (int, default=-1): The dimension along which to compute the correlation.
        eps (float, default=1e-8): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The batched Pearson correlation coefficients.
    """
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    covariance = (x_centered * y_centered).sum(dim=dim)

    x_std = torch.sqrt((x_centered**2).sum(dim=dim)) + eps
    y_std = torch.sqrt((y_centered**2).sum(dim=dim)) + eps

    correlation = covariance / (x_std * y_std)

    return correlation


def estimate_batch_size(
    dataloader: PairwiseDataloader,
    starting_n_pairs: int = 4096,
    n_trials: int = 64,
    threshold: float = 0.01,
    factor: float = 1.2,
    max_n_pairs: int = 2**20,
    verbose: bool = True,
) -> int:
    """Estimate a minimal batch size for training.

    It samples `n_trials` batches of feature distances on `n_pairs` pairs of stimuli,
    and computes the correlation between all pairs of features along the stimuli dimension.
    If the standard deviation of these correlations across trials is below `threshold`,
    the current `n_pairs` is returned. Otherwise, it multiplies the number of pairs by
    `factor` and repeats the process.

    Args:
        dataloader (PairwiseDataloader): The dataloader to use for feature distances.
        starting_n_pairs (int, default=4096): The initial number of pairs to sample.
        n_trials (int, default=64): The number of trials to estimate the standard
            deviation of correlations.
        threshold (float, default=0.01): The threshold for stopping the search.
        factor (float, default=1.2): The factor by which to increase the number of pairs.
        max_n_pairs (int, default=2**20): The maximum number of pairs to sample.
        verbose (bool, default=True): If True, displays a progress bar.

    Returns:
        int: Estimated batch size.
    """
    assert factor > 1, "factor should be greater than 1"

    if dataloader.interactions:
        n = dataloader.n_features * (dataloader.n_features - 1) // 2
    else:
        n = dataloader.n_features
    i, j = torch.triu_indices(n, n, 1)

    n_pairs = starting_n_pairs
    pbar = tqdm(
        total=int(np.log(max_n_pairs / n_pairs) / np.log(factor)),
        desc="Estimating batch size",
        disable=not verbose,
    )
    with pbar:
        while n_pairs < max_n_pairs:
            # (n_trials, n_pairs, n_features)
            X_batch = dataloader.sample(n_pairs, n_trials)
            # (n_trials, n_pairs)
            corrs = batch_corrcoef(X_batch[:, :, i], X_batch[:, :, j], dim=1)  # type: ignore

            var = corrs.std(dim=0).max().item()
            pbar.set_postfix(
                {"Batch size": n_pairs, "max std": var, "threshold": threshold}
            )
            if var < threshold:
                if verbose:
                    print(
                        f"Batch size: {n_pairs} sufficient (max std: {var:.2g} < threshold: {threshold})"
                    )
                pbar.total = pbar.n
                pbar.refresh()
                break
            n_pairs = int(n_pairs * factor)
            pbar.update(1)

    if n_pairs >= max_n_pairs:
        warnings.warn(
            f"Max number of pairs ({max_n_pairs:2g}) reached during batch size estimation. "
            f"Returning current number of pairs: {n_pairs:2g}.",
            UserWarning,
        )

    return n_pairs
