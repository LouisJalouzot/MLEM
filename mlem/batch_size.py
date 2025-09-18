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
    batch_size_min: int = 256,
    n_trials: int = 64,
    threshold: float = 0.01,
    factor: float = 1.2,
    batch_size_max: int = 2**20,
    corr_warning_threshold: float = 0.5,
    verbose: bool = True,
) -> int:
    """Estimate a minimal batch size for training.

    It samples `n_trials` batches of feature distances on `batch_size` pairs of stimuli,
    and computes the correlation between all pairs of features along the stimuli
    dimension. If the standard deviation of these correlations across trials is below
    `threshold`, the current `batch_size` is returned. Otherwise, it multiplies the number
    of pairs by `factor` and repeats the process.

    Args:
        dataloader (PairwiseDataloader): The dataloader to use for feature distances.
        batch_size_min (int, default=256): The initial number of pairs to sample.
        n_trials (int, default=64): The number of trials to estimate the standard
            deviation of correlations.
        threshold (float, default=0.01): The threshold for stopping the search.
        factor (float, default=1.2): The factor by which to increase the number of pairs.
        batch_size_max (int, default=2**20): The maximum number of pairs to sample.
        corr_warning_threshold (float, default=0.5): A warning is issued if
                the absolute correlation between any pair of features exceeds this
                threshold.
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

    batch_size = batch_size_min
    pbar = tqdm(
        total=int(np.log(batch_size_max / batch_size) / np.log(factor)),
        desc=f"Estimating batch size on {dataloader.device}",
        disable=not verbose,
    )
    with pbar:
        while batch_size < batch_size_max:
            # (n_trials, batch_size, n_features)
            X_batch = dataloader.sample(batch_size, n_trials)
            # (n_trials, batch_size)
            corrs = batch_corrcoef(X_batch[:, :, i], X_batch[:, :, j], dim=1)  # type: ignore

            var = corrs.std(dim=0).max().item()
            pbar.set_postfix(
                {"Batch size": batch_size, "max std": var, "threshold": threshold}
            )
            if var < threshold:
                pbar.total = pbar.n
                pbar.refresh()
                break
            batch_size = int(batch_size * factor)
            pbar.update(1)

    max_corr = corrs.mean(dim=0).abs().max()
    if max_corr > corr_warning_threshold:
        warnings.warn(
            f"Largest absolute correlation between features is {max_corr:.2g} > warning "
            f"threshold of {corr_warning_threshold:.2g}. Be cautious when interpreting "
            "feature importance. Consider balancing better the features in the dataset.",
            UserWarning,
        )

    if batch_size >= batch_size_max:
        warnings.warn(
            f"Max number of pairs ({batch_size_max:2g}) reached during batch size estimation. "
            f"Returning current number of pairs: {batch_size:2g}.",
            UserWarning,
        )
    elif verbose:
        print(
            f"Batch size: {batch_size} sufficient (max std: {var:.2g} < threshold: {threshold})"
        )

    return batch_size
