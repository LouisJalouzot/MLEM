import warnings

import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm

from .pairwise_dataloader import PairwiseDataloader


def batch_corrcoef(
    x: jnp.ndarray, y: jnp.ndarray, dim: int = -1, eps: float = 1e-8
) -> jnp.ndarray:
    """Compute the Pearson correlation coefficient between two batches of vectors.

    Args:
        x: The first input array.
        y: The second input array.
        dim: The dimension along which to compute the correlation.
        eps: A small value to avoid division by zero.

    Returns:
        The batched Pearson correlation coefficients.
    """
    x_mean = x.mean(axis=dim, keepdims=True)
    y_mean = y.mean(axis=dim, keepdims=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    covariance = (x_centered * y_centered).sum(axis=dim)

    x_std = jnp.sqrt((x_centered**2).sum(axis=dim)) + eps
    y_std = jnp.sqrt((y_centered**2).sum(axis=dim)) + eps

    correlation = covariance / (x_std * y_std)

    return correlation


def estimate_batch_size(
    dataloader: PairwiseDataloader,
    batch_size_min: int = 256,
    n_trials: int = 64,
    threshold: float = 0.01,
    factor: float = 1.2,
    batch_size_max: int = 2**20,
    verbose: bool = True,
) -> int:
    """Estimate a minimal batch size for training.

    It samples `n_trials` batches of feature distances on `batch_size` pairs of stimuli,
    and computes the correlation between all pairs of features along the stimuli
    dimension. If the standard deviation of these correlations across trials is below
    `threshold`, the current `batch_size` is returned. Otherwise, it multiplies the number
    of pairs by `factor` and repeats the process.

    Args:
        dataloader: The dataloader to use for feature distances.
        batch_size_min: The initial number of pairs to sample.
        n_trials: The number of trials to estimate the standard
            deviation of correlations.
        threshold: The threshold for stopping the search.
        factor: The factor by which to increase the number of pairs.
        batch_size_max: The maximum number of pairs to sample.
        verbose: If True, displays a progress bar.

    Returns:
        Estimated batch size.
    """
    assert factor > 1, "factor should be greater than 1"

    if dataloader.interactions:
        n = dataloader.n_features * (dataloader.n_features - 1) // 2
    else:
        n = dataloader.n_features
    i, j = jnp.triu_indices(n, k=1)

    batch_size = batch_size_min
    pbar = tqdm(
        total=int(np.log(batch_size_max / batch_size) / np.log(factor)),
        desc=f"Estimating batch size",
        disable=not verbose,
    )
    with pbar:
        while batch_size < batch_size_max:
            # (n_trials, batch_size, n_features)
            X_batch = dataloader.sample(batch_size, n_trials)
            # (n_trials, batch_size)
            corrs = batch_corrcoef(X_batch[:, :, i], X_batch[:, :, j], dim=1)

            var = float(corrs.std(axis=0).max())
            pbar.set_postfix(
                {"Batch size": batch_size, "max std": var, "threshold": threshold}
            )
            if var < threshold:
                pbar.total = pbar.n
                pbar.refresh()
                break
            batch_size = int(batch_size * factor)
            pbar.update(1)

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
