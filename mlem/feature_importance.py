import typing as tp
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from .batch_size import batch_corrcoef
from .model import Model
from .pairwise_dataloader import PairwiseDataloader


def batch_spearman(x: jnp.ndarray, y: jnp.ndarray, dim: int = -1) -> jnp.ndarray:
    """Computes batched Spearman rank correlation coefficient.

    Args:
        x: The first input array.
        y: The second input array.
        dim: The dimension along which to compute the correlation.

    Returns:
        The batched Spearman correlation coefficients.
    """
    dtype = x.dtype
    x_rank = jnp.argsort(jnp.argsort(x, axis=dim), axis=dim).astype(dtype)
    y_rank = jnp.argsort(jnp.argsort(y, axis=dim), axis=dim).astype(dtype)

    return batch_corrcoef(x_rank, y_rank, dim=dim)


def compute_feature_importance(
    model: Model,
    dataloader: PairwiseDataloader,
    n_permutations: int = 5,
    batch_size: int = 256,
    verbose: bool = True,
    warning_threshold: float = 0.05,
    memory: str = "medium",
    rng: tp.Optional[jax.random.PRNGKey] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Computes permutation feature importances: for each feature, it samples a batch of
    data of size `batch_size`, computes the baseline score, permutes the feature, and
    computes the drop in score which the importance of the feature. This is repeated for
    `n_permutations` different permutations per feature. The `memory` parameter controls
    the memory usage profile:
    - 'low': iterates over features and permutations, using minimal memory but is slow.
    - 'medium': iterates over features but computes all permutations in a batch,
      balancing speed and memory usage.
    - 'high': computes all features and permutations in a single batch, using the most
        memory but is the fastest.

    Args:
        model: The trained model.
        dataloader: The dataloader for sampling data.
        n_permutations: The number of permutations for each feature.
        batch_size: The number of pairs to sample in each batch.
        verbose: If True, displays a progress bar.
        warning_threshold: The threshold for score variability to trigger a warning.
        memory: The memory usage profile ('low', 'medium', 'high').
        rng: A JAX random key for reproducible permutations.

    Returns:
        A tuple containing a DataFrame of feature importances and a Series of 
        baseline scores across permutations.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    feature_names = dataloader.feature_names
    n_features = len(feature_names)

    if memory == "low":  # iterate over features and permutations
        all_importances = defaultdict(list)
        baseline_scores = []
        pbar = tqdm(
            total=n_features * n_permutations,
            desc=f"Computing feature importance with batches of size {batch_size}",
            disable=not verbose,
        )
        with pbar:
            for i, f in enumerate(feature_names):
                for _ in range(n_permutations):
                    X_batch, Y_batch = dataloader.sample(batch_size)
                    baseline_score = float(model.spearman(model.forward(X_batch), Y_batch))
                    baseline_scores.append(baseline_score)
                    rng, perm_rng = jax.random.split(rng)
                    perm = jax.random.permutation(perm_rng, batch_size)
                    X_batch = X_batch.at[:, i].set(X_batch[perm, i])
                    permuted_score = float(model.spearman(model.forward(X_batch), Y_batch))
                    all_importances[f].append(baseline_score - permuted_score)
                    pbar.update(1)
        all_importances = pd.DataFrame(all_importances)
        baseline_scores = pd.Series(baseline_scores, name="spearman")

    elif memory == "medium":
        all_importances = defaultdict(list)
        baseline_scores = []
        pbar = tqdm(
            total=n_features,
            desc=f"Computing feature importance with batches of size {batch_size}",
            disable=not verbose,
        )
        with pbar:
            for i, f in enumerate(feature_names):
                # (n_permutations, batch_size, n_features)
                # (n_permutations, batch_size)
                X_batch, Y_batch = dataloader.sample(
                    batch_size, n_trials=n_permutations
                )

                # Compute baseline scores
                # (n_permutations,)
                baseline_score = batch_spearman(
                    jax.vmap(model.forward)(X_batch), Y_batch, dim=1
                )
                baseline_scores.extend(np.array(baseline_score))

                # Generate batched permutations on the batch_size dimension efficiently
                # (n_permutations, batch_size)
                rng, perm_rng = jax.random.split(rng)
                batched_perms = jax.random.uniform(
                    perm_rng, (n_permutations, batch_size)
                ).argsort(axis=1)

                # Permute feature i in place in a batched way
                X_batch = X_batch.at[:, :, i].set(
                    jnp.take_along_axis(X_batch[:, :, i], batched_perms, axis=1)
                )

                # Compute permuted scores
                # (n_permutations,)
                permuted_score = batch_spearman(
                    jax.vmap(model.forward)(X_batch), Y_batch, dim=1
                )

                all_importances[f].extend(
                    np.array(baseline_score - permuted_score)
                )
                pbar.update(1)
        all_importances = pd.DataFrame(all_importances)
        baseline_scores = pd.Series(baseline_scores, name="spearman")

    else:  # memory == "high"
        X_batch, Y_batch = dataloader.sample(
            batch_size, n_trials=n_features * n_permutations
        )
        X_batch = X_batch.reshape(n_features, n_permutations, batch_size, n_features)
        Y_batch = Y_batch.reshape(n_features, n_permutations, batch_size)

        # Compute baseline scores
        # (n_features, n_permutations)
        baseline_scores = jax.vmap(
            lambda x, y: batch_spearman(jax.vmap(model.forward)(x), y, dim=1)
        )(X_batch, Y_batch)

        # Generate batched permutations on the batch_size dimension efficiently
        # (n_features, n_permutations, batch_size)
        rng, perm_rng = jax.random.split(rng)
        batched_perms = jax.random.uniform(
            perm_rng, (n_features, n_permutations, batch_size)
        ).argsort(axis=2)

        # Permute in-place in a batched way on 2 dimensions
        # Get a diagonal view of the data to permute
        # (n_permutations, batch_size, n_features)
        diag_view = jnp.diagonal(X_batch, axis1=0, axis2=3, offset=0)
        # (n_features, n_permutations, batch_size)
        diag_view = jnp.transpose(diag_view, (2, 0, 1))
        # Permute along dimension batch_size
        diag_view_permuted = jnp.take_along_axis(diag_view, batched_perms, axis=2)
        
        # Create new X_batch with permuted diagonal
        # This is tricky in JAX since we need to modify the diagonal
        # We'll use a different approach: permute each feature independently
        def permute_feature(f_idx, X_batch_f):
            # X_batch_f is (n_permutations, batch_size, n_features)
            # permute the f_idx-th feature
            perms = batched_perms[f_idx]  # (n_permutations, batch_size)
            X_batch_f = X_batch_f.at[:, :, f_idx].set(
                jnp.take_along_axis(X_batch_f[:, :, f_idx], perms, axis=1)
            )
            return X_batch_f
        
        # Apply permutation to each feature
        X_batch_permuted = []
        for f_idx in range(n_features):
            X_f = X_batch[f_idx]  # (n_permutations, batch_size, n_features)
            X_f_perm = permute_feature(f_idx, X_f)
            X_batch_permuted.append(X_f_perm)
        X_batch = jnp.stack(X_batch_permuted, axis=0)

        # Compute permuted scores
        # (n_features, n_permutations)
        permuted_scores = jax.vmap(
            lambda x, y: batch_spearman(jax.vmap(model.forward)(x), y, dim=1)
        )(X_batch, Y_batch)

        # (n_features, n_permutations)
        all_importances = baseline_scores - permuted_scores
        # (n_permutations, n_features)
        all_importances = np.array(all_importances.T)
        all_importances = pd.DataFrame(all_importances, columns=feature_names)
        # (n_permutations * n_features,)
        baseline_scores = np.array(baseline_scores.flatten())
        baseline_scores = pd.Series(baseline_scores, name="spearman")

    var = baseline_scores.std()
    if var > warning_threshold:
        warnings.warn(
            f"Warning: There is a high variability in score between batches (std={var:.3f} > {warning_threshold:.3f}=warning threshold).\n"
            "   Consider decreasing `threshold` or increasing `batch_size_min` for a better estimation of `batch_size`.\n"
            "   Alternatively, you can manually increase `batch_size`.",
            UserWarning,
        )

    return all_importances, baseline_scores
