import typing as tp
import warnings
from collections import defaultdict

import pandas as pd
import torch
from torch.nn.utils import parametrize
from tqdm.auto import tqdm

from .batch_size import batch_corrcoef
from .model import Model
from .pairwise_dataloader import PairwiseDataloader


@torch.no_grad()
def batch_spearman(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes batched Spearman rank correlation coefficient.

    Args:
        x (torch.Tensor): The first input tensor.
        y (torch.Tensor): The second input tensor.
        dim (int, default=-1): The dimension along which to compute the correlation.

    Returns:
        torch.Tensor: The batched Spearman correlation coefficients.
    """
    dtype = x.dtype
    x_rank = x.argsort(dim=dim).argsort(dim=dim).to(dtype)
    y_rank = y.argsort(dim=dim).argsort(dim=dim).to(dtype)

    return batch_corrcoef(x_rank, y_rank, dim=dim)


@torch.no_grad()
@parametrize.cached()
def compute_feature_importance(
    model: Model,
    dataloader: PairwiseDataloader,
    n_permutations: int = 5,
    batch_size: int = 256,
    verbose: bool = True,
    warning_threshold: float = 0.05,
    memory: str = "medium",
    rng: tp.Optional[torch.Generator] = None,
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
        model (Model): The trained model.
        dataloader (PairwiseDataloader): The dataloader for sampling data.
        n_permutations (int, default=5): The number of permutations for each feature.
        batch_size (int, default=256): The number of pairs to sample in each batch.
        verbose (bool, default=True): If True, displays a progress bar.
        warning_threshold (float, default=0.05): The threshold for score variability
            to trigger a warning.
        memory (str, default='medium'): The memory usage profile
            ('low', 'medium', 'high').
        rng (torch.Generator | None, default=None): A random number generator for
            reproducible permutations.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing a DataFrame of feature
            importances and a Series of baseline scores across permutations.
    """
    model.eval()
    feature_names = dataloader.feature_names
    n_features = len(feature_names)

    if memory == "low":  # iterate over features and permutations
        all_importances = defaultdict(list)
        baseline_scores = []
        pbar = tqdm(
            total=n_features * n_permutations,
            desc=f"Computing feature importance with batches of size {batch_size} on {dataloader.device}",
            disable=not verbose,
        )
        with pbar:
            for i, f in enumerate(feature_names):
                for _ in range(n_permutations):
                    X_batch, Y_batch = dataloader.sample(batch_size)
                    baseline_score = model.spearman(model(X_batch), Y_batch).item()
                    baseline_scores.append(baseline_score)
                    perm = torch.randperm(
                        batch_size, device=X_batch.device, generator=rng
                    )
                    X_batch[:, i] = X_batch[perm, i]
                    permuted_score = model.spearman(model(X_batch), Y_batch).item()
                    all_importances[f].append(baseline_score - permuted_score)
                    pbar.update(1)
        all_importances = pd.DataFrame(all_importances)
        baseline_scores = pd.Series(baseline_scores, name="spearman")

    elif memory == "medium":
        all_importances = defaultdict(list)
        baseline_scores = []
        pbar = tqdm(
            total=n_features,
            desc=f"Computing feature importance with batches of size {batch_size} on {dataloader.device}",
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
                baseline_score = batch_spearman(model(X_batch), Y_batch, dim=1)
                baseline_scores.extend(baseline_score.cpu().numpy())

                # Generate batched permutations on the batch_size dimension efficiently
                # from https://discuss.pytorch.org/t/batch-version-of-torch-randperm/111121/3
                # (n_permutations, batch_size)
                batched_perms = torch.rand(
                    n_permutations, batch_size, device=X_batch.device, generator=rng
                ).argsort(dim=1)

                # Permute feature i in place in a batched way
                X_batch[:, :, i] = X_batch[:, :, i].gather(dim=1, index=batched_perms)

                # Compute permuted scores
                # (n_permutations,)
                permuted_score = batch_spearman(model(X_batch), Y_batch, dim=1)

                all_importances[f].extend(
                    (baseline_score - permuted_score).cpu().numpy()
                )
                pbar.update(1)
        all_importances = pd.DataFrame(all_importances)
        baseline_scores = pd.Series(baseline_scores, name="spearman")

    else:  # memory == "high",
        X_batch, Y_batch = dataloader.sample(
            batch_size, n_trials=n_features * n_permutations
        )
        X_batch = X_batch.reshape(n_features, n_permutations, batch_size, n_features)
        Y_batch = Y_batch.reshape(n_features, n_permutations, batch_size)

        # Compute baseline scores
        # (n_features, n_permutations)
        baseline_scores = batch_spearman(model(X_batch), Y_batch, dim=2)

        # Generate batched permutations on the batch_size dimension efficiently
        # from https://discuss.pytorch.org/t/batch-version-of-torch-randperm/111121/3
        # (n_features, n_permutations, batch_size)
        batched_perms = torch.rand(
            n_features, n_permutations, batch_size, device=X_batch.device, generator=rng
        ).argsort(dim=2)

        # Permute in-place in a batched way on 2 dimensions
        # batched_perms[f] will be used to permute X_batch[f, :, :, f] in a batched way

        # Get a diagonal view of the data to permute
        # (n_permutations, batch_size, n_features)
        diag_view = X_batch.diagonal(dim1=0, dim2=3, offset=0)
        # (n_features, n_permutations, batch_size)
        diag_view = diag_view.permute(2, 0, 1)
        # Permute along dimension batch_size
        diag_view_permuted = diag_view.gather(dim=2, index=batched_perms)
        # Copy in original tensor X_batch
        diag_view.copy_(diag_view_permuted)

        # Compute permuted scores
        # (n_features, n_permutations)
        permuted_scores = batch_spearman(model(X_batch), Y_batch, dim=2)

        # (n_features, n_permutations)
        all_importances = baseline_scores - permuted_scores
        # (n_permutations, n_features)
        all_importances = all_importances.T.cpu().numpy()
        all_importances = pd.DataFrame(all_importances, columns=feature_names)
        # (n_permutations * n_features,)
        baseline_scores = baseline_scores.flatten().cpu().numpy()
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
