import warnings
from collections import defaultdict

import pandas as pd
import torch
from torch.nn.utils import parametrize
from tqdm.auto import tqdm

from .batch_size import batch_corrcoef


@torch.no_grad()
def batch_spearman(x, y, dim=-1):
    dtype = x.dtype
    x_rank = x.argsort(dim=dim).argsort(dim=dim).to(dtype)
    y_rank = y.argsort(dim=dim).argsort(dim=dim).to(dtype)

    return batch_corrcoef(x_rank, y_rank, dim=dim)


@torch.no_grad()
@parametrize.cached()
def compute_feature_importance(
    model,
    dataloader,
    n_permutations=5,
    n_pairs=4096,
    verbose=True,
    warning_threshold=0.05,
    low_memory=False,
):
    model.eval()
    all_importances = defaultdict(list)
    baseline_scores = []
    feature_names = dataloader.feature_names
    n_features = len(feature_names)

    if low_memory:
        pbar = tqdm(
            total=n_features * n_permutations,
            desc=f"Computing feature importance on batches of size {n_pairs}",
            disable=not verbose,
        )
        with pbar:
            for i, f in enumerate(feature_names):
                for _ in range(n_permutations):
                    X_batch, Y_batch = dataloader.sample(n_pairs)
                    score = model.spearman(model(X_batch), Y_batch).item()
                    baseline_scores.append(score)
                    X_batch[:, i] = X_batch[
                        torch.randperm(X_batch.shape[0], device=X_batch.device),
                        i,
                    ]
                    score_permuted = model.spearman(model(X_batch), Y_batch).item()
                    all_importances[f].append(score - score_permuted)
                    pbar.update(1)
        all_importances = pd.DataFrame(all_importances)
        baseline_scores = pd.Series(baseline_scores, name="spearman")
    else:
        n_total_trials = n_features * n_permutations
        X_batch, Y_batch = dataloader.sample(n_pairs, n_trials=n_total_trials)
        # (n_features, n_permutations, n_pairs, n_features)
        X_batch = X_batch.reshape(n_features, n_permutations, n_pairs, n_features)
        # (n_features, n_permutations, n_pairs)
        Y_batch = Y_batch.reshape(n_features, n_permutations, n_pairs)

        # Compute baseline scores
        # (n_features, n_permutations)
        baseline_scores = batch_spearman(model(X_batch), Y_batch, dim=2)

        # Generate batched permutations on the n_pairs dimension efficiently
        # from https://discuss.pytorch.org/t/batch-version-of-torch-randperm/111121/3
        # (n_features, n_permutations, n_pairs)
        batched_perms = torch.rand(
            n_features, n_permutations, n_pairs, device=X_batch.device
        ).argsort(dim=2)

        # Perform in-place batched column permutations
        # batched_perms[f] will be used to permute X_batch[f, :, :, f] in a batched way

        # Get a diagonal view of the data to permute
        # (n_permutations, n_pairs, n_features)
        diag_view = X_batch.diagonal(dim1=0, dim2=3, offset=0)
        # (n_features, n_permutations, n_pairs)
        diag_view = diag_view.permute(2, 0, 1)
        # Permute along dimension n_pairs and copy in original tensor X_batch
        diag_view.copy_(diag_view.gather(dim=2, index=batched_perms))

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
            f"Warning: High score variability between batches (std={var} > warning threshold={warning_threshold}). "
            "Consider decreasing `threshold`, increasing `starting_n_pairs`, `batch_size_increase_factor` or directly `n_pairs`.",
            UserWarning,
        )

    return all_importances, baseline_scores
