import warnings
from collections import defaultdict

import pandas as pd
import torch
from torch.nn.utils import parametrize
from tqdm.auto import tqdm


@parametrize.cached()
def compute_feature_importance(
    model,
    dataloader,
    n_permutations=5,
    n_pairs=4096,
    verbose=True,
    warning_threshold=0.05,
):
    model.eval()
    all_importances = defaultdict(list)
    scores = []
    feature_names = dataloader.feature_names
    pbar = tqdm(
        total=len(feature_names) * n_permutations,
        desc=f"Computing feature importance on batches of size {n_pairs}",
        disable=not verbose,
    )
    with pbar:
        for i, f in enumerate(feature_names):
            for _ in range(n_permutations):
                X_batch, Y_batch = dataloader.sample(n_pairs)
                score = model.spearman(model(X_batch), Y_batch).item()
                scores.append(score)
                X_batch[:, i] = X_batch[
                    torch.randperm(X_batch.shape[0], device=X_batch.device),
                    i,
                ]
                score_permuted = model.spearman(model(X_batch), Y_batch).item()
                all_importances[f].append(score - score_permuted)
                pbar.update(1)

    all_importances = pd.DataFrame(all_importances)
    scores = pd.Series(scores, name="spearman")

    if scores.std() > warning_threshold:
        warnings.warn(
            f"Warning: High score variability between batches (std={scores.std():.4f} > warning threshold={warning_threshold}). "
            "Consider decreasing `threshold`, increasing `starting_n_pairs`, `batch_size_increase_factor` or directly `n_pairs`.",
            UserWarning,
        )

    return all_importances, scores
