from collections import defaultdict

import pandas as pd
import torch
from tqdm.auto import tqdm


def compute_feature_importance(
    model, dataloader, n_permutations=5, n_pairs=4096, verbose=True
):
    model.eval()
    all_importances = defaultdict(list)
    all_scores = defaultdict(list)
    pbar = tqdm(
        dataloader.dataset.X.shape[1] * n_permutations,
        desc="Computing feature importance",
        disable=not verbose,
    )
    with pbar:
        for i, f in enumerate(dataloader.feature_names):
            for _ in range(n_permutations):
                X_batch, Y_batch = dataloader.sample(n_pairs)
                score = model.spearman(model(X_batch), Y_batch).item()
                all_scores[f].append(score)
                X_batch[:, i] = X_batch[torch.randperm(X_batch.shape[0]), i]
                score_permuted = model.spearman(model(X_batch), Y_batch).item()
                all_importances[f].append(score - score_permuted)
                pbar.update(1)

    return pd.DataFrame(all_importances), pd.DataFrame(all_scores)
