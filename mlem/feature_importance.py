import numpy as np
import torch
from tqdm.auto import tqdm


def compute_feature_importance(model, X, Y, n_permutations=5, device="cpu"):
    model.eval()

    original_score = _get_score(model, X, Y, device)

    importances = {}

    pbar = tqdm(X.shape[1] * n_permutations, desc="Computing feature importance")
    with pbar:
        for i in range(X.shape[1]):
            permuted_scores = []
            for _ in range(n_permutations):
                X_permuted = X.clone()
                X_permuted[:, i] = X_permuted[torch.randperm(X.shape[0]), i]

                score = _get_score(model, X_permuted, Y, device)
                permuted_scores.append(score)

            importances[i] = original_score - np.mean(permuted_scores)
            pbar.update(1)

    return importances


def _get_score(model, X, Y, device):
    from .pairwise_dataloader import PairwiseDataloader

    dataloader = PairwiseDataloader(X, Y)

    scores = []
    with torch.no_grad():
        for X_pairs, true_dist in dataloader:
            X_pairs, true_dist = X_pairs.to(device), true_dist.to(device)
            pred_dist = model(X_pairs)
            scores.append(model.loss(pred_dist, true_dist).item())

    return np.mean(scores)
