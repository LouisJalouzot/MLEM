import pandas as pd
import torch
from tqdm.auto import tqdm

from .pairwise_dataloader import PairwiseDataloader


def train(
    model,
    X,
    Y,
    lr=0.1,
    weight_decay: float = 0.0,
    max_epochs=500,
    patience=50,
    n_pairs=4096,
    device="cpu",
):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        maximize=model.maximize,
        weight_decay=weight_decay,
    )

    dataloader = PairwiseDataloader(X, Y, n_pairs=n_pairs)

    best_score = -torch.inf if model.maximize else torch.inf
    epochs_without_improvement = 0

    pbar = tqdm(range(1, max_epochs + 1), desc="Training")
    for epoch in pbar:
        for X_pairs, true_dist in dataloader:
            X_pairs, true_dist = X_pairs.to(device), true_dist.to(device)

            optimizer.zero_grad()
            pred_dist = model(X_pairs)
            loss = model.loss(pred_dist, true_dist)
            loss.backward()
            optimizer.step()

        # Evaluation at the end of epoch
        model.eval()
        with torch.no_grad():
            X_pairs, true_dist = dataloader.sample()
            X_pairs, true_dist = X_pairs.to(device), true_dist.to(device)
            pred_dist = model(X_pairs)
            score = model.loss(pred_dist, true_dist).item()

        pbar.set_postfix({"loss": score})

        if (model.maximize and score > best_score) or (
            not model.maximize and score < best_score
        ):
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        model.train()

    return model
