import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchsort import soft_rank


def log1pexp(x):
    # From https://gitlab.com/jbleger/parametrization-cookbook/-/blob/main/parametrization_cookbook/functions/torch.py#:~:text=def%20log1pexp(x,relu(x)
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.relu(x)


class SPDMatrixLearner(nn.Module):
    def __init__(
        self,
        n_features: int,
        interactions: bool = True,
        spearman_regularization: str = "l2",
        spearman_regularization_strength: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.interactions = interactions
        self.spearman_regularization = spearman_regularization
        self.spearman_regularization_strength = spearman_regularization_strength
        self.maximize = True

        if self.interactions:
            # Use a simple lower triangular matrix for Cholesky decomposition
            self.L = nn.Parameter(torch.randn(n_features, n_features).tril_())
        else:
            # Use a vector for diagonal elements
            self.diag_vec = nn.Parameter(torch.randn(n_features))

    def get_W(self) -> torch.Tensor:
        if self.interactions:
            # Ensure diagonal is positive
            L = self.L.clone()
            L.diagonal().copy_(log1pexp(L.diagonal()))
            W = L @ L.T
        else:
            # Ensure diagonal is positive
            W = torch.diag(log1pexp(self.diag_vec))

        return W / W.norm(p="fro")

    def get_formatted_W(self, features=None) -> pd.DataFrame:
        W = self.get_W().detach().cpu().numpy()
        if features is None:
            features = [f"Feature {i+1}" for i in range(self.n_features)]
        return pd.DataFrame(W, index=features, columns=features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        W = self.get_W()
        pX = (X[:, 0] - X[:, 1]).abs().clip(0, 1)
        pred_dist = (pX @ W * pX).sum(dim=1)
        return pred_dist

    def spearman(self, x, y):
        # Soft rank uses a regularization parameter, 'regularization_strength'
        x_rank = soft_rank(
            x.unsqueeze(0),
            regularization=self.spearman_regularization,
            regularization_strength=self.spearman_regularization_strength,
        )
        y_rank = soft_rank(
            y.unsqueeze(0),
            regularization=self.spearman_regularization,
            regularization_strength=self.spearman_regularization_strength,
        )

        # Center the ranks
        x_rank_mean = x_rank.mean()
        y_rank_mean = y_rank.mean()
        x_centered = x_rank - x_rank_mean
        y_centered = y_rank - y_rank_mean

        # Compute covariance and standard deviations
        cov = (x_centered * y_centered).mean()
        x_std = x_centered.std()
        y_std = y_centered.std()

        # Compute Spearman correlation
        corr = cov / (x_std * y_std + 1e-8)
        return corr

    def loss(self, pred_dist, true_dist):
        return self.spearman(pred_dist, true_dist)
