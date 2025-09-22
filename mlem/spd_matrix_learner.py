import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import parametrize
from torchsort import soft_rank


def log1pexp(x):
    # From https://gitlab.com/jbleger/parametrization-cookbook/-/blob/main/parametrization_cookbook/functions/torch.py#:~:text=def%20log1pexp(x,relu(x)
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.relu(x)


class CholeskySPD(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.triu_indices = torch.triu_indices(n_features, n_features)

    def forward(self, L):
        L_prime = L.tril()
        d = torch.diagonal(L_prime)
        d_positive = log1pexp(d)
        L_prime.diagonal().copy_(d_positive)
        W = L_prime @ L_prime.T

        # Keep to Frobenius norm 1
        W = W / torch.norm(W, p="fro")

        # Get the upper triangle with off-diagonal elements doubled and flatten it
        # (n_features, n_features) -> (n_features * (n_features + 1) / 2,)
        W_triu = torch.triu(W) + torch.triu(W, diagonal=1)
        W_flat = W_triu[*self.triu_indices]

        # (1, n_features * (n_features + 1) / 2)
        return W_flat[None]


class Positive(nn.Module):
    def forward(self, diag_vec):
        diag_vec_pos = log1pexp(diag_vec)
        # Keep to L2 norm 1
        return diag_vec_pos / diag_vec_pos.norm()


class SPDMatrixLearner(nn.Module):
    def __init__(
        self,
        n_features: int,
        interactions: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.interactions = interactions

        if self.interactions:
            self.W = nn.Linear(n_features, n_features, bias=False)
            parametrize.register_parametrization(
                self.W, "weight", CholeskySPD(n_features), unsafe=True
            )
        else:
            self.W = nn.Linear(n_features, 1, bias=False)
            parametrize.register_parametrization(self.W, "weight", Positive())

    def forward(self, X):
        return self.W(X).squeeze()

    def format_W(self, features=None):
        if features is None:
            features = [f"Feature {i+1}" for i in range(self.n_features)]
        W = self.W.weight.detach().cpu().squeeze().numpy()
        if self.interactions:
            W_square = np.full((self.n_features, self.n_features), float("nan"))
            W_square[*np.triu_indices(self.n_features)] = W
            return pd.DataFrame(W_square, index=features, columns=features)
        else:
            return pd.DataFrame(W, index=features, columns=["Weight"])

    def corrcoef(self, x, y):
        x = x.float()
        y = y.float()
        y_n = y - y.mean()
        x_n = x - x.mean()
        y_n = y_n / y_n.norm()
        x_n = x_n / x_n.norm()

        return (y_n * x_n).sum()

    @torch.no_grad()
    def spearman(self, x, y):
        dtype = x.dtype
        x_rank = x.argsort().argsort().to(dtype)
        y_rank = y.argsort().argsort().to(dtype)

        return self.corrcoef(x_rank, y_rank)

    def spearman_diff(self, x, y):
        n = x.shape[0]
        x_rank = soft_rank(x.reshape(1, -1))
        y_rank = soft_rank(y.reshape(1, -1))

        return self.corrcoef(x_rank / n, y_rank / n)
