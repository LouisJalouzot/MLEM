import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from torchsort import soft_rank


class CholeskySPD(nn.Module):
    """
    Creates a Symmetric Positive Definite (SPD) matrix from a vector of
    lower triangular elements.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        tril = torch.tril_indices(n_features, n_features, offset=0)
        self.register_buffer("rows", tril[0])
        self.register_buffer("cols", tril[1])
        self.register_buffer("diag_indices", torch.arange(n_features))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Build lower triangular matrix L from vector w
        L = w.new_zeros(self.n_features, self.n_features)
        L[self.rows, self.cols] = w  # type: ignore
        # Ensure diagonal elements are positive
        diag = L[self.diag_indices, self.diag_indices]  # type: ignore
        L[self.diag_indices, self.diag_indices] = F.softplus(diag)  # type: ignore
        # Build the SPD matrix W = L @ L^T
        W = torch.matmul(L, L.T)
        # Normalize W to have unit Frobenius norm
        W = W / torch.linalg.vector_norm(W)
        # Return flattened lower triangular part of W
        return W[self.rows, self.cols]


class Positive(nn.Module):
    def forward(self, w):
        # Ensure weights are positive
        w_pos = F.softplus(w)
        # Keep to L2 norm 1
        return w_pos / w_pos.norm()


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
            n_params = n_features * (n_features + 1) // 2
            self.W = nn.Parameter(2 * torch.rand(n_params) - 1)
            parametrize.register_parametrization(
                self, "W", CholeskySPD(n_features), unsafe=True
            )
        else:
            self.W = nn.Parameter(torch.rand(n_features))
            parametrize.register_parametrization(self, "W", Positive())

    def forward(self, X):
        return X @ self.W

    def format_W(self, features=None):
        if features is None:
            features = [f"Feature {i+1}" for i in range(self.n_features)]
        W = self.W.detach().cpu().squeeze().numpy()
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
