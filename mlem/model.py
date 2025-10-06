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
        # Indices for the lower triangular part of a matrix
        self.tril_indices = torch.tril_indices(n_features, n_features, offset=0)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Constructs the SPD matrix.
        Args:
            w: A flat vector of size n_features * (n_features + 1) / 2,
               representing the lower triangular elements of a matrix.
        Returns:
            A symmetric positive definite matrix W of shape (n_features, n_features).
        """
        # Create a zero matrix and fill the lower triangular part
        L = torch.zeros(self.n_features, self.n_features, device=w.device, dtype=w.dtype)
        L[self.tril_indices[0], self.tril_indices[1]] = w

        # Ensure diagonal elements are positive for positive definiteness
        L_diag = torch.diag(L)
        L.diagonal().copy_(F.softplus(L_diag))

        # Compute the SPD matrix W = L @ L.T
        W = L @ L.T

        # Normalize to have a Frobenius norm of 1
        W = W / torch.norm(W, p="fro")

        return W[self.tril_indices[0], self.tril_indices[1]]


class Positive(nn.Module):
    def forward(self, w):
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
