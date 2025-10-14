import typing as tp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank
from torch import nn
from torch.nn.utils import parametrize


class CholeskySPD(nn.Module):
    """Parametrizes a vector of lower triangular elements into a Symmetric Positive
    Definite (SPD) matrix using the Cholesky decomposition.
    """

    def __init__(self, n_features: int):
        """Initializes the CholeskySPD module.

        Args:
            n_features (int): The number of features, which determines the size of the
                matrix.
        """
        super().__init__()
        self.n_features = n_features
        tril = torch.tril_indices(n_features, n_features, offset=0)
        self.rows = tril[0]
        self.cols = tril[1]
        self.diag_indices = torch.arange(n_features)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Builds an SPD matrix from a vector of lower triangular elements.

        Args:
            w (torch.Tensor): A 1D tensor containing the elements for the lower triangular
                matrix.

        Returns:
            torch.Tensor: The flattened lower triangular part of the resulting SPD matrix.
        """
        # Build empty lower triangular matrix L with same dtype and device as w
        L = w.new_zeros(self.n_features, self.n_features)
        # Fill the lower triangular part of L with elements from w
        L[self.rows, self.cols] = w  # type: ignore
        # Ensure diagonal elements are positive
        diag = L[self.diag_indices, self.diag_indices]  # type: ignore
        L[self.diag_indices, self.diag_indices] = F.softplus(diag)  # type: ignore
        # Build the SPD matrix W = LL.T
        W = torch.matmul(L, L.T)
        # Normalize W to have unit Frobenius norm
        W = W / torch.linalg.vector_norm(W)
        # Return flattened lower triangular part of W
        return W[self.rows, self.cols]


class Positive(nn.Module):
    """A module that ensures the output tensor is positive and has a unit L2 norm."""

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Applies softplus to ensure positivity and normalizes to unit L2 norm.

        Args:
            w (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The processed tensor with positive values and unit L2 norm.
        """
        # Ensure weights are positive
        w_pos = F.softplus(w)
        # Keep to L2 norm 1
        return w_pos / w_pos.norm()


class Model(nn.Module):
    def __init__(
        self,
        n_features: int,
        interactions: bool = False,
        device: tp.Optional[str] = None,
        rng: tp.Optional[torch.Generator] = None,
    ):
        """Initializes the Model module.

        Args:
            n_features (int): The number of input features.
            interactions (bool, default=False): If True, learns an SPD matrix for feature
                interactions. If False, learns a positive vector of feature weights.
            rng (torch.Generator | None, default=None): A random number generator for
                reproducible weight initialization.
        """
        super().__init__()
        self.n_features = n_features
        self.interactions = interactions
        tril = torch.tril_indices(n_features, n_features, 0)
        self.rows = tril[0]
        self.cols = tril[1]

        if self.interactions:
            n_params = n_features * (n_features + 1) // 2
            self.W = nn.Parameter(
                2 * torch.rand(n_params, device=device, generator=rng) - 1
            )
            parametrize.register_parametrization(
                self, "W", CholeskySPD(n_features), unsafe=True
            )
        else:
            self.W = nn.Parameter(torch.rand(n_features, device=device, generator=rng))
            parametrize.register_parametrization(self, "W", Positive())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the weighted sum of features.

        Args:
            X (torch.Tensor): The input feature distances tensor.

        Returns:
            torch.Tensor: The predicted neural distances.
        """
        return X @ self.W

    def format_W(self, features: tp.Optional[list[str]] = None) -> pd.DataFrame:
        """Formats the learned weights into a pandas DataFrame.

        Args:
            features (list[str] | None, default=None): A list of feature names.

        Returns:
            pd.DataFrame: A DataFrame containing the formatted weights.
        """
        if features is None:
            features = [f"Feature {i+1}" for i in range(self.n_features)]
        W = self.W.detach().cpu().squeeze().numpy()
        if self.interactions:
            W_square = np.full((self.n_features, self.n_features), float("nan"))
            W_square[self.rows, self.cols] = W
            return pd.DataFrame(W_square, index=features, columns=features)
        else:
            return pd.DataFrame(W, index=features, columns=["Weight"])

    def corrcoef(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the Pearson correlation coefficient between two tensors.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The correlation coefficient.
        """
        x = x.float()
        y = y.float()
        y_n = y - y.mean()
        x_n = x - x.mean()
        y_n = y_n / y_n.norm()
        x_n = x_n / x_n.norm()

        return (y_n * x_n).sum()

    @torch.no_grad()
    def spearman(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the Spearman rank correlation coefficient between two tensors.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The Spearman correlation coefficient.
        """
        dtype = x.dtype
        x_rank = x.argsort().argsort().to(dtype)
        y_rank = y.argsort().argsort().to(dtype)

        return self.corrcoef(x_rank, y_rank)

    def spearman_diff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes a differentiable Spearman rank correlation using soft ranks.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The differentiable Spearman correlation coefficient.
        """
        n = x.shape[0]
        x_rank = soft_rank(x.reshape(1, -1))
        y_rank = soft_rank(y.reshape(1, -1))

        return self.corrcoef(x_rank / n, y_rank / n)  # type: ignore
