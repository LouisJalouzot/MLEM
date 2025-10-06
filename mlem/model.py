import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from torchsort import soft_rank


class CholeskySPD(nn.Module):
    """Creates a Symmetric Positive Definite (SPD) matrix from a vector of
    lower triangular elements.
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
        self.register_buffer("rows", tril[0])
        self.register_buffer("cols", tril[1])
        self.register_buffer("diag_indices", torch.arange(n_features))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Builds an SPD matrix from a vector of lower triangular elements.

        Args:
            w (torch.Tensor): A 1D tensor containing the elements for the lower triangular
                matrix.

        Returns:
            torch.Tensor: The flattened lower triangular part of the resulting SPD matrix.
        """
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
    ):
        """Initializes the Model module.

        Args:
            n_features (int): The number of input features.
            interactions (bool, default=False): If True, learns an SPD matrix for feature
                interactions. If False, learns a positive vector of feature weights.
        """
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the weighted sum of features.

        Args:
            X (torch.Tensor): The input feature distances tensor.

        Returns:
            torch.Tensor: The predicted neural distances.
        """
        return X @ self.W

    def format_W(self, features: list[str] | None = None) -> pd.DataFrame:
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
            W_square[*np.triu_indices(self.n_features)] = W
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
