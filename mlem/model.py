import typing as tp

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from fast_soft_sort.jax_ops import soft_rank


def cholesky_spd_transform(w: jnp.ndarray, n_features: int) -> jnp.ndarray:
    """Transforms a vector of lower triangular elements into a Symmetric Positive
    Definite (SPD) matrix using the Cholesky decomposition.
    
    Args:
        w: A 1D array containing the elements for the lower triangular matrix.
        n_features: The number of features, which determines the size of the matrix.
        
    Returns:
        The flattened lower triangular part of the resulting SPD matrix.
    """
    tril = jnp.tril_indices(n_features, k=0)
    rows, cols = tril[0], tril[1]
    diag_indices = jnp.arange(n_features)
    
    # Build empty lower triangular matrix L
    L = jnp.zeros((n_features, n_features), dtype=w.dtype)
    # Fill the lower triangular part of L with elements from w
    L = L.at[rows, cols].set(w)
    # Ensure diagonal elements are positive
    diag = L[diag_indices, diag_indices]
    L = L.at[diag_indices, diag_indices].set(jax.nn.softplus(diag))
    # Build the SPD matrix W = LL.T
    W = jnp.matmul(L, L.T)
    # Normalize W to have unit Frobenius norm
    W = W / jnp.linalg.norm(W, ord='fro')
    # Return flattened lower triangular part of W
    return W[rows, cols]


def positive_transform(w: jnp.ndarray) -> jnp.ndarray:
    """Ensures the output array is positive and has a unit L2 norm.
    
    Args:
        w: The input array.
        
    Returns:
        The processed array with positive values and unit L2 norm.
    """
    # Ensure weights are positive
    w_pos = jax.nn.softplus(w)
    # Keep to L2 norm 1
    return w_pos / jnp.linalg.norm(w_pos)


class Model:
    """JAX-based model for MLEM."""
    
    def __init__(
        self,
        n_features: int,
        interactions: bool = False,
        rng: tp.Optional[jax.random.PRNGKey] = None,
    ):
        """Initializes the Model.

        Args:
            n_features: The number of input features.
            interactions: If True, learns an SPD matrix for feature
                interactions. If False, learns a positive vector of feature weights.
            rng: A JAX random key for reproducible weight initialization.
        """
        self.n_features = n_features
        self.interactions = interactions
        tril = jnp.tril_indices(n_features, k=0)
        self.rows = tril[0]
        self.cols = tril[1]

        if rng is None:
            rng = jax.random.PRNGKey(0)

        if self.interactions:
            n_params = n_features * (n_features + 1) // 2
            self.W = 2 * jax.random.uniform(rng, (n_params,)) - 1
        else:
            self.W = jax.random.uniform(rng, (n_features,))
    
    def get_params(self) -> jnp.ndarray:
        """Returns the current model parameters."""
        return self.W
    
    def set_params(self, W: jnp.ndarray):
        """Sets the model parameters."""
        self.W = W
    
    def get_transformed_W(self) -> jnp.ndarray:
        """Returns the transformed weights (after applying constraints)."""
        if self.interactions:
            return cholesky_spd_transform(self.W, self.n_features)
        else:
            return positive_transform(self.W)
    
    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        """Computes the weighted sum of features.

        Args:
            X: The input feature distances array.

        Returns:
            The predicted neural distances.
        """
        W_transformed = self.get_transformed_W()
        return X @ W_transformed

    def format_W(self, features: tp.Optional[list[str]] = None) -> pd.DataFrame:
        """Formats the learned weights into a pandas DataFrame.

        Args:
            features: A list of feature names.

        Returns:
            A DataFrame containing the formatted weights.
        """
        if features is None:
            features = [f"Feature {i+1}" for i in range(self.n_features)]
        W = np.array(self.get_transformed_W())
        if self.interactions:
            W_square = np.full((self.n_features, self.n_features), float("nan"))
            W_square[self.rows, self.cols] = W
            return pd.DataFrame(W_square, index=features, columns=features)
        else:
            return pd.DataFrame(W, index=features, columns=["Weight"])

    def corrcoef(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the Pearson correlation coefficient between two arrays.

        Args:
            x: The first input array.
            y: The second input array.

        Returns:
            The correlation coefficient.
        """
        x = x.astype(jnp.float32)
        y = y.astype(jnp.float32)
        y_n = y - y.mean()
        x_n = x - x.mean()
        y_n = y_n / jnp.linalg.norm(y_n)
        x_n = x_n / jnp.linalg.norm(x_n)

        return (y_n * x_n).sum()

    def spearman(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the Spearman rank correlation coefficient between two arrays.

        Args:
            x: The first input array.
            y: The second input array.

        Returns:
            The Spearman correlation coefficient.
        """
        dtype = x.dtype
        x_rank = jnp.argsort(jnp.argsort(x)).astype(dtype)
        y_rank = jnp.argsort(jnp.argsort(y)).astype(dtype)

        return self.corrcoef(x_rank, y_rank)

    def spearman_diff(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Computes a differentiable Spearman rank correlation using soft ranks.

        Args:
            x: The first input array.
            y: The second input array.

        Returns:
            The differentiable Spearman correlation coefficient.
        """
        n = x.shape[0]
        x_rank = soft_rank(x.reshape(1, -1))
        y_rank = soft_rank(y.reshape(1, -1))

        return self.corrcoef(x_rank / n, y_rank / n)
