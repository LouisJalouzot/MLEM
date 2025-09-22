import typing as tp
import warnings

import torch
import torch.nn.functional as F


class PairwiseDataloader:
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        distance: (
            tp.Literal["euclidean", "manhattan", "cosine", "norm_diff", "precomputed"]
            | tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ) = "euclidean",
        nan_to_num: float = 0.0,
    ):
        """
        Initializes the PairwiseDataloader.

        Args:
            X (torch.Tensor): Feature tensor of shape (n_stimuli, n_features) or
                (n_stimuli, n_stimuli, n_features) if distance is 'precomputed'.
            Y (torch.Tensor): Neural representation tensor of shape (n_stimuli, hidden_dim) or
                (n_stimuli, n_stimuli) if distance is 'precomputed'.
            distance (str or callable): The distance metric to use for Y.
                Can be 'euclidean', 'manhattan', 'cosine', 'norm_diff', 'precomputed',
                or a custom callable that takes two tensors of shape (n_pairs, hidden_dim) and
                returns a tensor of distances of shape (n_pairs,).
            nan_to_num (float): Value to replace NaNs with when computing distances.
        """
        self.X = X
        self.Y = Y
        self.distance = distance
        self.nan_to_num = nan_to_num

        assert (
            X.shape[0] == Y.shape[0]
        ), f"X and Y should have the same first dimension (number of stimuli) but found {X.shape[0]} and {Y.shape[0]}"
        if self.distance == "precomputed":
            assert (
                X.ndim == 3 and X.shape[0] == X.shape[1]
            ), f"If precomputed, X should be of shape (n_stimuli, n_stimuli, n_features) but found {X.shape}"
            assert (
                Y.ndim == 2 and Y.shape[0] == Y.shape[1]
            ), f"If precomputed, Y should be of shape (n_stimuli, n_stimuli) but found {Y.shape}"
        else:
            assert (
                X.ndim == 2
            ), f"If not precomputed, X should be of shape (n_stimuli, n_features) but found {X.shape}"

        self.n_stimuli = X.shape[0]
        self.n_features = X.shape[-1]
        self.device = X.device

        if distance == "euclidean":
            # L2 distance
            self.distance_fn = lambda y1, y2: torch.linalg.norm(y1 - y2, ord=2, dim=1)
        elif distance == "manhattan":
            # L1 distance
            self.distance_fn = lambda y1, y2: torch.linalg.norm(y1 - y2, ord=1, dim=1)
        elif distance == "cosine":
            # Cosine distance
            self.distance_fn = lambda y1, y2: 1 - F.cosine_similarity(y1, y2, dim=1)
        elif distance == "norm_diff":
            # Absolute difference of norms
            self.distance_fn = lambda y1, y2: torch.abs(
                torch.linalg.norm(y1, dim=1) - torch.linalg.norm(y2, dim=1)
            )
        elif distance == "precomputed":
            # Precomputed distances
            self.distance_fn = None
        elif callable(distance):
            # Custom distance function
            self.distance_fn = distance
        else:
            raise ValueError(f"Unknown distance: {distance}")

    def sample(self, n_pairs: int = 4096, n_trials: int = 1):
        """
        Samples pairs of stimuli and computes their feature and neural distances.

        This method efficiently samples `n_pairs` * `n_trials` pairs of stimuli and
        computes the pairwise distances for features (X) and neural representations (Y).
        If `distance` is 'precomputed', it directly retrieves precomputed distances.
        Otherwise, it calculates distances on the fly.

        Args:
            n_pairs (int): The number of pairs to sample per trial.
            n_trials (int): The number of independent sampling trials.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - X_dist (torch.Tensor): The feature distances.
                  Shape is (n_pairs, n_features) if n_trials is 1,
                  or (n_trials, n_pairs, n_features) otherwise.
                - Y_dist (torch.Tensor): The neural distances.
                  Shape is (n_pairs,) if n_trials is 1,
                  or (n_trials, n_pairs) otherwise.
        """
        # Sample n_pairs pairs and compute feature and neural pairwise distances
        # Do n_trials different samplings efficiently
        n_samples = n_pairs * n_trials
        # Select indices to make n_pairs pairs
        # (n_samples,)
        ind_1 = torch.randint(0, self.n_stimuli, (n_samples,), device=self.device)
        ind_2 = torch.randint(0, self.n_stimuli, (n_samples,), device=self.device)

        if self.distance == "precomputed":
            # If distances are precomputed, just index into the matrices
            # (n_samples, n_features)
            X_dist = self.X[ind_1, ind_2]
            # (n_samples)
            Y_dist = self.Y[ind_1, ind_2]
        else:
            # Compute the pairwise distances for all the features which are encoded in X
            # (n_samples, n_features)
            X_1 = self.X[ind_1]
            X_2 = self.X[ind_2]
            # Simple computation of pairwise feature distances thanks to their encoding
            # (n_samples, n_features)
            X_dist = (X_1 - X_2).nan_to_num(self.nan_to_num).abs().clip(0, 1)

            # Compute the pairwise distances in neural space
            # (n_samples, hidden_dim)
            Y_1 = self.Y[ind_1]
            Y_2 = self.Y[ind_2]

            # Calculate true distance based on the specified metric
            # (n_samples)
            Y_dist = self.distance_fn(Y_1, Y_2).nan_to_num(self.nan_to_num)  # type: ignore

        # Reshape to (n_trials, n_pairs, ...) shape if n_trials > 1
        if n_trials > 1:
            X_dist = X_dist.reshape(n_trials, n_pairs, self.n_features)
            Y_dist = Y_dist.reshape(n_trials, n_pairs)

        return X_dist, Y_dist
