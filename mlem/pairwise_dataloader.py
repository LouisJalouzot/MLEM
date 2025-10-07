import typing as tp

import torch
import torch.nn.functional as F


class PairwiseDataloader:
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor | None,
        feature_names: tp.List[str] | None = None,
        distance: (
            tp.Literal["euclidean", "manhattan", "cosine", "norm_diff", "precomputed"]
            | tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ) = "euclidean",
        interactions: bool = False,
        nan_to_num: float = 0.0,
        min_max_scale: bool = True,
        # memory: str = "low",  # TODO: Implement a low memory version if necessary
    ):
        """Initializes the PairwiseDataloader.

        Args:
            X (torch.Tensor): Feature tensor of shape (n_stimuli, n_features) or
                (n_stimuli, n_stimuli, n_features) if `distance` is 'precomputed'.
            Y (torch.Tensor | None): Neural representation tensor of shape
                (n_stimuli, hidden_dim) or (n_stimuli, n_stimuli) if `distance` is
                'precomputed'.
            feature_names (tp.List[str] | None, default=None): List of feature names.
            distance (str or callable, default='euclidean'): The distance metric to use
                for Y. Can be 'euclidean', 'manhattan', 'cosine', 'norm_diff',
                'precomputed', or a custom callable that takes two tensors of shape
                (batch_size, hidden_dim) and returns a tensor of distances of shape
                (batch_size,).
            interactions (bool, default=False): Whether to include feature interactions.
            nan_to_num (float, default=0.0): Value to replace NaNs with when computing
                feature distances.
            min_max_scale (bool, default=True): Whether to min-max scale the neural
                distances for numerical stability.
        """
        self.X = X
        self.Y = Y
        self.distance = distance
        self.interactions = interactions
        self.nan_to_num = nan_to_num
        self.min_max_scale = min_max_scale

        if Y is not None:
            assert (
                X.shape[0] == Y.shape[0]
            ), f"X and Y should have the same first dimension (number of stimuli) but found {X.shape[0]} and {Y.shape[0]}"
        if self.distance == "precomputed":
            assert (
                X.ndim == 3 and X.shape[0] == X.shape[1]
            ), f"If precomputed, X should be of shape (n_stimuli, n_stimuli, n_features) but found {X.shape}"
            if Y is not None:
                assert (
                    Y.ndim == 2 and Y.shape[0] == Y.shape[1]
                ), f"If precomputed, Y should be of shape (n_stimuli, n_stimuli) but found {Y.shape}"
        else:
            assert (
                X.ndim == 2
            ), f"If not precomputed, X should be of shape (n_stimuli, n_features) but found {X.shape}"

        self.n_stimuli = X.shape[0]
        self.n_features = X.shape[-1]
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        self.triu_indices = torch.triu_indices(self.n_features, self.n_features)
        if self.interactions:
            self.feature_names = [
                (
                    self.feature_names[i] + " x " + self.feature_names[j]
                    if i != j
                    else self.feature_names[i]
                )
                for i, j in zip(*self.triu_indices)
            ]
        self.device = X.device
        self.m = torch.inf
        self.M = -torch.inf

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

    def sample_X(self, ind_1: torch.Tensor, ind_2: torch.Tensor) -> torch.Tensor:
        """Computes pairwise feature distances for given indices.

        Args:
            ind_1 (torch.Tensor): Indices of the first elements in pairs.
            ind_2 (torch.Tensor): Indices of the second elements in pairs.

        Returns:
            torch.Tensor: Pairwise feature distances.
        """
        if self.distance == "precomputed":
            # If distances are precomputed, just index into the matrices
            # (n_samples, n_features)
            X_dist = self.X[ind_1, ind_2]
        else:
            # Compute the pairwise distances for all the features which are encoded in X
            # (n_samples, n_features)
            X_1 = self.X[ind_1]
            X_2 = self.X[ind_2]
            # Simple computation of pairwise feature distances thanks to their encoding
            # (n_samples, n_features)
            X_dist = (X_1 - X_2).nan_to_num(self.nan_to_num).abs().clip(0, 1)

        if self.interactions:
            X_dist = X_dist[:, self.triu_indices[0]] * X_dist[:, self.triu_indices[1]]
        else:
            # Square the distances to match the squared neural distances
            X_dist **= 2

        return X_dist

    def sample_Y(self, ind_1: torch.Tensor, ind_2: torch.Tensor) -> torch.Tensor:
        """Computes pairwise neural distances for given indices.

        Args:
            ind_1 (torch.Tensor): Indices of the first elements in pairs.
            ind_2 (torch.Tensor): Indices of the second elements in pairs.

        Returns:
            torch.Tensor: Pairwise neural distances.
        """
        if self.distance == "precomputed":
            # If distances are precomputed, just index into the matrices
            # (n_samples)
            Y_dist = self.Y[ind_1, ind_2]  # type: ignore
        else:
            # Compute the pairwise distances in neural space
            # (n_samples, hidden_dim)
            Y_1 = self.Y[ind_1]  # type: ignore
            Y_2 = self.Y[ind_2]  # type: ignore

            # Compute true distance based on the specified metric
            # (n_samples)
            Y_dist = self.distance_fn(Y_1, Y_2).nan_to_num(self.nan_to_num)  # type: ignore

        if self.min_max_scale:
            # Min-max scale Y_dist for numerical stability
            # Update running min and max
            self.m = min(self.m, Y_dist.min())
            self.M = max(self.M, Y_dist.max())
            Y_dist = (Y_dist - self.m) / (self.M - self.m + 1e-8)

        return Y_dist**2

    def sample(
        self, batch_size: int = 4096, n_trials: int = 1
    ) -> tp.Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Samples pairs of stimuli and computes their feature and neural distances.

        This method efficiently samples `batch_size` x `n_trials` pairs of stimuli and
        computes the pairwise distances for features (X) and neural representations (Y).
        If `distance` is 'precomputed', it directly retrieves precomputed distances.
        Otherwise, it computes distances on the fly.

        Args:
            batch_size (int, default=4096): The number of pairs to sample per trial.
            n_trials (int, default=1): The number of independent sampling trials.

        Returns:
            (torch.Tensor, torch.Tensor) | torch.Tensor:
                - X_dist (torch.Tensor): The feature distances. Shape is
                    (batch_size, n_features) if n_trials is 1,
                  or (n_trials, batch_size, n_features) otherwise.
                - Y_dist (torch.Tensor): The neural distances.
                  Shape is (batch_size,) if n_trials is 1,
                  or (n_trials, batch_size) otherwise.
                If Y is None, only X_dist is returned.
        """
        # Sample batch_size pairs and compute feature and neural pairwise distances
        # Do n_trials different samplings efficiently
        n_samples = batch_size * n_trials
        # Select indices to make batch_size pairs
        # (n_samples,)
        ind_1 = torch.randint(0, self.n_stimuli, (n_samples,), device=self.device)
        ind_2 = torch.randint(0, self.n_stimuli, (n_samples,), device=self.device)

        X_dist = self.sample_X(ind_1, ind_2)
        if self.Y is not None:
            Y_dist = self.sample_Y(ind_1, ind_2)

        # Reshape to (n_trials, batch_size, ...) shape if n_trials > 1
        if n_trials > 1:
            X_dist = X_dist.reshape(n_trials, batch_size, -1).squeeze()
            if self.Y is not None:
                Y_dist = Y_dist.reshape(n_trials, batch_size).squeeze()

        if self.Y is None:
            return X_dist
        else:
            return X_dist, Y_dist
