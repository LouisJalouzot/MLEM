import typing as tp

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .estimate_batch_size import estimate_batch_size
from .feature_importance import compute_feature_importance
from .pairwise_dataloader import PairwiseDataloader
from .spd_matrix_learner import SPDMatrixLearner


class MLEM:
    def __init__(
        self,
        interactions: bool = False,
        # conditional_pfi: bool = True,
        n_permutations: int = 5,
        distance: (
            tp.Literal["euclidean", "manhattan", "cosine", "norm_diff", "precomputed"]
            | tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ) = "euclidean",
        nan_to_num: float = 0.0,
        n_pairs: int = 4096,
        n_trials: int = 64,
        threshold: float = 0.01,
        factor: float = 1.2,
        max_n_pairs: int = 2**20,
        batch_size_increase_factor: int = 1,
        max_epochs: int = 500,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        patience: int = 50,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.interactions = interactions
        # self.conditional_pfi = conditional_pfi
        self.n_permutations = n_permutations
        self.distance = distance
        self.nan_to_num = nan_to_num
        self.n_pairs = n_pairs
        self.n_trials = n_trials
        self.threshold = threshold
        self.factor = factor
        self.max_n_pairs = max_n_pairs
        self.batch_size_increase_factor = batch_size_increase_factor
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.feature_names = None
        self.X = None
        self.Y = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
        Y: pd.DataFrame | np.ndarray | torch.Tensor,
        feature_names: list[str] | None = None,
    ):
        if feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.X = self._preprocess_features(X)
        self.Y = self._preprocess_representations(Y)

        dl = PairwiseDataloader(
            self.X,
            self.Y,
            distance=self.distance,  # type: ignore
            interactions=self.interactions,
            nan_to_num=self.nan_to_num,
        )

        n_pairs = estimate_batch_size(
            dl,
            self.n_pairs,
            self.n_trials,
            self.threshold,
            self.factor,
            self.max_n_pairs,
            self.verbose,
        )

        self.model_ = SPDMatrixLearner(
            n_features=self.X.shape[1], interactions=self.interactions
        ).to(self.device)

        best_spearman = -torch.inf
        pbar = tqdm()
        with pbar:
            pass

        compute_feature_importance(
            self.model_, self.X, self.Y, self.n_permutations, self.device
        )

        return self

    def get_weights(self):
        if self.model_ is None:
            raise RuntimeError("You must call fit before getting the weights.")
        return self.model_.format_W(self.feature_names)

    def _preprocess_representations(self, Y):
        if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
            Y = Y.values
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)

        return Y.to(self.device)  # type: ignore

    def _preprocess_features(self, X):
        if not self.distance == "precomputed" and isinstance(X, pd.DataFrame):
            X = self._encode_df(X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        return X.to(self.device)

    def _encode_df(self, df: pd.DataFrame) -> torch.Tensor:
        if df.empty:
            # Return an empty tensor with the correct number of columns but 0 rows
            return torch.empty((0, df.shape[1]), dtype=torch.float32)

        X = np.zeros(df.shape, dtype=np.float32)
        number_cols = np.array(
            [
                (not isinstance(t, pd.CategoricalDtype) and np.issubdtype(t, np.number))
                for t in df.dtypes
            ]
        )
        for i in range(df.shape[1]):
            s = df.iloc[:, i]
            if number_cols[i]:
                # min max scale
                m, M = s.nanmin(), s.nanmax()
                scale = 1 / (M - m) if M > m else 1
                X[:, i] = (s.values - m) * scale
            else:
                s = s.astype("category").cat.codes
                # -1 category code corresponds to NaN values
                s[s == -1] = np.nan
                X[:, i] = s

        return torch.from_numpy(X)
        return torch.from_numpy(X)
