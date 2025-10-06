import typing as tp
from time import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .compute_feature_importance import compute_feature_importance
from .estimate_batch_size import estimate_batch_size
from .model import SPDMatrixLearner
from .pairwise_dataloader import PairwiseDataloader


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
        n_pairs: int | None = None,
        n_trials: int = 16,
        threshold: float = 0.01,
        factor: float = 1.2,
        estimate_interactions: bool = False,
        starting_n_pairs: int = 256,
        max_n_pairs: int = 2**20,
        batch_size_increase_factor: int = 1,
        max_epochs: int = 1000,
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
        self.estimate_interactions = estimate_interactions
        self.starting_n_pairs = starting_n_pairs
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
            self.feature_names = X.columns.astype(str).tolist()
        elif feature_names is not None:
            self.feature_names = [str(f) for f in feature_names]
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.X = self._preprocess_features(X)
        self.Y = self._preprocess_representations(Y)

        if self.n_pairs is None:
            dl_estimation = PairwiseDataloader(
                X=self.X,
                Y=None,
                feature_names=self.feature_names,
                distance=self.distance,  # type: ignore
                interactions=self.estimate_interactions,
                nan_to_num=self.nan_to_num,
            )
            n_pairs = estimate_batch_size(
                dl_estimation,
                self.starting_n_pairs,
                self.n_trials,
                self.threshold,
                self.factor,
                self.max_n_pairs,
                self.verbose,
            )
        else:
            n_pairs = self.n_pairs

        dl = PairwiseDataloader(
            X=self.X,
            Y=self.Y,
            feature_names=self.feature_names,
            distance=self.distance,  # type: ignore
            interactions=self.interactions,
            nan_to_num=self.nan_to_num,
        )

        self.model_ = SPDMatrixLearner(
            n_features=self.X.shape[1], interactions=self.interactions
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            maximize=True,
            amsgrad=True,
            fused=True,
        )

        best_spearman = -torch.inf
        best_state_dict = None
        epochs_no_improve = 0
        pbar = tqdm(total=self.max_epochs, desc="Fitting model", disable=not self.verbose)
        self.model_.train()
        with pbar:
            for _ in range(self.max_epochs):
                X_batch, Y_batch = dl.sample(n_pairs)
                optimizer.zero_grad()
                Y_pred = self.model_(X_batch)
                score = self.model_.spearman_diff(Y_pred, Y_batch)
                score.backward()
                optimizer.step()
                if score > best_spearman:
                    best_spearman = score
                    best_state_dict = self.model_.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                pbar.set_postfix(
                    {
                        "Score": best_spearman.item(),  # type: ignore
                        "Batch size": self.n_pairs,
                        "Patience": self.patience - epochs_no_improve,
                    }
                )
                if epochs_no_improve >= self.patience:
                    break
                n_pairs = int(n_pairs * self.batch_size_increase_factor)
                pbar.update(1)

        self.model_.load_state_dict(best_state_dict)  # type: ignore

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
                m, M = np.nanmin(s.values), np.nanmax(s.values)  # type: ignore
                scale = 1 / (M - m) if M > m else 1
                X[:, i] = (s.values - m) * scale
            else:
                s = s.astype("category").cat.codes
                # -1 category code corresponds to NaN values
                s[s == -1] = np.nan
                X[:, i] = s

        return torch.from_numpy(X)
