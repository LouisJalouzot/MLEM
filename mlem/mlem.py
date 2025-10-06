import typing as tp

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .batch_size import estimate_batch_size
from .feature_importance import compute_feature_importance
from .model import SPDMatrixLearner
from .pairwise_dataloader import PairwiseDataloader


class MLEM:
    def __init__(
        self,
        interactions: bool = False,
        # conditional_pfi: bool = True, # TODO: implement conditional PFI
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
        starting_n_pairs: int = 256,
        max_n_pairs: int = 2**20,
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
        self.starting_n_pairs = starting_n_pairs
        self.max_n_pairs = max_n_pairs
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
        self.n_pairs_fit = None

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
                interactions=False,  # n_pairs is estimated with correlations between features not between pairs of features
                nan_to_num=self.nan_to_num,
            )
            self.n_pairs_fit = estimate_batch_size(
                dl_estimation,
                self.starting_n_pairs,
                self.n_trials,
                self.threshold,
                self.factor,
                self.max_n_pairs,
                self.verbose,
            )
        else:
            self.n_pairs_fit = self.n_pairs

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
        self.model_.train()
        pbar = tqdm(
            total=self.max_epochs,
            desc=f"Fitting model with batches of size {self.n_pairs_fit}",
            disable=not self.verbose,
        )
        with pbar:
            for _ in range(self.max_epochs):
                X_batch, Y_batch = dl.sample(self.n_pairs_fit)
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
                        "Patience": self.patience - epochs_no_improve,
                    }
                )
                if epochs_no_improve >= self.patience:
                    break
                pbar.update(1)

        if best_state_dict is not None:
            # Restore best model
            self.model_.load_state_dict(best_state_dict)  # type: ignore

        return self

    def score(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor | None = None,
        Y: pd.DataFrame | np.ndarray | torch.Tensor | None = None,
        warning_threshold: float = 0.05,
        n_pairs: int | None = None,
    ):
        if self.model_ is None:
            raise RuntimeError("You must call fit before calling score.")
        if X is None or Y is None:
            assert (
                X is None and Y is None
            ), "You must provide both X and Y or none of them."
            # If X and Y are not provided, use the training data
            X = self.X
            Y = self.Y
        else:
            X = self._preprocess_features(X)
            Y = self._preprocess_representations(Y)

        dl = PairwiseDataloader(
            X=X,  # type: ignore
            Y=Y,
            feature_names=self.feature_names,
            distance=self.distance,  # type: ignore
            interactions=self.interactions,
            nan_to_num=self.nan_to_num,
        )

        return compute_feature_importance(
            self.model_,
            dataloader=dl,
            n_permutations=self.n_permutations,
            # If not overridden, use the n_pairs used during fitting
            n_pairs=n_pairs or self.n_pairs or self.n_pairs_fit,  # type: ignore
            verbose=self.verbose,
            warning_threshold=warning_threshold,
        )

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
