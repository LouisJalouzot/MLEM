import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from .feature_importance import compute_feature_importance
from .spd_matrix_learner import SPDMatrixLearner
from .trainer import train


class MLEM(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        interactions: bool = True,
        lr=0.1,
        weight_decay=0,
        max_epochs=500,
        eps=1e-3,
        patience=50,
        n_pairs=4096,
        device="cpu",
    ):
        self.interactions = interactions
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.eps = eps
        self.patience = patience
        self.n_pairs = n_pairs
        self.device = device
        self.model_ = None
        self.feature_names_in_ = None
        self.X_scaled_ = None
        self.y_tensor_ = None

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns)
        self.X_scaled_ = self._preprocess_features(X)
        self.y_tensor_ = self._preprocess_representations(y)

        self.model_ = SPDMatrixLearner(
            n_features=self.X_scaled_.shape[1], interactions=self.interactions
        )

        train(
            self.model_,
            self.X_scaled_,
            self.y_tensor_,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            eps=self.eps,
            patience=self.patience,
            n_pairs=self.n_pairs,
            device=self.device,
        )
        return self

    def get_spd_matrix(self):
        if self.model_ is None:
            raise RuntimeError("You must call fit before getting the SPD matrix.")
        return self.model_.get_formatted_W(self.feature_names_in_)

    def compute_feature_importance(self, n_permutations=5):
        if self.model_ is None:
            raise RuntimeError("You must call fit before computing feature importance.")

        importances = compute_feature_importance(
            self.model_, self.X_scaled_, self.y_tensor_, n_permutations, self.device
        )

        return pd.Series(importances, index=self.feature_names_in_)

    def _preprocess_features(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Simple encoding for categorical features
        if not np.issubdtype(X.dtype, np.number):
            X_encoded = np.zeros_like(X, dtype=np.float32)
            for i in range(X.shape[1]):
                try:
                    X_encoded[:, i] = pd.to_numeric(X[:, i])
                except (ValueError, TypeError):
                    X_encoded[:, i] = pd.Categorical(X[:, i]).codes
        else:
            X_encoded = X.astype(np.float32)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        return torch.from_numpy(X_scaled).to(self.device)

    def _preprocess_representations(self, y):
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        return y.to(self.device)

    def transform(self, X):
        # Not applicable for this model, but required by sklearn API
        return X
