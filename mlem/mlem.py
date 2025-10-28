import typing as tp

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from .batch_size import estimate_batch_size
from .feature_importance import compute_feature_importance
from .model import Model
from .pairwise_dataloader import PairwiseDataloader


class MLEM:
    def __init__(
        self,
        interactions: bool = False,
        # conditional_pfi: bool = True, # TODO: implement conditional PFI
        n_permutations: int = 5,
        distance: tp.Union[
            str, tp.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = "euclidean",
        nan_to_num: float = 0.0,
        batch_size: tp.Optional[int] = None,
        n_trials: int = 16,
        threshold: float = 0.01,
        factor: float = 1.2,
        batch_size_min: int = 256,
        batch_size_max: int = 2**20,
        max_steps: int = 1000,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        patience: int = 50,
        verbose: bool = False,
        memory: tp.Literal["low", "medium", "high"] = "medium",
        random_seed: tp.Optional[int] = None,
    ):
        """Initializes the MLEM model.

        Args:
            interactions (bool, default=False): If True, the model will learn weights for
                feature interactions (e.g., feature_A x feature_B) and compute
                corresponding feature importance.
            n_permutations (int, default=5): The number of permutations used to compute
                feature importance in `.score()`.
            distance (str or callable, default='euclidean'): The distance metric to
                compute pairwise distances between neural representations (Y). Can be
                'euclidean', 'manhattan', 'cosine', 'norm_diff', 'precomputed', or a
                custom callable. If 'precomputed', Y is expected to be a precomputed
                distance matrix.
            nan_to_num (float, default=0.0): The value used to replace any NaN values that
                may arise during feature distances computation.
            batch_size (int | None, default=None): The number of pairs of stimuli to sample
                in each batch during training and scoring. If None, a batch size is
                automatically estimated.
            n_trials (int, default=16): The number of batches to sample when automatically
                estimating the batch size (`batch_size`). A higher number leads to a more
                reliable estimate.
            threshold (float, default=0.02): The stability threshold for automatic batch
                size estimation. The estimation process stops when the standard deviation
                of feature-feature correlations across trials falls below this value.
            factor (float, default=1.2): The multiplicative factor by which `batch_size` is
                increased at each step of the automatic batch size estimation.
            batch_size_min (int, default=256): The initial number of pairs to try when
                starting the automatic batch size estimation.
            batch_size_max (int, default=2**20): Maximum number of pairs to consider
                during automatic batch size estimation.
            max_steps (int, default=1000): Maximum number of training steps.
            lr (float, default=0.1): The learning rate for the optimizer used to
                train the model.
            weight_decay (float, default=0.0): L2 regularization strength applied by
                the optimizer.
            patience (int, default=50): Number of steps to wait for an improvement
                in the train loss before early stopping.
            verbose (bool, default=False): If True, progress bars will be displayed during
                model fitting and scoring to show progress.
            memory (tp.Literal["low", "medium", "high"], default='medium'): The memory
                usage profile for feature importance computation. 'low' is the slowest but
                most memory-efficient by iterating over features and permutations. 'high'
                is the fastest but uses the most memory by vectorizing over permutations
                and features. 'medium' is intermediate as it vectorizes over permutations
                but iterates over features.
            random_seed (int | None, default=None): A seed for the random number
                generator to ensure reproducibility of results.
        """
        self.interactions = interactions
        # self.conditional_pfi = conditional_pfi
        self.n_permutations = n_permutations
        self.distance = distance
        self.nan_to_num = nan_to_num
        self.batch_size = batch_size
        self.n_trials = n_trials
        self.threshold = threshold
        self.factor = factor
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.max_steps = max_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        self.memory = memory
        self.random_seed = random_seed

        self.model_ = None
        self.feature_names = None
        self.X_ = None
        self.Y_ = None
        self.batch_size_fit_ = None
        self.rng_ = None
        if self.random_seed is not None:
            self.rng_ = jax.random.PRNGKey(self.random_seed)

    def fit(
        self,
        X: tp.Union[pd.DataFrame, np.ndarray, jnp.ndarray],
        Y: tp.Union[pd.DataFrame, np.ndarray, jnp.ndarray],
        feature_names: tp.Optional[list[str]] = None,
    ) -> "MLEM":
        """Fits the MLEM model to the provided data.

        Args:
            X: Feature data. It needs to be of
                shape (n_samples, n_features) or (n_samples, n_samples, n_features) if
                `distance` is 'precomputed'.
            Y: Neural representation data. It
                needs to be of shape (n_samples, n_features) or
                (n_samples, n_samples, n_features) if `distance` is 'precomputed'.
            feature_names: A list of feature names. If not provided,
                it will be inferred from the columns of X if it's a DataFrame.

        Returns:
            The fitted model instance.
        """
        if feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.astype(str).tolist()
        elif feature_names is not None:
            self.feature_names = [str(f) for f in feature_names]
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.X_ = self._preprocess_features(X)
        self.Y_ = self._preprocess_representations(Y)

        if self.batch_size is None:
            if self.rng_ is not None:
                rng_est, self.rng_ = jax.random.split(self.rng_)
            else:
                rng_est = None
            dl_estimation = PairwiseDataloader(
                X=self.X_,
                Y=None,
                feature_names=self.feature_names,
                distance=self.distance,  # type: ignore
                interactions=False,  # batch_size is estimated with correlations between features not between pairs of features
                nan_to_num=self.nan_to_num,
                rng=rng_est,
            )
            self.batch_size_fit_ = estimate_batch_size(
                dl_estimation,
                self.batch_size_min,
                self.n_trials,
                self.threshold,
                self.factor,
                self.batch_size_max,
                self.verbose,
            )
        else:
            self.batch_size_fit_ = self.batch_size

        if self.rng_ is not None:
            rng_dl, rng_model, self.rng_ = jax.random.split(self.rng_, 3)
        else:
            rng_dl = None
            rng_model = None
            
        dl = PairwiseDataloader(
            X=self.X_,
            Y=self.Y_,
            feature_names=self.feature_names,
            distance=self.distance,  # type: ignore
            interactions=self.interactions,
            nan_to_num=self.nan_to_num,
            rng=rng_dl,
        )

        self.model_ = Model(
            n_features=self.X_.shape[1],
            interactions=self.interactions,
            rng=rng_model,
        )

        # Use optax for JAX optimization
        import optax
        optimizer = optax.adamw(
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
        )
        opt_state = optimizer.init(self.model_.get_params())

        # Define the loss function and gradient
        def loss_fn(params, X_batch, Y_batch):
            # Temporarily set params for forward pass
            old_params = self.model_.get_params()
            self.model_.set_params(params)
            Y_pred = self.model_.forward(X_batch)
            score = self.model_.spearman_diff(Y_pred, Y_batch)
            self.model_.set_params(old_params)
            return -score  # negative because we want to maximize
        
        grad_fn = jax.grad(loss_fn)

        best_spearman = -jnp.inf
        best_params = None
        steps_no_improve = 0
        pbar = tqdm(
            total=self.max_steps,
            desc=f"Fitting model with batches of size {self.batch_size_fit_}",
            disable=not self.verbose,
        )
        with pbar:
            for _ in range(self.max_steps):
                X_batch, Y_batch = dl.sample(self.batch_size_fit_)
                
                # Compute gradients
                params = self.model_.get_params()
                grads = grad_fn(params, X_batch, Y_batch)
                
                # Update parameters
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                self.model_.set_params(params)
                
                # Compute score for tracking
                Y_pred = self.model_.forward(X_batch)
                score = self.model_.spearman_diff(Y_pred, Y_batch)
                
                if score > best_spearman:
                    best_spearman = score
                    best_params = params.copy()
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
                pbar.set_postfix(
                    {
                        "Score": float(best_spearman),
                        "Patience": self.patience - steps_no_improve,
                    }
                )
                if steps_no_improve >= self.patience:
                    pbar.total = pbar.n
                    pbar.refresh()
                    break
                pbar.update(1)

        if best_params is not None:
            # Restore best model
            self.model_.set_params(best_params)

        return self

    def score(
        self,
        X: tp.Optional[tp.Union[pd.DataFrame, np.ndarray, jnp.ndarray]] = None,
        Y: tp.Optional[tp.Union[pd.DataFrame, np.ndarray, jnp.ndarray]] = None,
        warning_threshold: float = 0.05,
        batch_size: tp.Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Computes permutation feature importances.

        Args:
            X: The feature data. It
                needs to be of shape (n_samples, n_features) or
                (n_samples, n_samples, n_features) if `distance` is 'precomputed'. If
                None, the training data is used.
            Y: The neural representation
                data. It needs to be of shape (n_samples, n_features) or
                (n_samples, n_samples, n_features) if `distance` is 'precomputed'. If
                None, the training data is used.
            warning_threshold: The threshold for the standard deviation of
                baseline scores above which a warning is issued.
            batch_size: The number of pairs to use for scoring. If None, the
                number of pairs used during fitting is used.

        Returns:
            A tuple containing a DataFrame of feature
                importances and a Series of baseline scores across permutations.
        """
        if self.model_ is None or self.batch_size_fit_ is None:
            raise RuntimeError("You must call fit before calling score.")
        if X is None or Y is None:
            assert (
                X is None and Y is None
            ), "You must provide both X and Y or none of them."
            # If X and Y are not provided, use the training data
            X = self.X_
            Y = self.Y_
        else:
            X = self._preprocess_features(X)
            Y = self._preprocess_representations(Y)

        if self.rng_ is not None:
            rng_dl, rng_score, self.rng_ = jax.random.split(self.rng_, 3)
        else:
            rng_dl = None
            rng_score = None
            
        dl = PairwiseDataloader(
            X=X,  # type: ignore
            Y=Y,
            feature_names=self.feature_names,
            distance=self.distance,  # type: ignore
            interactions=self.interactions,
            nan_to_num=self.nan_to_num,
            rng=rng_dl,
        )

        return compute_feature_importance(
            self.model_,
            dataloader=dl,
            n_permutations=self.n_permutations,
            # If not overridden, use the batch_size used during fitting
            batch_size=batch_size or self.batch_size or self.batch_size_fit_,  # type: ignore
            verbose=self.verbose,
            warning_threshold=warning_threshold,
            memory=self.memory,
            rng=rng_score,
        )

    def get_weights(self) -> pd.DataFrame:
        """Returns the learned feature weights as a pandas DataFrame.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Returns:
            pd.DataFrame: A DataFrame containing the feature weights.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before getting the weights.")
        return self.model_.format_W(self.feature_names)

    def _preprocess_representations(
        self, Y: tp.Union[pd.DataFrame, pd.Series, np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Preprocesses the neural representation data by converting it to a
        jax array.

        Args:
            Y: The input neural data.

        Returns:
            The preprocessed neural data as a JAX array.
        """
        if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
            Y = Y.values  # type: ignore
        if isinstance(Y, np.ndarray):
            Y = jnp.array(Y)

        # add hidden size dimension if needed
        if Y.ndim == 1:
            Y = Y[:, None]

        # flatten if needed
        Y = Y.reshape(Y.shape[0], -1)  # type: ignore

        return Y.astype(jnp.float32)  # type: ignore

    def _preprocess_features(
        self, X: tp.Union[pd.DataFrame, np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Preprocesses the feature data by encoding and converting it to a
        jax array.

        Args:
            X: The input feature data.

        Returns:
            The preprocessed feature data as a JAX array.
        """
        if not self.distance == "precomputed" and isinstance(X, pd.DataFrame):
            X = self._encode_df(X)
        if isinstance(X, np.ndarray):
            X = jnp.array(X)

        return X.astype(jnp.float32)  # type: ignore

    def _encode_df(self, df: pd.DataFrame) -> jnp.ndarray:
        """Encodes a pandas DataFrame into a jax array.

        Numerical columns are min-max scaled. Categorical columns are converted to codes.

        Args:
            df: The DataFrame to encode.

        Returns:
            The encoded data as a JAX array.
        """
        if df.empty:
            # Return an empty array with the correct number of columns but 0 rows
            return jnp.empty((0, df.shape[1]), dtype=jnp.float32)

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

        return jnp.array(X)
