from tqdm.auto import tqdm

from MLEM.distance_class import *


class RegressionClass(DistanceClass):
    def __init__(
        self,
        min_max=True,
        conditional=False,
        verbose=0,
        top_units=None,
        top_units_rank=None,
        param_grid={"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]},
        n_estimators=10,
        **kwargs,
    ):
        """Regression class with methods to
            - compute feature importance for all the units in a layer or only a subset
            - compute the decoding baseline

        Parameters
        ----------
        _`min_max` : bool, default=True
            Whether to min_max scale in [0, 1] the activations distances.
        _`conditional` : bool, default=False
            If True, Conditional Permutation Feature Importance will be computed. Otherwise only Permutation Feature Importance.
        _`verbose` : {0, 1, 2, 3}, default=0
            A higher value means more details on the computation are printed.
        _`top_units_rank` : str or None, default=None
            Measure used to sort the units of a given layer. Can be one of {"R2", "R", "MSE", "MAE"} or a feature in the dataset to consider it's feature importance.
        _`top_units` : int or None, default=None
            Number of top units considered for the order defined with `top_units_rank`_.
        _`param_grid` : dict, default={"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]}
            Regularization parameter candidates for the cross-validated Ridge regressions
        _`n_estimators` : int, default=10
            In case of Conditional Permutation Feature Importance, n_estimators parameter for the importance_estimator `sklearn.ensemble.RandomForestRegressor`. Lower than the default value of 100 for computation efficiency.

        Parameters inherited from Distance class
        ----------------------------------------
        _`take_activation_from` : {"first-token", "mean", "last-token"}, default="first-token"
            How to aggregate the activations on the dimension of the tokens.
        _`distance_metric` : str, default="euclidean"
            Distance metrics used in `sklearn.metrics.pairwise_distances` to compute the distance matrix.
        _`layer` : int or "net", default=5
            Layer at which the activations should be taken (starting from 1). The distance and MDS matrix will be computed at this layer. If layer is "net" the activations for all the layers are concatenated.
        _`zscore` : bool, default=False
            Whether to zscore normalize per unit, across samples
        _`n_jobs` : int, default=-1
            n_jobs parameter for `sklearn.metrics.pairwise_distances` and `sklearn.manifold.MDS`

        Parameters inherited from Recording class
        -----------------------------------------
        _`dataset` : str, default="short_sentence"
            Name of the dataset. It should be saved at `work_dir`_/datasets/`dataset`_.csv, have a sentence column and the other columns will be treated as linguistics features.
        _`model` : str, default="bert-base-uncased"
            Name of the model on HuggingFace. For a multiberts model, specify the seed and step with the corresponding parameters.
        _`add_special_tokens` : bool, default=True
            Whether to add special tokens.
        _`seed` : int, default=0
            Random seed. Used in particular for multiberts models for which is should be in [0, 24]
        _`step` : one of [0, 10, 20, ..., 2000] or None, default=None
            Number of training steps (in k) for multiberts models.
        _`skip_existing` : int, default=np.inf
            Threshold to decide if computation should be skipped when a file already exists.

        Parameters inherited from Base class
        ------------------------------------
        _`work_dir` : str, Path, default=os.getcwd()
            Working directory.
        _`only_cpu` : bool, default=False
            Whether to force the use of the CPU even when a GPU is available.
        """
        super().__init__(**kwargs)

        self.min_max = min_max
        self.conditional = conditional
        self.verbose = verbose
        self.top_units = top_units
        self.top_units_rank = top_units_rank
        self.n_estimators = n_estimators
        self.param_grid = param_grid

    @property
    def feature_importance_path(self):
        s = f"feature_importance_conditional_{self.conditional}"
        if self.top_units_rank is not None and self.top_units is not None:
            s += f"_{self.top_units_rank}"
            s += "_top"
            s += f"_{self.top_units}"
        s += ".csv"
        return self.analysis_path / f"min_max_{self.min_max}" / s

    @property
    def decoding_baseline_path(self):
        return self.analysis_path / f"decoding_baseline.csv"

    @property
    def univariate_feature_importance_path(self):
        return (
            self.analysis_path
            / f"min_max_{self.min_max}"
            / f"univariate_feature_importance_conditional_{self.conditional}.csv.gz"
        )

    def get_activations_distance_matrix_triu(self, units=None):
        """Loads or calls the computation of the activation distance matrix. Keeps only the upper triangle to have each pair exactly once and flattens it. min_max scale in [0,1] if `min_max`_ is True.

        Parameters
        ----------
        units: array of int or None, default=None
            Array of unit indices which can be specified to compute the distance matrix only on those units.

        Returns
        -------
        distance_matrix_triu: array of shape (n_pairs)
        """
        activations_distance_matrix = self.compute_distance_matrix(units=units)
        num_samples = activations_distance_matrix.shape[0]
        triu_indices = np.triu_indices(num_samples, k=1)
        activations_distance_matrix_triu = activations_distance_matrix[triu_indices][
            :, None
        ]
        if self.min_max:
            from sklearn.preprocessing import MinMaxScaler

            activations_distance_matrix_triu = MinMaxScaler().fit_transform(
                activations_distance_matrix_triu
            )
        return activations_distance_matrix_triu

    def get_features_distance_triu(self):
        """Loads the features distance matrix.

        Returns
        -------
        features_distance_triu: array of shape (n_pairs, n_features)

        Raises
        ------
        ValueError
            If `compute_features_distance` has not been called.
        """
        if not self.features_distance_path.exists():
            raise ValueError(
                f"Features distances not computed for dataset {self.dataset}"
            )
        features_distance_matrix_triu = np.load(self.features_distance_path)["arr_0"]
        return features_distance_matrix_triu

    def compute_feature_importance(self):
        """Computes multivariate or top units feature importances and saves it at `feature_importance_path`.

        If `top_units`_ is not None, it will compute the multivariate feature importance only on the `top_units`_ units with the best univariate measure `top_units_rank`_ which can be "R2", "R", "MAE", "MSE" or a feature of the dataset in which case the measure is its feature importance. Doing that requires the univariate feature importances to be computed using `compute_univariate_feature_importance`.

        Skipped if the file exists and if `skip_existing`_ > 2.

        Raises
        ------
        Exception
            If top_units parameters are specified but the univariate feature importances are not computed.
        """
        if self.test_exists(
            desc="feature importance", path=self.feature_importance_path, threshold=2
        ):
            return

        top_units = None
        if self.top_units is not None:
            if self.univariate_feature_importance_path.exists():
                df_uni = pd.read_csv(self.univariate_feature_importance_path)
                if self.top_units_rank in self.features:
                    df_uni = df_uni[df_uni.Feature == self.top_units_rank]
                    rank_col = "importance"
                else:
                    df_uni = df_uni[df_uni.Feature == df_uni.Feature.iloc[0]]
                    rank_col = "score_" + self.top_units_rank
                top_units = df_uni.set_index("unit")[rank_col]
                if self.top_units_rank in ["MSE", "MAE"]:
                    top_units = top_units.nsmallest(self.top_units, keep="all").index
                else:
                    top_units = top_units.nlargest(self.top_units, keep="all").index
            else:
                raise Exception("Univariate feature importance not computed")

        features_distance_matrix_triu = self.get_features_distance_triu()
        activations_distance_matrix_triu = self.get_activations_distance_matrix_triu(
            units=top_units
        )

        logger.info(
            "Computing"
            + (" conditional" if self.conditional else "")
            + " permutation importance"
        )

        df = self.compute_importance(
            features_distance_matrix_triu,
            activations_distance_matrix_triu,
        )
        if self.top_units is not None:
            df["unit"] = top_units[-1]
            df["top_units"] = self.top_units
            df["top_units_rank"] = self.top_units_rank
        for param in self.identity_params:
            df[param] = self.__dict__[param]
        df["Method"] = "Multivariate"

        self.feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.feature_importance_path, index=False)
        logger.info(f"Done, saved at {self.feature_importance_path}")

    def compute_decoding_baseline(self, n_splits=3):
        """Computes the decoding baseline and saves it at `decoding_baseline_path`.

        Skipped if the file exists and if `skip_existing`_ > 2.

        Parameters
        ----------
        n_splits : int, default=3
            Number of cross-validation splits
        """
        if self.test_exists(
            desc="decoding baseline", path=self.decoding_baseline_path, threshold=2
        ):
            return

        logger.info("Computing decoding baseline")
        X = self.get_aggregated_activations()
        Y = self.get_dataset().drop(columns=["sentence"]).astype("str")

        from scipy.stats import norm
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        res = []
        model = LogisticRegressionCV(
            cv=3,
            class_weight="balanced",
            scoring="roc_auc_ovr_weighted",
            n_jobs=self.n_jobs,
            random_state=self.seed,
            solver="newton-cholesky",
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for feature in (pbar := tqdm(self.features, leave=(self.verbose > 0))):
            for i, (train_index, test_index) in enumerate(skf.split(X, Y[feature])):
                model.fit(X[train_index], Y.loc[train_index, feature])
                AUC = model.score(X[test_index], Y.loc[test_index, feature])
                res.append([feature, AUC])
                pbar.set_description(f"{feature} split {i + 1}/{n_splits}")
        res = pd.DataFrame(res, columns=["Feature", "AUC"]).groupby("Feature")
        res = res.agg(["mean", "std", "count"])
        res.columns = ["AUC", "std", "n_splits"]
        res = res.fillna(0).reset_index()
        res["sem"] = res["std"] / np.sqrt(res["n_splits"])
        res["pval"] = 2 * norm.sf(((res["AUC"] - 0.5) / res["sem"]).abs())
        res["pval"] = res.pval.fillna(1)
        for param in self.identity_params:
            res[param] = self.__dict__[param]
        res["Method"] = "Decoding"

        self.decoding_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(self.decoding_baseline_path, index=False)
        logger.info(f"Done, saved at {self.decoding_baseline_path}")

    def compute_univariate_feature_importance(self):
        """Computes univariate feature importance for all the units in the layer and saves it at `univariate_feature_importance_path`.

        Skipped if the file exists and if `skip_existing`_ > 2."""
        if self.test_exists(
            desc="univariate feature importance",
            path=self.univariate_feature_importance_path,
            threshold=2,
        ):
            return

        aggregated_activations = self.get_aggregated_activations()
        features_distance_matrix_triu = self.get_features_distance_triu()

        logger.info(f"Computing univariate feature importance at layer {self.layer}")
        from scipy.spatial.distance import pdist

        df = []
        n_units = aggregated_activations.shape[1]
        Y = np.apply_along_axis(
            lambda x: pdist(x[:, None], metric=self.distance_metric),
            axis=0,
            arr=aggregated_activations,
        )
        if self.min_max:
            from sklearn.preprocessing import MinMaxScaler

            Y = MinMaxScaler().fit_transform(Y)

        for i in tqdm(range(n_units)):
            res = self.compute_importance(
                features_distance_matrix_triu,
                Y[:, i],
            )
            res["unit"] = i
            df.append(res)
        df = pd.concat(df)
        for param in self.identity_params:
            df[param] = self.__dict__[param]
        df["Method"] = "Univariate"

        self.univariate_feature_importance_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        df.to_csv(self.univariate_feature_importance_path, index=False)
        logger.info(f"Done, saved at {self.univariate_feature_importance_path}")

    def compute_importance(self, X, Y):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge

        from MLEM.Variable_Importance.BBI_package.src.BBI import BlockBasedImportance

        ie = {
            "regression": RandomForestRegressor(
                n_estimators=self.n_estimators, n_jobs=self.n_jobs
            )
        }
        BBI = BlockBasedImportance(
            estimator=Ridge(positive=True, random_state=self.seed),
            importance_estimator=ie,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            dict_hyper=self.param_grid,
            bootstrap=False,
            scale=False,
            conditional=self.conditional,
            verbose=max(0, self.verbose - 1),
        )
        if self.verbose > 0:
            logger.info(f"Fitting")
        BBI.fit(X, Y**2)
        if self.verbose > 0:
            logger.info(
                f"Computing importance for {100 * len(self.features)} permutations"
            )
        results = BBI.compute_importance()
        df = pd.DataFrame(self.features, columns=["Feature"])
        for key in results:
            df[key] = results[key]
        return df
