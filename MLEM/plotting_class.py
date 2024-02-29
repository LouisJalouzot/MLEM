from glob import glob

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib.concurrent import thread_map

from MLEM.regression_class import *

empty_handle = lambda: mlines.Line2D([], [], alpha=0)


class PlottingClass(RegressionClass):
    def __init__(
        self,
        style="ticks",
        context="talk",
        figsize=(12, 8),
        rename={},
        alpha=0.01,
        correction="FDR",
        palette=None,
        **kwargs,
    ):
        """Plotting class with methods to
            - aggregate the results into Pandas dataframes
            - reproduce the different figures

        Parameters
        ----------
        _`style` : {"darkgrid", "whitegrid", "dark", "white", "ticks"}, default="ticks"
            Seaborn style
        _`context` : {"paper", "notebook", "talk", "poster"}, default="talk"
            Seaborn context
        _`figsize` : tuple, default=(12, 8)
        _`rename` : dict, default={}
            Dictionary to rename anything related to the dataset to display on the figures
        _`alpha` : float, default=0.01
            Statistical significance threshold, feature importances with a p-value above this level are replaced by NaNs.
        _`correction` : {None, "FDR", "HS"}, default="FDR"
            Type of correction used on the univariate p-values. "FDR" for `scipy.stats.false_discovery_control` and "HS" for `statsmodels.stats.multitest.multipletests`.
        _`palette` : str, dict or None, default=None
            Color palette to use. Can the name of a seaborn palette or a dictionary mapping (renamed) features to colors supported by seaborn.

        Parameters inherited from Regression class
        ------------------------------------------
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

        self.style = style
        self.context = context
        self.figsize = figsize
        self.rename = rename
        self.alpha = alpha
        self.correction = correction
        self.palette = palette

    @property
    def top_units_rank(self):
        if (
            "rename" in dir(self)
            and self.rename is not None
            and self.top_units_rank_ in self.rename
        ):
            return self.rename[self.top_units_rank_]
        return self.top_units_rank_

    @top_units_rank.setter
    def top_units_rank(self, new_top_units_rank):
        self.top_units_rank_ = new_top_units_rank

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        sns.set_style(style)
        self._style = style

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context):
        sns.set_context(context)
        self._context = context

    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, figsize):
        sns.set(
            context=self.context,
            style=self.style,
            rc={"figure.figsize": figsize},
        )
        self._figsize = figsize

    @property
    def features(self):
        features = list(self.get_dataset(nrows=0))
        features.remove("sentence")
        if "rename" in dir(self) and self.rename is not None:
            return [self.rename[f] if f in self.rename else f for f in features]
        else:
            return features

    def get_results(self, method="multivariate"):
        """Loads computed results for all the layers. It also concatenates for all the seeds and steps for multiberts and all the top_units if top_units_rank is defined.

        Parameters
        ----------
        method : {"multivariate", "univariate", "decoding"}
        """
        method = method.lower()
        df = []
        layer = self.layer
        seed = self.seed
        step = self.step
        top_units = self.top_units
        self.layer = self.seed = self.step = "*"
        if self.top_units_rank is not None:
            self.top_units = "*"
        if method == "multivariate":
            files = glob(str(self.feature_importance_path))
        elif method == "univariate":
            files = glob(str(self.univariate_feature_importance_path))
        elif method == "decoding":
            files = glob(str(self.decoding_baseline_path))
        else:
            raise Exception(
                f"Method {method} not supported, choose from multivariate/univariate/decoding"
            )
        self.layer = layer
        self.seed = seed
        self.step = step
        self.top_units = top_units
        desc_method = (
            f"top {self.top_units_rank} "
            if self.top_units_rank is not None and method == "multivariate"
            else ""
        ) + method
        if len(files) == 0:
            raise Exception(f"No result files found for the {desc_method} method.")
        if self.verbose > 0:
            logger.info(
                f"{len(files)} result files found for the {desc_method} method."
            )
        df = pd.concat(
            thread_map(
                pd.read_csv,
                files,
                leave=(self.verbose > 0),
                max_workers=(self.n_jobs if self.n_jobs > 0 else None),
            ),
            ignore_index=True,
        )
        df = df.replace(self.rename)
        df = df.rename(columns=self.rename)
        if method == "univariate":
            gb = df.groupby(["Feature", "layer", "seed", "step"], dropna=False)
            if self.correction == "FDR":
                from scipy.stats import false_discovery_control

                df["pval"] = gb.pval.transform(false_discovery_control)
            elif self.correction == "HS":
                from statsmodels.stats.multitest import multipletests

                df["pval"] = gb.pval.transform(
                    lambda x: multipletests(x, alpha=self.alpha)[1]
                )
            elif self.correction is None:
                pass
            else:
                raise Exception(
                    f"Correction {self.correction} is not implemented, choose from None/FDR/HS"
                )
        if method != "decoding":
            df.loc[df.pval > self.alpha, "importance"] = np.nan
        return df

    def top_features(self, df, top=4):
        """Finds the features achieving maximum feature importance across layers (averaged across other dimensions, e.g. seed)

        Parameters
        ----------
        df : input dataframe
        top : int, default=4
            number of features to keep

        Returns
        -------
        df : input dataframe with only the top features
        top_features : list of top features
        palette : color palette for those features
        """
        top_features = (
            df.groupby(["Feature", "layer"])
            .importance.mean()
            .reset_index()
            .groupby("Feature")
            .importance.max()
            .sort_values(ascending=False)
            .index[:top]
        )
        df = df[df.Feature.isin(top_features)].copy()
        df["Feature"] = pd.Categorical(
            df.Feature, categories=top_features, ordered=True
        )
        if isinstance(self.palette, dict):
            palette = self.palette
        else:
            palette = {
                f: sns.color_palette(self.palette)[i]
                for i, f in enumerate(top_features)
            }
        return df, top_features, palette

    def plot_correlation(self, thresh=0.5):
        """Plot the correlations between the features in the features distance matrix.

        Parameters
        ----------
        thresh : float, default=0.5
            Correlations under this threshold will not be annotated.

        Returns
        -------
        ax : matplotlib axes
        """
        df = pd.DataFrame(self.get_features_distance_triu(), columns=self.features)
        corr = df.corr()
        mask_triu = np.triu(np.ones_like(corr, dtype=bool))
        mask = (corr.abs() > thresh) & ~mask_triu
        corr.index *= mask.any(axis=1)
        corr.columns *= mask.any(axis=0)
        annot = corr.round(1)
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        ax = sns.heatmap(
            corr,
            annot=annot,
            annot_kws=dict(fontsize=10),
            cmap=cmap,
            vmin=-1,
            vmax=1,
            mask=(~mask.values | mask_triu),
        )
        ax = sns.heatmap(corr, mask=mask_triu, cmap=cmap, vmin=-1, vmax=1, cbar=False)
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False)
        ax.set_title(self.dataset)
        ax.figure.tight_layout()
        return ax

    def plot_feature_importance(
        self,
        decoding=False,
        top=4,
        log_y=True,
        linewidth=7,
        **kwargs,
    ):
        """Feature Importance profile across layers (averaged over any other dimension relevant for the the chosen parameters). If `decoding` is True, the decoding baseline AUCs are also shown on a second x-axis.

        `compute_feature_importance` (and `compute_decoding_baseline` if `decoding` is True) should be run for each layer beforehand, otherwise points will be missing.

        Parameters
        ----------
        top : int, default=4
            Number of top features to display.
        decoding : bool, default=False
            If True, the decoding baseline AUCs are also displayed.
        log_y : bool, default=True
        linewidth : int, default=7

        Returns
        -------
        ax: matplotlib axes
        """
        df = self.get_results()
        df, top_features, palette = self.top_features(df, top=top)
        if log_y:
            s = df.groupby(["Feature", "layer"], observed=True).importance.transform(
                "mean"
            )
            df.loc[s < 1e-10, "importance"] = np.nan
        if decoding:
            df_decoding = self.get_results(method="decoding")
            df_decoding = df_decoding[df_decoding.Feature.isin(top_features)]
            df = pd.concat([df, df_decoding])

        df["Method"] = df.Method.apply(
            lambda x: (x + " (AUC)" if x == "Decoding" else x + " (FI)")
        )
        df["Method"] = pd.Categorical(
            df.Method, categories=df.Method.unique(), ordered=True
        )
        ax = sns.lineplot(
            df[-df.importance.isna()],
            x="layer",
            y="importance",
            hue="Feature",
            style=("Method" if decoding else None),
            palette=palette,
            errorbar=("ci", 100 * (1 - self.alpha)),
            linewidth=linewidth,
            **kwargs,
        )
        if decoding:
            ax2 = ax.twinx()
            ax2.grid(False)
            sns.lineplot(
                df[-df.AUC.isna()],
                x="layer",
                y="AUC",
                hue="Feature",
                style="Method",
                legend=None,
                ax=ax2,
                palette=palette,
                linewidth=linewidth,
            )
            ax2.set_ylim(0.45, 1.05)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Feature Importance")
        ax.set_xticks(sorted(df.layer.unique()))
        if log_y:
            ax.set_yscale("log")
        if decoding:
            empty_handles = [
                empty_handle() for _ in range(len(top_features) - decoding - 1)
            ]
        else:
            empty_handles = []
        ax.figure.tight_layout()
        ax.legend(
            title="" if decoding else "Feature",
            handles=ax.get_legend_handles_labels()[0] + empty_handles,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            columnspacing=1,
            handlelength=3,
        )
        ax.figure.tight_layout()
        return ax

    def plot_mds(
        self,
        hierarchy,
        palette="tab20b",
        markers=["o", "^", "X"],
        **kwargs,
    ):
        """Plots the MDS at `layer`_ and highlights the hierarchy defined by the 3 features provided in `hierarchy`.

        Parameters
        ----------
        hierarchy : list of size 3
            List of 3 (renamed) features defining the hierarchy
        palette : str, default="tab20b"
        markers : list, default=["o", "^", "X"]

        Returns
        -------
        ax: matplotlib axes
        """
        palette = sns.color_palette(palette)
        df = self.get_dataset(mds=True)
        if df is None:
            return
        df = df.rename(columns=self.rename)
        df = df.replace(self.rename)
        if self.rename is not None:
            hierarchy = [self.rename[h] if h in self.rename else h for h in hierarchy]
        for h in hierarchy:
            df[h] = df[h].astype(str)

        cat_1 = sorted(df[hierarchy[0]].unique())
        cat_2 = sorted(df[hierarchy[1]].unique())
        cat_3 = sorted(df[hierarchy[2]].unique())

        marker_map = {val: markers[i] for i, val in enumerate(cat_1)}
        color_map = {
            val_2 + val_3: palette[4 * i + (2 if len(cat_3) < 3 else 1) * j]
            for i, val_2 in enumerate(cat_2)
            for j, val_3 in enumerate(cat_3)
        }

        df[hierarchy[1] + hierarchy[2]] = df[hierarchy[1]].astype("str") + df[
            hierarchy[2]
        ].astype("str")

        ax = sns.scatterplot(
            df,
            x="x",
            y="y",
            style=hierarchy[0],
            markers=marker_map,
            hue=hierarchy[1] + hierarchy[2],
            palette=color_map,
            legend=None,
            **kwargs,
        )

        from matplotlib.legend_handler import HandlerTuple

        for c in [cat_1, cat_2, cat_3]:
            if "Undefined" in c:
                c.remove("Undefined")
        max_len = max(len(cat_1), max(len(cat_2), len(cat_3)))
        h, l = [], []
        h.append(empty_handle())
        l.append(hierarchy[0])
        for val in cat_1:
            h.append(
                mlines.Line2D([], [], linestyle="", marker=marker_map[val], color="k")
            )
            l.append(val)
        for _ in range(max_len - len(cat_1)):
            h.append(empty_handle())
            l.append("")
        h.append(empty_handle())
        l.append(hierarchy[1])
        for val in cat_2:
            handles = [
                mlines.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker=markers[0],
                    color=color_map[val + val_],
                )
                for val_ in cat_3
                if ((df[hierarchy[1]] == val) & (df[hierarchy[2]] == val_)).any()
            ]
            h.append(tuple(handles))
            l.append(val)
        for _ in range(max_len - len(cat_2)):
            h.append(empty_handle())
            l.append("")
        h.append(empty_handle())
        l.append(hierarchy[2])
        for i, val in enumerate(cat_3):
            h.append(
                mlines.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker=markers[0],
                    color="k",
                    alpha=1 - i / len(cat_3),
                )
            )
            l.append(val)
        for _ in range(max_len - len(cat_3)):
            h.append(empty_handle())
            l.append("")
        ax.legend(
            h,
            l,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            columnspacing=0,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel(f"MDS at layer {self.layer}")
        ax.set_ylabel("")
        ax.figure.tight_layout()
        return ax

    def find_best_n_clusters(self, df, max_clusters=6):
        """Finds the optimal number of clusters for the points in `df` using KMeans and the silhouette method.

        Parameters
        ----------
        df : pandas dataframe
            Dataframe of points to cluster
        max_clusters : int, default=6
            Maximum number of clusters

        Returns
        -------
        best_num_clusters: int
            Optimal number of clusters
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_num_clusters = 0
        best_silhouette_score = -1
        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(
                n_clusters=num_clusters, n_init="auto", random_state=self.seed
            )
            labels = kmeans.fit_predict(df)
            silhouette_avg = silhouette_score(df, labels)

            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = num_clusters

        return best_num_clusters

    def cluster_fi(
        self,
        scale="rel",
        max_clusters=6,
    ):
        """Clusters the units of the layer `layer`_ based on their univariate Feature Importance using KMeans and the slihouette method to find the optimal number of clusters.

        Requires `compute_univariate_feature_importance` to be run beforehand for this layer.

        Parameters
        ----------
        scale : {None, "rel", "zscore"}, default="rel"
            Scaling applied on the FIs. "rel": the FIs are divided by the univariate R2 performance of the corresponding unit. "zscore": the FIs are zscored across units.
        max_clusters : int, default=6

        Returns
        -------
        df_uni: pandas dataframe
            Dataframe with univariate results and a Cluster column
        """
        df_uni = self.get_results(method="univariate").query(f"layer == {self.layer}")
        if scale == "rel":
            df_uni["importance"] /= df_uni.score_R2
        pivot_table = df_uni.pivot(index="unit", columns="Feature", values="importance")

        if scale == "zscore":
            from sklearn.preprocessing import StandardScaler

            pivot_table = pd.DataFrame(
                StandardScaler().fit_transform(pivot_table),
                index=pivot_table.index,
                columns=pivot_table.columns,
            )

        pivot_table = pivot_table.fillna(0)
        num_clusters = self.find_best_n_clusters(pivot_table, max_clusters=max_clusters)
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=self.seed)
        pivot_table["Cluster"] = kmeans.fit_predict(pivot_table) + 1

        return df_uni.merge(pivot_table.reset_index()[["unit", "Cluster"]])

    def barplot_clusters(
        self,
        scale="rel",
        max_clusters=6,
        log_y=False,
        **kwargs,
    ):
        """Univariate Feature Importance averaged over clusters of units at `layer`_. The clusters are determined by `cluster_fi` using KMeans and the silhouette method.

        Parameters
        ----------
        scale : {None, "rel", "zscore"}, default="rel"
            Scaling method on the FIs for `cluster_fi`.
        max_clusters : int, default=6
        log_y : bool, default=False

        Returns
        -------
        ax: matplotlib axes
        """
        df_uni = self.cluster_fi(scale=scale, max_clusters=max_clusters)
        df_uni = df_uni.sort_values("importance", ascending=False)
        df_uni["Feature"] = df_uni.Feature.astype("category")

        ax = sns.barplot(
            df_uni,
            x="Cluster",
            y="importance",
            hue="Feature",
            err_kws={"linewidth": 2},
            errorbar=("ci", 100 * (1 - self.alpha)),
            capsize=0.5,
            **kwargs,
        )
        ax.legend(title="Feature", ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
        ax.set_ylabel(f"Feature Importance at layer {self.layer}")
        ax.set_xlabel("Cluster")
        if log_y:
            ax.set_yscale("log")
        ax.figure.tight_layout()
        return ax

    def plot_distrib_univariate(
        self,
        optimal=False,
        palette=None,
        linewidth=5,
        **kwargs,
    ):
        """Distribution of the univariate measure defined by `top_units_rank`_ which needs to be defined. It can be one of {"R2", "R", "MSE", "MAE"} or a feature in the dataset. The corresponding multivariate measure (and top k optimal if `optimal` is True) is also displayed.

        `compute_feature_importance` and `compute_univariate_feature_importance` should be run for each layer beforehand. Furthermore if `optimal` is True, `compute_feature_importance` should be run for each layer and each possible value of `top_units`_ for the corresponding `top_units_rank`_.

        Parameters
        ----------
        optimal : bool, default=False
            Whether to display the top k optimal multivariate measure corresponding to `top_units_rank`_
        palette : str, list, default=None
            Name of a seaborn color palette or list of colors to differentiate between the methods
        linewidth : int, default=5

        Returns
        -------
        ax: matplotlib axes
        """
        if not isinstance(palette, list):
            palette = sns.color_palette(palette)

        if self.top_units_rank is None:
            logger.info(
                "top_units_rank is None, a measure is required (e.g. R2, R, MSE, MAE or a feature of the dataset)"
            )
            return
        top_units_rank = self.top_units_rank
        self.top_units_rank = None
        df = self.get_results(method="multivariate")
        self.top_units_rank = top_units_rank
        if optimal:
            df_star = self.get_results(method="multivariate")
        df_uni = self.get_results(method="univariate")

        if self.top_units_rank in self.features:
            df = df[df.Feature == self.top_units_rank]
            if optimal:
                df = df[df.Feature == self.top_units_rank]
            df_uni = df_uni[df_uni.Feature == self.top_units_rank]
            measure = "FI " + self.top_units_rank
            y = "importance"
        else:
            df = df[df.Feature == df.Feature.iloc[0]]
            if optimal:
                df_star = df_star[df_star.Feature == df_star.Feature.iloc[0]]
            df_uni = df_uni[df_uni.Feature == df_uni.Feature.iloc[0]]
            measure = self.top_units_rank
            y = "score_" + self.top_units_rank
        df_uni["Q1"] = df_uni.groupby("layer")[y].transform(lambda x: x.quantile(0.25))
        df_uni["Q3"] = df_uni.groupby("layer")[y].transform(lambda x: x.quantile(0.75))
        df_uni["IQR"] = df_uni["Q3"] - df_uni["Q1"]
        df_uni["outlier"] = (df_uni[y] <= df_uni.Q1 - 1.5 * df_uni.IQR) | (
            df_uni[y] >= df_uni.Q3 + 1.5 * df_uni.IQR
        )
        df["Method"] = measure
        if optimal:
            if self.top_units_rank in ["MAE", "MSE"]:
                df_star = df_star.loc[df_star.groupby("layer")[y].idxmin()]
            else:
                df_star = df_star.loc[df_star.groupby("layer")[y].idxmax()]
            df_star["Method"] = f"{measure}*"
            df = pd.concat([df, df_star])
        layers = sorted(pd.concat([df.layer, df_uni.layer]).unique())
        df["layer"] -= 1
        outliers = (
            df_uni.groupby("layer", observed=True)
            .outlier.sum()
            .reset_index(name="#outliers")
        )
        df_uni["layer"] = pd.Categorical(df_uni.layer, categories=layers, ordered=True)
        ax = sns.boxplot(
            data=df_uni,
            x="layer",
            y=y,
            hue="Method",
            flierprops={"marker": "x"},
            linewidth=linewidth - 2,
            legend=None,
            palette=palette[:1],
            **kwargs,
        )
        ax = sns.lineplot(
            data=df,
            x="layer",
            y=y,
            style="Method",
            color=palette[1],
            linewidth=linewidth,
            ax=ax,
        )
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.fill_between(
            outliers.layer - 1, outliers["#outliers"], alpha=0.3, label="#outliers"
        )
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim(y_min, y_max * 1.1)
        ax2.set_ylabel("#outliers")
        ax.set_xlabel("Layer")
        ax.set_ylabel(measure)
        if y.startswith("score_R"):
            ax.set_ylim(-0.05, 1.1)
        else:
            ax.set_yscale("log")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, ncols=3, loc="upper center")
        ax.figure.tight_layout()
        return ax

    def plot_top_k(self, log_x=True, linewidth=7, **kwargs):
        """Plot the measure defined by `top_units_rank`_ for the different top k multivariate which have been computed with `compute_feature_importance` at this layer. `top_units_rank`_ can be one of {"R2", "R", "MAE", "MSE"} or a feature in the dataset.

        Parameters
        ----------
        log_x : bool, default=True
        linewidth : int, default=7

        Returns
        -------
        ax : matplotlib axes
        """
        if self.top_units_rank is None:
            logger.info(
                "top_units_rank is None, a measure is required (e.g. R2, R, MSE, MAE or a feature of the dataset)"
            )
            return
        df = self.get_results(method="multivariate").query(f"layer == {self.layer}")
        if self.top_units_rank in self.features:
            df = df[df.Feature == self.top_units_rank]
            y = "importance"
        else:
            df = df[df.Feature == df.Feature.iloc[0]]
            y = "score_" + self.top_units_rank
        ax = sns.lineplot(
            df,
            x="top_units",
            y=y,
            linewidth=linewidth,
            **kwargs,
        )
        if self.top_units_rank in ["MAE", "MSE"]:
            idx_measure_star = df[y].idxmin()
        else:
            idx_measure_star = df[y].idxmax()
        d_star, measure_star = df.loc[idx_measure_star, ["top_units", y]]
        ax.axvline(
            d_star, linestyle="--", linewidth=linewidth - 2, color="k", alpha=0.5
        )
        if log_x:
            ax.set_xscale("log")
        if self.top_units_rank[0] == "R":
            ax.set_ylim(-0.05, 1.05)
        ax.axhline(
            measure_star,
            linestyle="--",
            linewidth=linewidth - 2,
            color="k",
            alpha=0.5,
        )
        ax.text(
            d_star,
            0.98 * ax.get_ylim()[0] + 0.02 * ax.get_ylim()[1],
            f"   d* = {d_star:.0f}",
            ha="left",
            va="bottom",
        )
        ax.text(
            0.9 * df.top_units.min(),
            measure_star * 1.02,
            f"{self.top_units_rank}* = {measure_star:.2f}",
            ha="left",
            va="bottom",
        )
        ax.set_ylabel(self.top_units_rank)
        ax.set_xlabel(f"Top k units of layer {self.layer + 1}")
        ax.figure.tight_layout()
        return ax
