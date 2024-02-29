from MLEM.recording_class import *


class DistanceClass(RecordingClass):
    def __init__(
        self,
        take_activation_from="first-token",
        distance_metric="euclidean",
        layer=5,
        zscore=False,
        n_jobs=-1,
        **kwargs,
    ):
        """Distance class with methods to
            - load and aggregate the activations
            - compute and save the activations distance and MDS matrices

        Parameters
        ----------
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

        self.take_activation_from = take_activation_from
        self.distance_metric = distance_metric
        self.layer = layer
        self.zscore = zscore
        self.n_jobs = n_jobs

    @property
    def analysis_path(self):
        return (
            self.net_path
            / "analysis"
            / self.take_activation_from
            / self.distance_metric
            / f"zscore_{self.zscore}"
            / f"layer_{self.layer}"
        )

    @property
    def distance_matrix_path(self):
        return self.analysis_path / "distance_matrix.npz"

    @property
    def mds_matrix_path(self):
        return self.analysis_path / "mds_matrix.npz"

    def load_distance_matrix(self):
        return np.load(self.distance_matrix_path)["arr_0"]

    def get_aggregated_activations(self):
        """Loads and aggregate the activations.

        Returns
        -------
        aggregated_activations: array of shape (n_pairs)

        Raises
        ------
        ValueError
            If `take_activation_from`_ is unsupported
        """
        self.record()
        device = self.get_device()

        from torch import concat, load

        activations_per_layer = load(
            self.recording_path / "recording.pt", map_location=device
        )
        # Concatenate the layers if `layer`_ == "net" or select the relevant one
        if self.layer == "net":
            activations = concat(activations_per_layer, dim=-1)
        else:
            activations = activations_per_layer[int(self.layer) - 1]

        attention_mask = load(
            self.recording_path / "inputs.pt", map_location=device
        ).attention_mask
        batch_size = activations.shape[0]
        # Aggregate the activations according to `take_activation_from`_
        if self.take_activation_from == "mean":
            # Because of the padding we need to use the attention_mask to know which tokens are relevant
            average_mask = (attention_mask / attention_mask.sum(dim=1)[:, None])[
                :, :, None
            ]
            aggregated_activations = (activations * average_mask).sum(dim=1)
        elif self.take_activation_from == "last-token":
            # Select the last-token, corresponding to the last 1 in the attention_mask on the second dimension
            keep_indices = attention_mask.cumsum(dim=1).argmax(dim=1)
            aggregated_activations = activations[np.arange(batch_size), keep_indices]
        elif self.take_activation_from == "first-token":
            aggregated_activations = activations[:, 0]
        else:
            raise ValueError(
                f"Unsupported activation aggregation mode {self.take_activation_from} (choose one of mean/last-token/first-token)"
            )
        aggregated_activations = np.array(aggregated_activations.cpu())
        if self.zscore:
            from scipy.stats import zscore

            # Normalize per unit, across samples
            aggregated_activations = zscore(aggregated_activations, axis=0)
        return aggregated_activations

    def test_exists(self, desc, path, threshold):
        """Helper function to skip computation if both
            - the file at `path`, which is described by `desc`, exists
            - `skip_existing`_ > `threshold`_

        Parameters
        ----------
        desc : str
        path : Path
        _`threshold` : int

        Returns
        -------
        bool: True if both conditions are met and computation should be skipped
        """
        if path.exists():
            if self.skip_existing > threshold:
                logger.info(f"{desc} exists, reading from {path}")
                return True
            else:
                logger.info(f"Deleting previous {desc} at {path}")
                try:
                    os.remove(path)
                except:
                    logger.info(f"{path} already removed")
                return False
        else:
            return False

    def compute_distance_matrix(self, units=None):
        """Computes the activations distance matrix. Saves it at `distance_matrix_path` if the parameter units is None.

        Skipped if the matrix exists and if `skip_existing`_ > 1.

        Parameters
        ----------
        units: array of int or None, default=None
            Array of unit indices which can be specified to compute the distance matrix only on those units.

        Returns
        -------
        distance_matrix: array of shape (n_sentences, n_sentences)
        """
        if units is None and self.test_exists(
            desc="distance matrix", path=self.distance_matrix_path, threshold=1
        ):
            return self.load_distance_matrix()

        aggregated_activations = self.get_aggregated_activations()

        logger.info(
            "Computing distance matrix"
            + (f" for top {len(units)} units" if units is not None else "")
        )
        from sklearn.metrics import pairwise_distances

        if units is not None:
            aggregated_activations = aggregated_activations[:, units]
        distance_matrix = pairwise_distances(
            aggregated_activations, metric=self.distance_metric, n_jobs=self.n_jobs  # type: ignore
        )

        if units is None:
            self.analysis_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.distance_matrix_path, distance_matrix)
            logger.info(f"Done, saved at {self.distance_matrix_path}")
        return distance_matrix

    def compute_mds(self):
        """Computes the MDS matrix and saves it at `mds_matrix_path`

        Skipped if the matrix exists and if `skip_existing`_ > 1.5"""
        if self.test_exists(
            desc="mds matrix", path=self.mds_matrix_path, threshold=1.5
        ):
            return

        distance_matrix = self.compute_distance_matrix()

        logger.info(f"Computing MDS matrix")
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=2,
            max_iter=200,
            eps=1e-4,
            n_init=1,
            dissimilarity="precomputed",
            random_state=self.seed,
            n_jobs=self.n_jobs,
            normalized_stress="auto",
        )
        mds_matrix = mds.fit_transform(distance_matrix)
        np.savez_compressed(self.mds_matrix_path, mds_matrix)
        self.dump_config(self.analysis_path)
        logger.info(f"Done, saved at {self.mds_matrix_path}")
