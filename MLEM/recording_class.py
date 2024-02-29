import shutil

import numpy as np
import pandas as pd

from MLEM.base_class import *


class RecordingClass(BaseClass):
    def __init__(
        self,
        dataset="short_sentence",
        model="bert-base-uncased",
        add_special_tokens=True,
        seed=0,
        step=None,
        skip_existing=np.inf,
        **kwargs,
    ):
        """Recording class with methods to
            - load datasets/language models/tokenizers
            - run a language model and save its intermediate activations

        Parameters
        ----------
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

        self.dataset = dataset
        self.seed = seed
        self.step = step
        self.model = model
        self.add_special_tokens = add_special_tokens
        self.skip_existing = skip_existing

    @property
    def model(self):
        if "multiberts" in self._model:
            if self.step is not None:
                return f"multiberts-seed_{self.seed}-step_{self.step}k"
            else:
                return f"multiberts-seed_{self.seed}"
        else:
            return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model

    @property
    def net_path(self):
        return (
            self.experiments_path
            / self.dataset
            / self.model.replace("sentence-transformers/", "")
            / ("spe_tok" if self.add_special_tokens else "no_spe_tok")
        )

    @property
    def recording_path(self):
        return self.net_path / "recordings"

    @property
    def dataset_path(self):
        return self.work_dir / "datasets" / f"{self.dataset}.csv"

    @property
    def features_distance_path(self):
        return self.work_dir / "data" / f"feature_dist_mat_triu_flat_{self.dataset}.npz"

    @property
    def features(self):
        features = list(self.get_dataset(nrows=0))
        features.remove("sentence")
        return features

    def get_dataset(self, nrows=None, usecols=None, mds=False):
        df = pd.read_csv(self.dataset_path, nrows=nrows, usecols=usecols)
        if mds:
            if self.mds_matrix_path.exists():
                mds = np.load(self.mds_matrix_path)["arr_0"]
                for param in self.identity_params:
                    df[param] = self.__dict__[param]
                df["x"] = mds[:, 0]
                df["y"] = mds[:, 1]
            else:
                logger.info("MDS matrix not computed")
                return
        return df

    def compute_features_distance(self):
        """Computes the features distance matrix and saves it at `features_distance_path`. It keeps only the upper triangle to have each pair exactly once and flattens it.

        The distance between sentence i and j for feature k is 1 if this feature is defined for both sentences and different. Otherwise it's 0.
        """
        logger.info(f"Prepare feature distance matrix for dataset {self.dataset}")
        logger.info(f"Dataset at {self.dataset_path}")
        df = self.get_dataset(usecols=lambda x: x != "sentence")
        logger.info(f"{len(df)} sentences, features: {', '.join(self.features)}")

        a = df.to_numpy()
        dist_mat = (a[:, None] != a[None, :]).astype(float)
        a = df.isna().to_numpy()
        na_mask = a[:, None] + a[None, :]
        dist_mat = np.where(na_mask, 0, dist_mat)
        dist_mat_triu_flat = dist_mat[np.triu_indices(len(dist_mat), k=1)]

        self.features_distance_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.features_distance_path, dist_mat_triu_flat)
        logger.info(
            f"Done, size {dist_mat_triu_flat.shape}, saved at {self.features_distance_path}"
        )

    def get_transformers_tokenizer(self):
        import transformers

        if self.model.startswith("multiberts"):
            model = "google/" + self.model
        else:
            model = self.model
        return transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    def get_language_model(self):
        import transformers

        if self.model.startswith("multiberts"):
            model = "google/" + self.model
        else:
            model = self.model

        model_config = transformers.AutoConfig.from_pretrained(
            model,
            output_hidden_states=True,
        )
        model = transformers.AutoModel.from_pretrained(
            model,
            config=model_config,
        )

        return model

    def test_recordings_exist(self):
        """Skip recomputing if the recordings already exist and if `skip_existing`_ > 0."""
        if not (self.recording_path / "inputs.pt").exists():
            return False
        elif not (self.recording_path / "recording.pt").exists():
            return False
        elif self.skip_existing > 0:
            logger.info(f"Recordings exist at {self.recording_path}, skipping.")
            return True
        else:
            logger.info(f"Deleting previous recordings in {self.recording_path}")
            shutil.rmtree(self.recording_path)
            return False

    def record(self):
        """Runs the language model `model`_ on the sentences in `dataset`_ and saves the intermediate activations at `recording_path`

        Requires `compute_features_distance` to have been called at least once for a given dataset and model specification
        """
        if self.test_recordings_exist():
            return

        logger.info(f"Recording activations...")
        sentences = self.get_dataset(usecols=["sentence"]).sentence

        import torch

        self.enable_gpu_determinism()
        device = self.get_device()

        net = self.get_language_model().to(device)
        tokenizer = self.get_transformers_tokenizer()  # type: ignore
        if tokenizer._pad_token is None:
            # pad_token is undefined for GPT2Tokenizer
            tokenizer._pad_token = tokenizer.eos_token

        inputs = tokenizer(
            list(sentences),
            return_tensors="pt",
            padding=True,
            add_special_tokens=self.add_special_tokens,
        ).to(device)

        self.recording_path.mkdir(parents=True, exist_ok=True)
        # Save inputs in particular for the attention_mask
        torch.save(inputs, self.recording_path / f"inputs.pt")

        net.eval()
        with torch.no_grad():
            output = net(**inputs)
            # First tuple in `hidden_states` is embeddings.
            hidden_states = output.hidden_states[1:]
            torch.save(hidden_states, self.recording_path / f"recording.pt")

        self.dump_config(self.net_path)
        logger.info(f"Done, saved at {self.recording_path}")
