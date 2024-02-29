from MLEM.plotting_class import *


class MLEMPipeline(PlottingClass):
    def __init__(self, **kwargs):
        """MLEM pipeline class

        Main parameters
        ---------------
        _`work_dir` : str, Path, default=os.getcwd()
            Working directory.
        _`dataset` : str, default="short_sentence"
            Name of the dataset. It should be saved at `work_dir`_/datasets/`dataset`_.csv, have a sentence column and the other columns will be treated as linguistics features.
        _`model` : str, default="bert-base-uncased"
            Name of the model on HuggingFace. For a multiberts model, specify the seed and step with the corresponding parameters.
        _`add_special_tokens` : bool, default=True
            Whether to add special tokens.
        _`take_activation_from` : {"first-token", "mean", "last-token"}, default="first-token"
            How to aggregate the activations on the dimension of the tokens.
        _`distance_metric` : str, default="euclidean"
            Distance metrics used in `sklearn.metrics.pairwise_distances` to compute the distance matrix.
        _`layer` : int or "net", default=5
            Layer at which the activations should be taken (starting from 1). The distance and MDS matrix will be computed at this layer. If layer is "net" the activations for all the layers are concatenated.
        _`conditional` : bool, default=False
            If True, Conditional Permutation Feature Importance will be computed. Otherwise only Permutation Feature Importance.
        """
        super().__init__(**kwargs)
