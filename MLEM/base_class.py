import json
import os
from pathlib import Path

import numpy as np
from loguru import logger

_IDENTITY_PARAMS = [
    "dataset",
    "_model",
    "seed",
    "step",
    "add_special_tokens",
    "zscore",
    "take_activation_from",
    "distance_metric",
    "layer",
    "min_max",
    "conditional",
]


class BaseClass:
    def __init__(self, work_dir=None, only_cpu=False, **kwargs):
        """Base class with methods for initializing devices and dumping configs.

        Parameters
        ----------
        _`work_dir` : str, Path, default=os.getcwd()
            Working directory.
        _`only_cpu` : bool, default=False
            Whether to force the use of the CPU even when a GPU is available.
        """
        self.only_cpu = only_cpu
        if work_dir is None:
            self.work_dir = os.getcwd()
        else:
            self.work_dir = work_dir
        self.identity_params = _IDENTITY_PARAMS

    @property
    def work_dir(self):
        return self.work_dir_

    @work_dir.setter
    def work_dir(self, new_work_dir):
        self.work_dir_ = Path(new_work_dir)

    @property
    def experiments_path(self):
        return self.work_dir / "experiments"

    def enable_gpu_determinism(self):
        """Force PyTorch to use deterministic operations for replicability"""
        from torch import use_deterministic_algorithms

        use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def get_free_gpu(self):
        """
        Returns:
            free_gpu_id (int): GPU id with free memory. Useful when cluster computing.
        """
        import subprocess

        rows = (
            subprocess.check_output(
                ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
            )
            .decode("utf-8")
            .split("\n")
        )
        free_rams = tuple(map(lambda x: float(x.rstrip(" [MiB]")), rows[1:-1]))
        max_free = max(free_rams)
        max_free_idxs = tuple(
            i for i in range(len(free_rams)) if abs(max_free - free_rams[i]) <= 200
        )
        return np.random.choice(max_free_idxs)

    def get_device(self):
        """
        Returns:
            device (str): a device to move PyTorch tensors to. 'CUDA' if available and only_cpu is false or 'cpu'.
        """
        from torch import device
        from torch.cuda import is_available

        if (not self.only_cpu) and is_available():
            free_gpu_id = self.get_free_gpu()
            device = device(free_gpu_id)
        else:
            device = device("cpu")
        logger.info(f"Running on {device}")
        return device

    def dump_config(self, path):
        """
        Dumps in path the identifying parameters of this configuration
        """
        with (path / "config.json").open("w") as f:
            config = {}
            for param in self.identity_params:
                if param in self.__dict__:
                    config[param] = self.__dict__[param]
            json.dump(config, f)

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\nWorking directory: {self.work_dir}"
        if "dataset_path" in dir(self):
            s += f"\nDataset path: {self.dataset_path}"
        s += "\nParameters"
        for param in self.identity_params:
            if param in self.__dict__:
                s += f"\n   {param}: {self.__dict__[param]}"
        return s
