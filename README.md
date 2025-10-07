# Installation

You can install MLEM via pip directly from GitHub:

```bash
pip install git+https://github.com/LouisJalouzot/MLEM_test
```

# Usage

```python
from mlem import MLEM

X = ...  # Your stimuli features as a pandas DataFrame (it can contain categorical features), numpy array, or PyTorch tensor, it has to be of shape (n_samples, n_features)
Y = ...  # Your neural representations of the stimuli as a NumPy array or PyTorch tensor, it has to be of shape (n_samples, hidden_size)
mlem = MLEM()
mlem.fit(X, Y) # Train the model
feature_importances, scores = mlem.score() # Compute feature importances on the same data
```
The output `feature_importances` is a pandas DataFrame containing the feature importances for each feature (columns) across all the `n_permutations` permutations (rows). The output `scores` is a pandas Series of all the Spearman scores computed during the computation of the feature importances (number of features x `n_permutations`).

## Test-train split

With a simple train-test split:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
mlem.fit(X_train, Y_train) # Train the model
feature_importances, scores = mlem.score(X_test, Y_test) # Compute feature importances on the test set
```

With cross-validation:
```python
from sklearn.model_selection import KFold
import pandas as pd

all_importances = []
all_scores = []

kf = KFold(shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    mlem.fit(X[train_index], Y[train_index])
    fi, s = mlem.score(X[test_index], Y[test_index])
    fi["split"] = i
    s["split"] = i
    all_importances.append(fi)
    all_scores.append(s)

all_importances = pd.concat(all_importances)
all_scores = pd.concat(all_scores)
```

## Precomputed distances

You can use MLEM on matrices of precomputed feature and neural distances.

```python
X = ...  # Your precomputed feature distance matrices of shape (n_samples, n_samples, n_features)
Y = ...  # Your precomputed matrix of pairwise neural distances of shape (n_samples, n_samples)
mlem = MLEM(distance='precomputed')
mlem.fit(X, Y)
fi, s = mlem.score()
```

# Common errors

## Undefined symbol error

When importing `torchsort`, you may encounter an error similar to:

```console
Error isotonic_cpu.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK3c105Error4whatEv
```

If this is the case, try using a newer version of PyTorch.

## CUDA errors

When trying to use torchsort on CUDA (i.e. when `device='cuda'`), you may encounter the following error:

```console
ImportError: You are trying to use the torchsort CUDA extension, but it looks like it is not available. Make sure you have the CUDA toolchain installed, and reinstall torchsort with `pip install --force-reinstall --no-cache-dir torchsort` to rebuild the extension.
```

One solution is to follow the instructions from the error message, i.e. install the CUDA toolchain and reinstall torchsort. However, if you do not have the ability to compile C++/CUDA code on your machine, you can also directly install a pre-built package from [torchsort releases](https://github.com/teddykoker/torchsort/releases), e.g. for Python 3.12, PyTorch 2.6, and CUDA 12.6:

```bash
pip install --force-reinstall --no-cache-dir --no-deps https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt26cu126-cp312-cp312-linux_x86_64.whl
```

Note that not all combinations of Python/PyTorch/CUDA are available as pre-built packages.