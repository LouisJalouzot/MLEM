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
It is recommended to use a `pandas.DataFrame` for `X` to correctly handle categorical features. Numerical columns will be min-max scaled, and categorical columns will be encoded as integer codes. If `X` is a NumPy array or a PyTorch tensor, it is assumed to contain only numerical features.
`Y` will be flattened to a 2D tensor of shape `(n_samples, -1)`.

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

You can use MLEM on matrices of precomputed feature and neural distances. In this case `X` and `Y` are not preprocessed.

```python
X = ...  # Your precomputed feature distance matrices of shape (n_samples, n_samples, n_features)
Y = ...  # Your precomputed matrix of pairwise neural distances of shape (n_samples, n_samples)
mlem = MLEM(distance='precomputed')
mlem.fit(X, Y)
fi, s = mlem.score()
```

## Batch size estimation

The first step of the pipeline is to estimate a `batch_size` to use during training. This is done automatically in `.fit()` if `batch_size` is not provided. Since this estimation only depends on the feature data `X`, you can estimate it once and reuse it for different `Y`s.

```python
batch_size = mlem.estimate_batch_size(X)
mlem_1 = MLEM(batch_size=batch_size)
mlem_1.fit(X, Y_1)
mlem_2 = MLEM(batch_size=batch_size)
mlem_2.fit(X, Y_2)
```

# Troubleshooting

## High variability across runs

If you observe high variability in feature importance or score across runs or seeds (set by `random_seed`), in particular if modelling interactions (`interactions=True`), the model has likely not converged during training. To mitigate this, you can try decreasing `threshold` (e.g. to 0.005 instead of the default 0.01) so that the estimated `batch_size` is larger (or override it at the initialization of MLEM). Note that a larger `batch_size` will increase memory usage and increase computation time. Alternatively you can try increasing the `patience` parameter (e.g. to 100 instead of the default 50).

## Out of memory errors

### During feature importance computation

If you encounter out of memory errors when computing feature importances you can try setting the parameter `memory` to `'low'` when initializing MLEM. This will reduce memory usage at the cost of increased computation time. On the other hand, if you have a lot of memory available, you can set `memory` to `'high'` to speed up computation.

### During batch size estimation or during training

If you encounter out of memory errors during batch size estimation or during training, you can try increasing the `threshold` parameter (e.g. to 0.02 instead of the default 0.01) so that the estimated `batch_size` is smaller. Or directly set the `batch_size` parameter to a smaller value. Note that this will decrease the precision of the method and induce more variability across runs.