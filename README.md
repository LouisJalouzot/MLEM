# MLEM: A Simplified Implementation

This package provides a simplified, `sklearn`-like implementation of the MLEM (Manifold-based Learning for Explainable Models) method for estimating feature importance.

## Installation

You can install the package directly from the source:

```bash
pip install .
```

## Usage

The main entry point is the `MLEM` class, which follows the `sklearn` `fit`/`transform` API.

### Basic Usage

```python
import pandas as pd
import numpy as np
from mlem import MLEM

# 1. Create dummy data
n_samples = 100
n_features = 10
n_repr_dims = 128

# Feature dataframe
features = pd.DataFrame({
    f'feature_{i}': np.random.rand(n_samples) for i in range(n_features)
})

# High-dimensional representations of stimuli
representations = np.random.rand(n_samples, n_repr_dims)

# 2. Initialize and fit the model
# With feature interactions (default)
mlem_model = MLEM(max_epochs=100, interactions=True)
mlem_model.fit(features, representations)

# 3. Get the learned SPD matrix
spd_matrix = mlem_model.get_spd_matrix()

print("Learned SPD Matrix (with interactions):")
print(spd_matrix)

# Without feature interactions (diagonal matrix)
mlem_model_no_interactions = MLEM(max_epochs=100, interactions=False)
mlem_model_no_interactions.fit(features, representations)
spd_matrix_no_interactions = mlem_model_no_interactions.get_spd_matrix()

print("\nLearned SPD Matrix (no interactions):")
print(spd_matrix_no_interactions)
```

### Feature Importance

You can also compute permutation feature importance after fitting the model.

```python
# 4. Compute feature importance
feature_importance = mlem_model.compute_feature_importance(n_permutations=10)

print("\nFeature Importance:")
print(feature_importance)
```

The learned SPD matrix represents the importance and correlations of the features. Higher values on the diagonal indicate higher importance for individual features, and off-diagonal values represent the learned correlations between feature pairs. When `interactions=False`, the model learns a diagonal SPD matrix, which is equivalent to a linear regression with positive weights.

### Multi-Target Regression Example

Here's how to use MLEM with a multi-target regression problem, using `sklearn.datasets.make_regression`.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from mlem import MLEM

# 1. Create dummy data for multi-target regression
n_samples = 100
n_features = 10
n_targets = 5  # Number of targets for multi-output regression

# Generate synthetic data
features_np, representations = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_targets=n_targets,
    random_state=42
)

# Feature dataframe
features = pd.DataFrame(
    features_np,
    columns=[f'feature_{i}' for i in range(n_features)]
)

# 2. Initialize and fit the model
mlem_model = MLEM(max_epochs=100, interactions=True)
mlem_model.fit(features, representations)

# 3. Get the learned SPD matrix
spd_matrix = mlem_model.get_spd_matrix()

print("Learned SPD Matrix (multi-target regression):")
print(spd_matrix)

# 4. Compute feature importance
feature_importance = mlem_model.compute_feature_importance(n_permutations=10)

print("\nFeature Importance (multi-target regression):")
print(feature_importance)
```

# Known issues

- Building `torchsort` from source is the default behavior and can be challenging (e.g. on systems without `gcc`).
- Even with a successful build, `torchsort` may still give an error on GPU.
In both cases, consider installing `torchsort` separately using [pre-built binaries](https://github.com/teddykoker/torchsort/releases) before installing this package:
```bash
# For instance for Python 3.11, PyTorch 2.6 and CUDA 12.6
pip install https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt26cu126-cp311-cp311-linux_x86_64.whl
# or for CPU only
pip install https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt26cpu-cp311-cp311-linux_x86_64.whl
```