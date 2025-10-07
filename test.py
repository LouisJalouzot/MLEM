import numpy as np
import pandas as pd

from mlem import MLEM

mlem = MLEM(
    batch_size=32,
    max_steps=5,
    verbose=True,
)

n_samples = 1024
n_features = 10
hidden_size = 32
X = pd.DataFrame({i: np.random.choice(["A", "B"], n_samples) for i in range(n_features)})
Y = pd.DataFrame({i: np.random.randn(n_samples) for i in range(hidden_size)})

mlem.fit(X, Y)
fi, s = mlem.score()
