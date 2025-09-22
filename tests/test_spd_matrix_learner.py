import numpy as np
import pandas as pd
import pytest
import torch

from mlem.spd_matrix_learner import SPDMatrixLearner


class TestSPDMatrixLearner:
    @pytest.mark.parametrize("n_features", [2, 3, 10])
    def test_spd(self, n_features):
        model = SPDMatrixLearner(n_features=n_features, interactions=True)

        # Test for correct flattened shape
        n_feature_pairs = n_features * (n_features + 1) // 2
        assert model.W.weight.shape == (1, n_feature_pairs)

        # Reconstruct the full matrix
        W = model.format_W().values
        W = np.nan_to_num(W, nan=0.0)
        W = (W + W.T) / 2  # Fill NaNs

        # Test for Frobenius norm 1
        assert np.isclose(np.linalg.norm(W, "fro"), 1.0)

        # Test for positive definiteness (eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(W)
        assert np.all(eigenvalues > 0)

        # Test forward pass
        out = model(torch.randn(5, n_feature_pairs))
        assert out.shape == (5, 1)
        assert hasattr(out, "grad_fn")

    def test_positive(self):
        model = SPDMatrixLearner(n_features=10, interactions=False)

        # Test for L2 norm 1
        assert torch.isclose(model.W.weight.norm(), torch.tensor(1.0))

        # Test for positive weights
        assert torch.all(model.W.weight > 0)

        # Test forward pass
        out = model(torch.randn(5, 10))
        assert out.shape == (5, 1)
        assert hasattr(out, "grad_fn")

    def test_spearman(self):
        model = SPDMatrixLearner(n_features=5)
        x = torch.arange(1, 6)
        y = x**2
        assert torch.allclose(model.spearman(x, y), torch.tensor(1.0))
        assert torch.allclose(model.spearman(x, y.flip(0)), torch.tensor(-1.0))
        perm = torch.randperm(5)
        assert torch.allclose(model.spearman(x[perm], y[perm]), torch.tensor(1.0))
