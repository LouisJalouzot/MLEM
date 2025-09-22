import pytest
import torch

from mlem.pairwise_dataloader import PairwiseDataloader


@pytest.fixture
def setup_data():
    """Provides common test data."""
    n_stimuli = 10
    n_features = 5
    hidden_dim = 8
    X = torch.randn(n_stimuli, n_features)
    Y = torch.randn(n_stimuli, hidden_dim)
    return n_features, X, Y


@pytest.mark.parametrize("n_trials", [1, 4])
@pytest.mark.parametrize("interactions", [True, False])
def test_sample_shapes(setup_data, n_trials, interactions):
    """Tests the output shapes of the sample method."""
    n_features, X, Y = setup_data
    n_pairs = 16
    dataloader = PairwiseDataloader(X, Y, interactions=interactions)
    X_dist, Y_dist = dataloader.sample(n_pairs=n_pairs, n_trials=n_trials)

    if interactions:
        expected_n_features = n_features * (n_features + 1) // 2
    else:
        expected_n_features = n_features

    if n_trials > 1:
        assert X_dist.shape == (n_trials, n_pairs, expected_n_features)
        assert Y_dist.shape == (n_trials, n_pairs)
    else:
        assert X_dist.shape == (n_pairs, expected_n_features)
        assert Y_dist.shape == (n_pairs,)


@pytest.mark.parametrize("n_trials", [1, 4])
@pytest.mark.parametrize("interactions", [True, False])
def test_precomputed_sampling(setup_data, n_trials, interactions):
    """Tests sampling with precomputed data."""
    n_features, X, Y = setup_data
    # Create precomputed distance matrices
    X = X.T.unsqueeze(-1)
    X_precomputed = torch.cdist(X, X).permute(1, 2, 0)
    Y_precomputed = torch.cdist(Y, Y)
    dataloader = PairwiseDataloader(
        X_precomputed, Y_precomputed, distance="precomputed", interactions=interactions
    )

    n_pairs = 16
    X_dist, Y_dist = dataloader.sample(n_pairs=n_pairs, n_trials=n_trials)

    if interactions:
        expected_n_features = n_features * (n_features + 1) // 2
    else:
        expected_n_features = n_features

    if n_trials > 1:
        assert X_dist.shape == (n_trials, n_pairs, expected_n_features)
        assert Y_dist.shape == (n_trials, n_pairs)
    else:
        assert X_dist.shape == (n_pairs, expected_n_features)
        assert Y_dist.shape == (n_pairs,)
