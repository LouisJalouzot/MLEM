import warnings

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
    return {
        "X": X,
        "Y": Y,
        "n_stimuli": n_stimuli,
        "n_features": n_features,
        "hidden_dim": hidden_dim,
    }


def test_initialization(setup_data):
    """Tests basic initialization."""
    X, Y = setup_data["X"], setup_data["Y"]
    dataloader = PairwiseDataloader(X, Y, distance="euclidean")
    assert dataloader.n_stimuli == X.shape[0]
    assert dataloader.n_features == X.shape[1]
    assert not dataloader.precomputed


def test_shape_mismatch_error(setup_data):
    """Tests that an error is raised for shape mismatches."""
    X, Y = setup_data["X"], setup_data["Y"]
    with pytest.raises(AssertionError):
        PairwiseDataloader(X, Y[1:], distance="euclidean")


def test_precomputed_initialization(setup_data):
    """Tests initialization with precomputed data."""
    n_stimuli = setup_data["n_stimuli"]
    n_features = setup_data["n_features"]
    X_precomputed = torch.randn(n_stimuli, n_stimuli, n_features)
    Y_precomputed = torch.randn(n_stimuli, n_stimuli)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dataloader = PairwiseDataloader(
            X_precomputed, Y_precomputed, precomputed=True, distance="euclidean"
        )
        assert len(w) == 1
        assert "Distance is ignored" in str(w[-1].message)

    assert dataloader.precomputed
    assert dataloader.distance_fn is None


def test_invalid_distance_metric(setup_data):
    """Tests that an error is raised for an invalid distance metric."""
    X, Y = setup_data["X"], setup_data["Y"]
    with pytest.raises(ValueError):
        PairwiseDataloader(X, Y, distance="invalid_metric")  # type: ignore


@pytest.mark.parametrize("n_trials", [1, 4])
def test_sample_shapes(setup_data, n_trials):
    """Tests the output shapes of the sample method."""
    X, Y = setup_data["X"], setup_data["Y"]
    n_features = setup_data["n_features"]
    n_pairs = 64
    dataloader = PairwiseDataloader(X, Y, distance="euclidean")
    X_dist, Y_dist = dataloader.sample(n_pairs=n_pairs, n_trials=n_trials)

    if n_trials > 1:
        assert X_dist.shape == (n_trials, n_pairs, n_features)
        assert Y_dist.shape == (n_trials, n_pairs)
    else:
        assert X_dist.shape == (n_pairs, n_features)
        assert Y_dist.shape == (n_pairs,)


@pytest.mark.parametrize(
    "distance, expected_dist_fn",
    [
        ("euclidean", lambda y1, y2: torch.linalg.norm(y1 - y2, ord=2, dim=1)),
        ("manhattan", lambda y1, y2: torch.linalg.norm(y1 - y2, ord=1, dim=1)),
        (
            "cosine",
            lambda y1, y2: 1 - torch.nn.functional.cosine_similarity(y1, y2, dim=1),
        ),
        (
            "norm_diff",
            lambda y1, y2: torch.abs(
                torch.linalg.norm(y1, dim=1) - torch.linalg.norm(y2, dim=1)
            ),
        ),
    ],
)
def test_distance_functions(setup_data, distance, expected_dist_fn):
    """Tests the different distance metrics."""
    X, Y = setup_data["X"], setup_data["Y"]
    dataloader = PairwiseDataloader(X, Y, distance=distance)
    _, Y_dist = dataloader.sample(n_pairs=1, n_trials=1)

    # To test the distance, we can't easily know the sampled indices,
    # so we'll manually call the distance function on two vectors and compare.
    y1, y2 = Y[0:1], Y[1:2]
    expected = expected_dist_fn(y1, y2)

    # We can't guarantee the sample will be (0,1), but we can check if the function is set correctly
    # by comparing its output on a fixed pair with the expected output.
    manual_dist = dataloader.distance_fn(y1, y2)  # type: ignore
    assert torch.allclose(manual_dist, expected)


def test_custom_distance_function(setup_data):
    """Tests a custom distance function."""
    X, Y = setup_data["X"], setup_data["Y"]
    custom_dist = lambda y1, y2: torch.sum((y1 - y2) ** 2, dim=1)
    dataloader = PairwiseDataloader(X, Y, distance=custom_dist)
    _, Y_dist = dataloader.sample(n_pairs=10)
    assert Y_dist.shape == (10,)


def test_precomputed_sampling(setup_data):
    """Tests sampling with precomputed data."""
    n_stimuli = setup_data["n_stimuli"]
    n_features = setup_data["n_features"]
    X_precomputed = torch.randn(n_stimuli, n_stimuli, n_features)
    Y_precomputed = torch.randn(n_stimuli, n_stimuli)
    dataloader = PairwiseDataloader(X_precomputed, Y_precomputed, precomputed=True)

    n_pairs = 50
    X_dist, Y_dist = dataloader.sample(n_pairs=n_pairs)
    assert X_dist.shape == (n_pairs, n_features)
    assert Y_dist.shape == (n_pairs,)


def test_nan_to_num(setup_data):
    """Tests the nan_to_num functionality."""
    X, Y = setup_data["X"], setup_data["Y"]
    X[0, 0] = float("nan")
    Y[1, 0] = float("nan")

    dataloader = PairwiseDataloader(X, Y, distance="euclidean", nan_to_num=123.0)
    X_dist, Y_dist = dataloader.sample(n_pairs=1000)  # Sample enough to likely get nan

    assert not torch.isnan(X_dist).any()
    assert not torch.isnan(Y_dist).any()
    assert not torch.isnan(X_dist).any()
    assert not torch.isnan(Y_dist).any()
