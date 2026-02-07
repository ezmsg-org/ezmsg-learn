import numpy as np
import pytest

from ezmsg.learn.model.cca import IncrementalCCA, _inv_sqrtm_spd


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng):
    """Create correlated synthetic data for CCA testing."""
    n_samples, d1, d2 = 100, 5, 4
    # Shared latent factor
    latent = rng.standard_normal((n_samples, 2))
    X1 = latent @ rng.standard_normal((2, d1)) + 0.1 * rng.standard_normal((n_samples, d1))
    X2 = latent @ rng.standard_normal((2, d2)) + 0.1 * rng.standard_normal((n_samples, d2))
    return X1, X2


def test_initialize():
    """Test that initialize creates matrices with correct shapes."""
    cca = IncrementalCCA(n_components=2)
    cca.initialize(5, 4)

    assert cca.C11.shape == (5, 5)
    assert cca.C22.shape == (4, 4)
    assert cca.C12.shape == (5, 4)
    assert cca.initialized is True
    assert cca.d1 == 5
    assert cca.d2 == 4


def test_initialize_with_ref_array():
    """Test that initialize respects ref_array namespace."""
    ref = np.zeros(3)
    cca = IncrementalCCA(n_components=2)
    cca.initialize(5, 4, ref_array=ref)

    assert cca.C11.shape == (5, 5)
    assert cca.C11.dtype == np.float64


def test_partial_fit(synthetic_data):
    """Test that partial_fit runs without error and updates covariance."""
    X1, X2 = synthetic_data
    cca = IncrementalCCA(n_components=2)
    cca.partial_fit(X1, X2)

    assert cca.initialized is True
    assert not np.allclose(cca.C11, 0)
    assert not np.allclose(cca.C22, 0)
    assert not np.allclose(cca.C12, 0)
    assert hasattr(cca, "x_weights_")
    assert hasattr(cca, "y_weights_")


def test_partial_fit_incremental(synthetic_data):
    """Test that multiple partial_fit calls update smoothly."""
    X1, X2 = synthetic_data
    cca = IncrementalCCA(n_components=2)

    # First fit
    cca.partial_fit(X1[:50], X2[:50])
    C11_first = cca.C11.copy()

    # Second fit should change covariance
    cca.partial_fit(X1[50:], X2[50:])
    assert not np.allclose(C11_first, cca.C11)


def test_transform(synthetic_data):
    """Test that transform produces correct output shapes."""
    X1, X2 = synthetic_data
    cca = IncrementalCCA(n_components=2)
    cca.partial_fit(X1, X2)

    X1_proj, X2_proj = cca.transform(X1, X2)
    assert X1_proj.shape == (100, 2)
    assert X2_proj.shape == (100, 2)


def test_transform_single_component(synthetic_data):
    """Test with a single canonical component."""
    X1, X2 = synthetic_data
    cca = IncrementalCCA(n_components=1)
    cca.partial_fit(X1, X2)

    X1_proj, X2_proj = cca.transform(X1, X2)
    assert X1_proj.shape == (100, 1)
    assert X2_proj.shape == (100, 1)


def test_numerical_equivalence_inv_sqrtm():
    """Compare eigh-based _inv_sqrtm_spd with scipy.linalg on known SPD matrices."""
    scipy_linalg = pytest.importorskip("scipy.linalg")

    rng = np.random.default_rng(123)
    # Create a known SPD matrix
    A_raw = rng.standard_normal((5, 5))
    A = A_raw.T @ A_raw + 0.1 * np.eye(5)  # Ensure SPD

    # scipy reference
    sqrtm_A = scipy_linalg.sqrtm(A)
    inv_sqrtm_scipy = scipy_linalg.inv(np.real(sqrtm_A))

    # Our implementation
    inv_sqrtm_eigh = _inv_sqrtm_spd(np, A)

    np.testing.assert_allclose(inv_sqrtm_eigh, inv_sqrtm_scipy, atol=1e-10)


def test_correlations_are_computed(synthetic_data):
    """Test that canonical correlations are stored after fit."""
    X1, X2 = synthetic_data
    cca = IncrementalCCA(n_components=2)
    cca.partial_fit(X1, X2)

    assert hasattr(cca, "correlations_")
    assert len(cca.correlations_) >= 2
    # Correlations should be between 0 and 1 for well-conditioned data
    assert np.all(cca.correlations_[:2] >= 0)
    assert np.all(cca.correlations_[:2] <= 1.0 + 1e-6)
