"""Tests for GMM-based estimation."""

from __future__ import annotations

from sklearn.exceptions import NotFittedError

import numpy as np
import pytest
from scipy.special import logsumexp

from gmm_estimator import GmmEstimator


def _make_fitted_estimator() -> GmmEstimator:
    """Create a deterministic fitted estimator without running EM."""
    estimator = GmmEstimator(n_components=2, covariance_type="full")

    estimator.weights_ = np.array([0.65, 0.35])
    estimator.means_ = np.array(
        [
            [1.0 + 0.5j, -0.5 + 0.2j],
            [-0.3 + 0.1j, 0.8 - 0.4j],
        ]
    )
    estimator.covariances_ = np.array(
        [
            [
                [1.4 + 0.0j, 0.2 + 0.1j],
                [0.2 - 0.1j, 0.9 + 0.0j],
            ],
            [
                [0.7 + 0.0j, -0.1 + 0.05j],
                [-0.1 - 0.05j, 1.2 + 0.0j],
            ],
        ]
    )
    estimator.precisions_cholesky_ = np.empty_like(estimator.covariances_)
    estimator.n_features_in_ = 2

    return estimator


def _manual_complex_logpdf(
    y: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Compute complex Gaussian log densities manually."""
    n_samples, n_features = y.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components))

    for component_idx in range(n_components):
        covariance = covariances[component_idx]
        covariance_inv = np.linalg.pinv(covariance, hermitian=True)
        sign, log_det = np.linalg.slogdet(covariance)

        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")

        for sample_idx in range(n_samples):
            residual = y[sample_idx] - means[component_idx]
            mahalanobis = residual.conj().T @ covariance_inv @ residual
            log_prob[sample_idx, component_idx] = (
                -n_features * np.log(np.pi) - log_det - np.real(mahalanobis)
            )

    return log_prob


def _manual_lmmse_estimate(
    y: np.ndarray,
    weights: np.ndarray,
    means_x: np.ndarray,
    covariances_x: np.ndarray,
    observation_matrix: np.ndarray,
    noise_covariance: np.ndarray,
) -> np.ndarray:
    """Compute the posterior-weighted LMMSE estimator manually."""
    means_y = np.array([observation_matrix @ mean for mean in means_x])
    covariances_y = np.array(
        [
            observation_matrix @ covariance @ observation_matrix.conj().T
            + noise_covariance
            for covariance in covariances_x
        ]
    )
    cross_covariances = np.array(
        [covariance @ observation_matrix.conj().T for covariance in covariances_x]
    )

    log_prob = _manual_complex_logpdf(y, means_y, covariances_y)
    weighted_log_prob = log_prob + np.log(weights)
    probabilities = np.exp(
        weighted_log_prob - logsumexp(weighted_log_prob, axis=1)[:, None]
    )

    estimates = np.zeros((y.shape[0], means_x.shape[1]), dtype=complex)

    for sample_idx, y_sample in enumerate(y):
        for component_idx in range(weights.shape[0]):
            covariance_y_inv = np.linalg.pinv(
                covariances_y[component_idx],
                hermitian=True,
            )
            estimates[sample_idx] += probabilities[
                sample_idx,
                component_idx,
            ] * (
                means_x[component_idx]
                + cross_covariances[component_idx]
                @ (covariance_y_inv @ (y_sample - means_y[component_idx]))
            )

    return estimates


def test_import() -> None:
    """Test that the public estimator can be imported."""
    assert GmmEstimator.__name__ == "GmmEstimator"


@pytest.mark.parametrize(
    ("probabilities", "selection", "expected"),
    [
        (np.array([0.2, 0.5, 0.3]), 1, np.array([1])),
        (np.array([0.2, 0.5, 0.3]), 2, np.array([1, 2])),
        (np.array([0.2, 0.5, 0.3]), 1.0, np.array([1, 2, 0])),
        (np.array([0.2, 0.5, 0.3]), 0.7, np.array([1, 2])),
        (np.array([0.2, 0.5, 0.3]), 0.9, np.array([1, 2, 0])),
    ],
)
def test_select_components(
    probabilities: np.ndarray,
    selection: int | float,
    expected: np.ndarray,
) -> None:
    """Test component selection by count and cumulative probability."""
    selected = GmmEstimator._select_components(probabilities, selection)

    np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize("selection", [0, -1, 3])
def test_validate_component_selection_rejects_invalid_component_counts(
    selection: int,
) -> None:
    """Test validation of invalid component counts."""
    with pytest.raises(ValueError):
        GmmEstimator._validate_component_selection(selection, n_components=2)


@pytest.mark.parametrize("selection", [0.0, -0.1, 1.1])
def test_validate_component_selection_rejects_invalid_probability_mass(
    selection: float,
) -> None:
    """Test validation of invalid probability-mass selections."""
    with pytest.raises(ValueError):
        GmmEstimator._validate_component_selection(selection, n_components=2)


def test_validate_component_selection_rejects_bool() -> None:
    """Test that bool is not accepted as component selection."""
    with pytest.raises(TypeError):
        GmmEstimator._validate_component_selection(True, n_components=2)


def test_validate_observations_converts_real_array_to_complex() -> None:
    """Test observation validation and dtype normalization."""
    y = np.array([[1.0, 2.0], [3.0, 4.0]])

    validated = GmmEstimator._validate_observations(y)

    assert validated.shape == (2, 2)
    assert np.iscomplexobj(validated)


def test_validate_observations_rejects_non_2d_input() -> None:
    """Test that observations must be two-dimensional."""
    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Expected y to be a 2D array"):
        GmmEstimator._validate_observations(y)


def test_validate_noise_covariance() -> None:
    """Test noise covariance validation and dtype normalization."""
    y = np.ones((3, 2), dtype=complex)
    noise_covariance = np.eye(2)

    validated = GmmEstimator._validate_noise_covariance(noise_covariance, y)

    assert validated.shape == (2, 2)
    assert np.iscomplexobj(validated)


def test_validate_noise_covariance_rejects_wrong_shape() -> None:
    """Test rejection of noise covariance matrices with incompatible shape."""
    y = np.ones((3, 2), dtype=complex)
    noise_covariance = np.eye(3)

    with pytest.raises(ValueError, match="Expected noise_covariance to have shape"):
        GmmEstimator._validate_noise_covariance(noise_covariance, y)


def test_validate_observation_matrix_uses_identity_by_default() -> None:
    """Test default identity observation matrix."""
    estimator = _make_fitted_estimator()

    observation_matrix = estimator._validate_observation_matrix(
        observation_matrix=None,
        n_observations=2,
    )

    np.testing.assert_allclose(observation_matrix, np.eye(2, dtype=complex))


def test_validate_observation_matrix_rejects_wrong_number_of_rows() -> None:
    """Test rejection when observation matrix rows do not match observations."""
    estimator = _make_fitted_estimator()
    observation_matrix = np.ones((3, 2), dtype=complex)

    with pytest.raises(ValueError, match="matching observation dimensions"):
        estimator._validate_observation_matrix(
            observation_matrix=observation_matrix,
            n_observations=2,
        )


def test_validate_observation_matrix_rejects_wrong_number_of_columns() -> None:
    """Test rejection when observation matrix columns do not match features."""
    estimator = _make_fitted_estimator()
    observation_matrix = np.ones((2, 3), dtype=complex)

    with pytest.raises(ValueError, match="Expected observation_matrix to have"):
        estimator._validate_observation_matrix(
            observation_matrix=observation_matrix,
            n_observations=2,
        )


def test_full_covariances_keeps_full_covariances() -> None:
    """Test full covariance normalization for full covariances."""
    estimator = _make_fitted_estimator()

    covariances = estimator._full_covariances()

    np.testing.assert_allclose(covariances, estimator.covariances_)


def test_full_covariances_expands_diagonal_covariances() -> None:
    """Test full covariance normalization for diagonal covariances."""
    estimator = _make_fitted_estimator()
    estimator.covariance_type = "diag"
    estimator.covariances_ = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    covariances = estimator._full_covariances()

    expected = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(covariances, expected)


def test_full_covariances_expands_spherical_covariances() -> None:
    """Test full covariance normalization for spherical covariances."""
    estimator = _make_fitted_estimator()
    estimator.covariance_type = "spherical"
    estimator.covariances_ = np.array([1.5, 2.5], dtype=complex)

    covariances = estimator._full_covariances()

    expected = np.array(
        [
            [[1.5, 0.0], [0.0, 1.5]],
            [[2.5, 0.0], [0.0, 2.5]],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(covariances, expected)


def test_full_covariances_rejects_inconsistent_shape() -> None:
    """Test rejection of covariance arrays inconsistent with covariance_type."""
    estimator = _make_fitted_estimator()
    estimator.covariance_type = "diag"
    estimator.covariances_ = np.ones((2, 2, 2), dtype=complex)

    with pytest.raises(ValueError, match="Expected covariance shape"):
        estimator._full_covariances()


def test_observation_parameters() -> None:
    """Test observation-domain means, covariances, and cross-covariances."""
    estimator = _make_fitted_estimator()

    observation_matrix = np.array(
        [
            [1.0 + 0.0j, 0.2 - 0.1j],
            [-0.3 + 0.2j, 0.7 + 0.0j],
            [0.5 - 0.1j, -0.4 + 0.3j],
        ]
    )
    noise_covariance = 0.1 * np.eye(3, dtype=complex)

    means_y, covariances_y, cross_covariances = estimator._observation_parameters(
        observation_matrix,
        noise_covariance,
    )

    expected_means_y = np.array(
        [observation_matrix @ mean for mean in estimator.means_]
    )
    expected_covariances_y = np.array(
        [
            observation_matrix @ covariance @ observation_matrix.conj().T
            + noise_covariance
            for covariance in estimator.covariances_
        ]
    )
    expected_cross_covariances = np.array(
        [
            covariance @ observation_matrix.conj().T
            for covariance in estimator.covariances_
        ]
    )

    np.testing.assert_allclose(means_y, expected_means_y)
    np.testing.assert_allclose(covariances_y, expected_covariances_y)
    np.testing.assert_allclose(cross_covariances, expected_cross_covariances)


def test_component_probabilities_match_manual_logpdf() -> None:
    """Test posterior component probabilities against a manual calculation."""
    estimator = _make_fitted_estimator()

    y = np.array(
        [
            [0.7 + 0.2j, -0.1 + 0.4j],
            [-0.2 + 0.3j, 0.5 - 0.1j],
        ]
    )
    observation_matrix = np.eye(2, dtype=complex)
    noise_covariance = 0.2 * np.eye(2, dtype=complex)

    means_y, covariances_y, _ = estimator._observation_parameters(
        observation_matrix,
        noise_covariance,
    )

    probabilities = estimator._component_probabilities(y, means_y, covariances_y)

    manual_log_prob = _manual_complex_logpdf(y, means_y, covariances_y)
    manual_weighted_log_prob = manual_log_prob + np.log(estimator.weights_)
    expected = np.exp(
        manual_weighted_log_prob
        - logsumexp(manual_weighted_log_prob, axis=1)[:, None]
    )

    np.testing.assert_allclose(probabilities, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_covariance_pinv_matches_manual_pinv() -> None:
    """Test the pseudo-inverse path used by the estimator."""
    estimator = _make_fitted_estimator()

    observation_matrix = np.array(
        [
            [1.0 + 0.0j, 0.2 - 0.1j],
            [-0.3 + 0.2j, 0.7 + 0.0j],
            [0.5 - 0.1j, -0.4 + 0.3j],
        ]
    )
    noise_covariance = 0.2 * np.eye(3, dtype=complex)

    _, covariances_y, _ = estimator._observation_parameters(
        observation_matrix,
        noise_covariance,
    )

    covariances_y_inv = np.linalg.pinv(covariances_y, hermitian=True)
    expected = np.array(
        [
            np.linalg.pinv(covariance_y, hermitian=True)
            for covariance_y in covariances_y
        ]
    )

    np.testing.assert_allclose(covariances_y_inv, expected)


def test_lmmse_estimate_matches_manual_formula() -> None:
    """Test the component-wise LMMSE formula."""
    y = np.array([0.7 + 0.2j, -0.1 + 0.4j])
    mean_x = np.array([0.2 + 0.1j, -0.3 + 0.4j])
    mean_y = np.array([0.1 - 0.2j, 0.5 + 0.3j])
    cross_covariance = np.array(
        [
            [0.7 + 0.0j, 0.2 - 0.1j],
            [-0.1 + 0.2j, 0.9 + 0.0j],
        ]
    )
    covariance_y = np.array(
        [
            [1.2 + 0.0j, 0.2 + 0.1j],
            [0.2 - 0.1j, 1.1 + 0.0j],
        ]
    )
    covariance_y_inv = np.linalg.pinv(covariance_y, hermitian=True)

    estimate = GmmEstimator._lmmse_estimate(
        y=y,
        mean_x=mean_x,
        cross_covariance=cross_covariance,
        covariance_y_inv=covariance_y_inv,
        mean_y=mean_y,
    )

    expected = mean_x + cross_covariance @ (covariance_y_inv @ (y - mean_y))

    np.testing.assert_allclose(estimate, expected)


def test_estimate_matches_manual_full_estimator() -> None:
    """Test full estimator output against an independent manual implementation."""
    estimator = _make_fitted_estimator()

    y = np.array(
        [
            [0.7 + 0.2j, -0.1 + 0.4j, 0.2 - 0.3j],
            [-0.2 + 0.3j, 0.5 - 0.1j, 0.4 + 0.2j],
        ]
    )
    observation_matrix = np.array(
        [
            [1.0 + 0.0j, 0.2 - 0.1j],
            [-0.3 + 0.2j, 0.7 + 0.0j],
            [0.5 - 0.1j, -0.4 + 0.3j],
        ]
    )
    noise_covariance = 0.2 * np.eye(3, dtype=complex)

    estimate = estimator.estimate(
        y=y,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
    )

    expected = _manual_lmmse_estimate(
        y=y,
        weights=estimator.weights_,
        means_x=estimator.means_,
        covariances_x=estimator.covariances_,
        observation_matrix=observation_matrix,
        noise_covariance=noise_covariance,
    )

    np.testing.assert_allclose(estimate, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "selection",
    [1, 2, 0.5, 1.0],
)
def test_estimate_matches_manual_full_estimator_for_component_selections(
    selection: int | float,
) -> None:
    """Test estimator output for different component-selection modes."""
    estimator = _make_fitted_estimator()

    y = np.array(
        [
            [0.7 + 0.2j, -0.1 + 0.4j, 0.2 - 0.3j],
            [-0.2 + 0.3j, 0.5 - 0.1j, 0.4 + 0.2j],
        ]
    )
    observation_matrix = np.array(
        [
            [1.0 + 0.0j, 0.2 - 0.1j],
            [-0.3 + 0.2j, 0.7 + 0.0j],
            [0.5 - 0.1j, -0.4 + 0.3j],
        ]
    )
    noise_covariance = 0.2 * np.eye(3, dtype=complex)

    estimate = estimator.estimate(
        y=y,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
        n_components_or_probability=selection,
    )

    assert estimate.shape == (2, 2)
    assert np.iscomplexobj(estimate)
    assert np.all(np.isfinite(estimate.real))
    assert np.all(np.isfinite(estimate.imag))


@pytest.mark.parametrize(
    ("covariance_type", "blocks"),
    [
        ("full", None),
        ("diag", None),
        ("spherical", None),
        ("circulant", None),
        ("block-circulant", (2, 2)),
        ("toeplitz", None),
        ("block-toeplitz", (2, 2)),
    ],
)
def test_estimate_after_fit_for_covariance_types(
    covariance_type: str,
    blocks: tuple[int, int] | None,
) -> None:
    """Test estimator usage after fitting all supported covariance types."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(80, 4)) + 1j * rng.normal(size=(80, 4))

    estimator = GmmEstimator(
        n_components=2,
        covariance_type=covariance_type,
        blocks=blocks,
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    estimator.fit(x_train)

    observation_matrix = np.array(
        [
            [1.0 + 0.0j, 0.2 - 0.1j, 0.0 + 0.1j, -0.2 + 0.0j],
            [0.1 + 0.2j, 0.9 + 0.0j, -0.3 + 0.0j, 0.0 - 0.1j],
            [-0.2 + 0.0j, 0.1 + 0.1j, 1.0 + 0.0j, 0.2 - 0.2j],
        ]
    )
    noise_covariance = 0.2 * np.eye(3, dtype=complex)

    x_test = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    noise = 0.01 * (rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3)))
    y = x_test @ observation_matrix.T + noise

    estimates = estimator.estimate(
        y=y,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
    )

    assert estimates.shape == (5, 4)
    assert np.iscomplexobj(estimates)
    assert np.all(np.isfinite(estimates.real))
    assert np.all(np.isfinite(estimates.imag))


def test_estimate_with_identity_observation_matrix() -> None:
    """Test estimation with default identity observation matrix."""
    estimator = _make_fitted_estimator()

    y = np.array(
        [
            [0.7 + 0.2j, -0.1 + 0.4j],
            [-0.2 + 0.3j, 0.5 - 0.1j],
        ]
    )
    noise_covariance = 0.2 * np.eye(2, dtype=complex)

    estimates = estimator.estimate(
        y=y,
        noise_covariance=noise_covariance,
    )

    assert estimates.shape == (2, 2)
    assert np.iscomplexobj(estimates)
    assert np.all(np.isfinite(estimates.real))
    assert np.all(np.isfinite(estimates.imag))


def test_estimate_rejects_too_many_selected_components() -> None:
    """Test that estimate rejects component counts larger than the fitted model."""
    estimator = _make_fitted_estimator()

    y = np.array([[0.7 + 0.2j, -0.1 + 0.4j]])
    noise_covariance = 0.2 * np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match="must not exceed"):
        estimator.estimate(
            y=y,
            noise_covariance=noise_covariance,
            n_components_or_probability=3,
        )


def test_estimate_rejects_unfitted_estimator() -> None:
    """Test that estimate rejects unfitted estimators."""
    estimator = GmmEstimator(n_components=2)

    y = np.ones((2, 2), dtype=complex)
    noise_covariance = np.eye(2, dtype=complex)

    with pytest.raises(NotFittedError):
        estimator.estimate(
            y=y,
            noise_covariance=noise_covariance,
        )


def test_validate_observation_matrix_converts_real_array_to_complex() -> None:
    """Test observation-matrix dtype normalization."""
    estimator = _make_fitted_estimator()
    observation_matrix = np.eye(2)

    validated = estimator._validate_observation_matrix(
        observation_matrix=observation_matrix,
        n_observations=2,
    )

    assert validated.shape == (2, 2)
    assert np.iscomplexobj(validated)


def test_validate_noise_covariance_rejects_non_2d_input() -> None:
    """Test that noise covariance must be two-dimensional."""
    y = np.ones((3, 2), dtype=complex)
    noise_covariance = np.ones(2)

    with pytest.raises(ValueError, match="Expected noise_covariance to be a 2D matrix"):
        GmmEstimator._validate_noise_covariance(noise_covariance, y)


def test_full_covariances_rejects_unsupported_covariance_type() -> None:
    """Test rejection of unsupported covariance types."""
    estimator = _make_fitted_estimator()
    estimator.covariance_type = "unsupported"

    with pytest.raises(NotImplementedError, match="Unsupported covariance_type"):
        estimator._full_covariances()


def test_estimate_with_single_component_matches_manual_formula() -> None:
    """Test estimate with only the most likely component."""
    estimator = _make_fitted_estimator()

    y = np.array([[0.7 + 0.2j, -0.1 + 0.4j]])
    observation_matrix = np.eye(2, dtype=complex)
    noise_covariance = 0.2 * np.eye(2, dtype=complex)

    means_y, covariances_y, cross_covariances = estimator._observation_parameters(
        observation_matrix,
        noise_covariance,
    )
    covariances_y_inv = np.linalg.pinv(covariances_y, hermitian=True)
    probabilities = estimator._component_probabilities(y, means_y, covariances_y)
    component_idx = np.argmax(probabilities[0])

    expected = estimator._lmmse_estimate(
        y=y[0],
        mean_x=estimator.means_[component_idx],
        cross_covariance=cross_covariances[component_idx],
        covariance_y_inv=covariances_y_inv[component_idx],
        mean_y=means_y[component_idx],
    )

    estimate = estimator.estimate(
        y=y,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
        n_components_or_probability=1,
    )

    np.testing.assert_allclose(estimate[0], expected)