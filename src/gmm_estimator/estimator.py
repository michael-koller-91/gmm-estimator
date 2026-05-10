"""GMM-based estimator for linear inverse problems with complex-valued priors."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from cplx_gmm import GaussianMixtureCplx
from cplx_gmm._covariances import compute_precision_cholesky
from scipy.special import logsumexp
from sklearn.utils.validation import check_is_fitted

ComplexArray: TypeAlias = npt.NDArray[np.complexfloating]



class GmmEstimator(GaussianMixtureCplx):
    """GMM-based estimator for linear inverse problems with complex-valued priors.

    The estimator assumes a linear noisy observation model

    ``y = A x + n``,

    where ``x`` is the complex-valued signal of interest, ``A`` is the known 
    observation matrix, and ``n`` is zero-mean complex Gaussian noise with 
    known covariance. Given a fitted complex-valued GMM prior for ``x``, 
    the estimator computes a posterior-weighted sum of component-wise LMMSE 
    estimates.
    """

    def estimate(
        self,
        y: npt.ArrayLike,
        noise_covariance: npt.ArrayLike,
        observation_matrix: npt.ArrayLike | None = None,
        n_components_or_probability: int | float = 1.0,
    ) -> ComplexArray:
        """Estimate latent vectors from noisy linear observations.

        Parameters
        ----------
        y:
            Complex-valued observations of shape ``(n_samples, n_observations)``.
        noise_covariance:
            Noise covariance matrix of shape
            ``(n_observations, n_observations)``.
        observation_matrix:
            Observation matrix ``A`` of shape
            ``(n_observations, n_features)``. If omitted, the identity matrix is
            used.
        n_components_or_probability:
            Controls how many mixture components contribute to each estimate.

            If an integer is provided, the estimator uses the components with
            the highest posterior probabilities. For ``1``, only the most likely
            component is used.

            If a float is provided, the estimator uses as many highest-probability
            components as necessary to reach at least this cumulative posterior
            probability. The default ``1.0`` uses all components.

        Returns
        -------
        h_est:
            Complex-valued estimates of shape ``(n_samples, n_features)``.
        """
        check_is_fitted(self)

        y = self._validate_observations(y)
        noise_covariance = self._validate_noise_covariance(noise_covariance, y)
        observation_matrix = self._validate_observation_matrix(
            observation_matrix,
            n_observations=y.shape[1],
        )
        self._validate_component_selection(
            n_components_or_probability,
            n_components=self.weights_.shape[0],
        )

        means_y, covariances_y, cross_covariances = self._observation_parameters(
            observation_matrix,
            noise_covariance,
        )
        covariances_y_inv = np.linalg.pinv(covariances_y, hermitian=True)
        probabilities = self._component_probabilities(y, means_y, covariances_y)

        estimates = np.zeros(
            (y.shape[0], observation_matrix.shape[1]),
            dtype=complex,
        )

        for sample_idx, y_sample in enumerate(y):
            component_indices = self._select_components(
                probabilities[sample_idx],
                n_components_or_probability,
            )
            probability_sum = np.sum(probabilities[sample_idx, component_indices])

            for component_idx in component_indices:
                estimates[sample_idx] += probabilities[
                    sample_idx,
                    component_idx,
                ] * self._lmmse_estimate(
                    y=y_sample,
                    mean_x=self.means_[component_idx],
                    cross_covariance=cross_covariances[component_idx],
                    covariance_y_inv=covariances_y_inv[component_idx],
                    mean_y=means_y[component_idx],
                )

            estimates[sample_idx] /= probability_sum

        return estimates

    @staticmethod
    def _validate_observations(y: npt.ArrayLike) -> ComplexArray:
        """Validate and normalize observations."""
        y = np.asarray(y)

        if y.ndim != 2:
            raise ValueError(f"Expected y to be a 2D array, got shape {y.shape}.")

        if not np.iscomplexobj(y):
            y = y.astype(complex)

        return y

    @staticmethod   
    def _validate_noise_covariance(
        noise_covariance: npt.ArrayLike,
        y: ComplexArray,
    ) -> ComplexArray:
        """Validate and normalize the noise covariance matrix."""
        noise_covariance = np.asarray(noise_covariance)

        if noise_covariance.ndim != 2:
            raise ValueError(
                "Expected noise_covariance to be a 2D matrix, got shape "
                f"{noise_covariance.shape}."
            )

        expected_shape = (y.shape[1], y.shape[1])
        if noise_covariance.shape != expected_shape:
            raise ValueError(
                "Expected noise_covariance to have shape "
                f"{expected_shape}, got {noise_covariance.shape}."
            )

        if not np.iscomplexobj(noise_covariance):
            noise_covariance = noise_covariance.astype(complex)

        return noise_covariance

    def _validate_observation_matrix(
        self,
        observation_matrix: npt.ArrayLike | None,
        n_observations: int,
    ) -> ComplexArray:
        """Validate and normalize the observation matrix."""
        if observation_matrix is None:
            observation_matrix = np.eye(self.n_features_in_, dtype=complex)

        observation_matrix = np.asarray(observation_matrix)

        if observation_matrix.ndim != 2:
            raise ValueError(
                "Expected observation_matrix to be a 2D matrix, got shape "
                f"{observation_matrix.shape}."
            )

        expected_observations = observation_matrix.shape[0]
        if expected_observations != n_observations:
            raise ValueError(
                "Expected observation_matrix and y to have matching observation "
                f"dimensions, got {expected_observations} and {n_observations}."
            )

        if observation_matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "Expected observation_matrix to have "
                f"{self.n_features_in_} columns, got {observation_matrix.shape[1]}."
            )

        if not np.iscomplexobj(observation_matrix):
            observation_matrix = observation_matrix.astype(complex)

        return observation_matrix

    @staticmethod
    def _validate_component_selection(
        n_components_or_probability: int | float,
        n_components: int,
    ) -> None:
        """Validate the component-selection parameter."""
        if isinstance(n_components_or_probability, bool):
            raise TypeError("n_components_or_probability must be an int or float.")

        if isinstance(n_components_or_probability, int):
            if n_components_or_probability < 1:
                raise ValueError("Number of components must be at least 1.")
            if n_components_or_probability > n_components:
                raise ValueError(
                    "Number of selected components must not exceed the number of "
                    f"fitted GMM components. Got {n_components_or_probability}, "
                    f"but the model has {n_components} components."
                )
            return

        if isinstance(n_components_or_probability, float):
            if not 0.0 < n_components_or_probability <= 1.0:
                raise ValueError("Component probability must be in the interval (0, 1].")
            return

        raise TypeError("n_components_or_probability must be an int or float.")

    def _observation_parameters(
        self,
        observation_matrix: ComplexArray,
        noise_covariance: ComplexArray,
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        """Compute component-wise observation-domain parameters."""
        covariances = self._full_covariances()

        means_y = np.einsum("ij,kj->ki", observation_matrix, self.means_)
        covariances_y = np.einsum(
            "ij,kjl,lm->kim",
            observation_matrix,
            covariances,
            observation_matrix.conj().T,
        )
        covariances_y = covariances_y + noise_covariance[None, :, :]

        cross_covariances = np.einsum(
            "kij,mj->kim",
            covariances,
            observation_matrix.conj(),
        )

        return means_y, covariances_y, cross_covariances

    def _component_probabilities(
        self,
        y: ComplexArray,
        means_y: ComplexArray,
        covariances_y: ComplexArray,
    ) -> npt.NDArray[np.floating]:
        """Compute posterior mixture probabilities in the observation domain."""
        precisions_cholesky = compute_precision_cholesky(
            covariances_y,
            covariance_type="full",
        )
        weighted_log_prob = self._estimate_log_gaussian_prob(
            y,
            means_y,
            precisions_cholesky,
            covariance_type="full",
        ) + np.log(self.weights_)

        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        return np.exp(weighted_log_prob - log_prob_norm[:, None])

    @staticmethod
    def _select_components(
        probabilities: npt.NDArray[np.floating],
        n_components_or_probability: int | float,
    ) -> npt.NDArray[np.integer]:
        """Select mixture components for one sample."""
        indices = np.argsort(probabilities)[::-1]

        if isinstance(n_components_or_probability, int):
            return indices[:n_components_or_probability]

        if n_components_or_probability == 1.0:
            return indices

        n_components = (
            np.searchsorted(
                np.cumsum(probabilities[indices]),
                n_components_or_probability,
            )
            + 1
        )
        return indices[:n_components]
    
    def _full_covariances(self) -> ComplexArray:
        """Return fitted component covariances as full covariance matrices."""
        covariances = np.asarray(self.covariances_)
        covariance_type = self.covariance_type
        n_components = self.weights_.shape[0]
        n_features = self.means_.shape[1]

        if covariance_type in {
            "full",
            "circulant",
            "block-circulant",
            "toeplitz",
            "block-toeplitz",
        }:
            expected_shape = (n_components, n_features, n_features)
            if covariances.shape != expected_shape:
                raise ValueError(
                    "Expected covariance shape "
                    f"{expected_shape} for covariance_type {covariance_type!r}, "
                    f"got {covariances.shape}."
                )
            return covariances

        if covariance_type == "diag":
            expected_shape = (n_components, n_features)
            if covariances.shape != expected_shape:
                raise ValueError(
                    "Expected covariance shape "
                    f"{expected_shape} for covariance_type 'diag', "
                    f"got {covariances.shape}."
                )
            return np.array([np.diag(covariance) for covariance in covariances])

        if covariance_type == "spherical":
            expected_shape = (n_components,)
            if covariances.shape != expected_shape:
                raise ValueError(
                    "Expected covariance shape "
                    f"{expected_shape} for covariance_type 'spherical', "
                    f"got {covariances.shape}."
                )
            return np.array(
                [
                    covariance * np.eye(n_features, dtype=complex)
                    for covariance in covariances
                ]
            )

        raise NotImplementedError(f"Unsupported covariance_type {covariance_type!r}.")

    @staticmethod
    def _lmmse_estimate(
        y: ComplexArray,
        mean_x: ComplexArray,
        cross_covariance: ComplexArray,
        covariance_y_inv: ComplexArray,
        mean_y: ComplexArray,
    ) -> ComplexArray:
        """Compute the component-wise LMMSE estimate."""
        return mean_x + cross_covariance @ (covariance_y_inv @ (y - mean_y))
