import numpy as np
import numpy.typing as npt
from GMM_cplx import gmm_cplx
from scipy import linalg as scilinalg


class GmmEstimator(gmm_cplx.GaussianMixtureCplx):
    def estimate(
        self,
        y: npt.NDArray,
        noise_covariance: npt.NDArray,
        n_dim: int,
        A: npt.NDArray = None,
        n_components_or_probability: float = 1.0,
    ) -> npt.NDArray:
        """
        Estimate channel vectors.

        Args:
            y: A 2D complex numpy array representing the observations (one per row).
            noise_covariance: The noise covariance matrix.
            n_dim: The dimension of the channels.
            A: The observation matrix.
            n_components_or_probability:
                If this is an integer, compute the sum of the top (highest
                    component probabilities) 'n_components_or_probability'
                    LMMSE estimates.
                If this is a float, compute the sum of as many LMMSE estimates
                    as are necessary to reach at least a cumulative component
                    probability of 'n_components_or_probability'.
                By default, all components are used.
        """

        if A is None:
            A = np.eye(n_dim, dtype=y.dtype)
        y_for_prediction, covs_Cy_inv = self._prepare_for_prediction(
            y, A, noise_covariance
        )

        h_est = np.zeros([y.shape[0], A.shape[-1]], dtype=complex)
        if isinstance(n_components_or_probability, int):
            # n_components_or_probability represents a number of summands

            if n_components_or_probability == 1:
                # use predicted label to choose the channel covariance matrix
                labels = self._predict_cplx(y_for_prediction)
                for yi in range(y.shape[0]):
                    mean_h = self.means_cplx[labels[yi], :]
                    h_est[yi, :] = self._lmmse_formula(
                        y[yi, :],
                        mean_h,
                        self.covs_cplx[labels[yi], :, :] @ A.conj().T,
                        covs_Cy_inv[labels[yi], :, :],
                        A @ mean_h,
                    )
            else:
                # use predicted probabilities to compute weighted sum of estimators
                proba = self.predict_proba_cplx(y_for_prediction)
                for yi in range(y.shape[0]):
                    # indices for probabilities in descending order
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_components_or_probability]:
                        mean_h = self.means_cplx[argproba, :]
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                            y[yi, :],
                            mean_h,
                            self.covs_cplx[argproba, :, :] @ A.conj().T,
                            covs_Cy_inv[argproba, :, :],
                            A @ mean_h,
                        )
                    h_est[yi, :] /= np.sum(
                        proba[yi, idx_sort[:n_components_or_probability]]
                    )
        elif n_components_or_probability == 1.0:
            # use all predicted probabilities to compute weighted sum of estimators
            proba = self.predict_proba_cplx(y_for_prediction)
            for yi in range(y.shape[0]):
                for argproba in range(proba.shape[1]):
                    mean_h = self.means_cplx[argproba, :]
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                        y[yi, :],
                        mean_h,
                        self.covs_cplx[argproba, :, :] @ A.conj().T,
                        covs_Cy_inv[argproba, :, :],
                        A @ mean_h,
                    )
        else:
            # n_components_or_probability represents a probability
            # use predicted probabilities to compute weighted sum of estimators
            proba = self.predict_proba_cplx(y_for_prediction)
            for yi in range(y.shape[0]):
                # probabilities and corresponding indices in descending order
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = (
                    np.searchsorted(
                        np.cumsum(proba[yi, idx_sort]), n_components_or_probability
                    )
                    + 1
                )
                for argproba in idx_sort[:nr_proba]:
                    mean_h = self.means_cplx[argproba, :]
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                        y[yi, :],
                        mean_h,
                        self.covs_cplx[argproba, :, :] @ A.conj().T,
                        covs_Cy_inv[argproba, :, :],
                        A @ mean_h,
                    )
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])

        return h_est

    def _prepare_for_prediction(self, y, A, noise_covariance):
        """
        Replace the GMM's means and covariance matrices by the means and
        covariance matrices of the observation. Further, in case of diagonal
        matrices, FFT-transform the observation.
        """

        if self.gm.covariance_type == "diag":
            # raise error if A is not identity or quadratic matrix
            try:
                diff = np.sum(np.abs(A - np.eye(A.shape[0])) ** 2)
                if diff > 1e-12:
                    NotImplementedError(
                        f"Estimation for covariance_type = {self.gm.covariance_type} with arbitrary matrix A is not implemented."
                    )
            except:
                raise NotImplementedError(
                    f"Estimation for covariance_type = {self.gm.covariance_type} with arbitrary matrix A is not implemented."
                )

            # update GMM covs
            covs_gm = self.gm.covariances_.copy()
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :] = covs_gm[i, :] + noise_covariance
            self.gm.covariances_ = covs_gm.copy()  # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(
                covs_gm, covariance_type="diag"
            )

            # FFT of observation
            if "block-diag" in self.params:
                y_for_prediction = np.squeeze(self.F2 @ np.expand_dims(y, 2))
            else:
                y_for_prediction = np.fft.fft(y, axis=1) / np.sqrt(y.shape[-1])
        elif self.gm.covariance_type == "full":
            # update GMM means
            Am = np.squeeze(np.matmul(A, np.expand_dims(self.means_cplx, axis=2)))
            # handle the case of only one GMM component
            if Am.ndim == 1:
                self.gm.means_ = Am[None, :]
            else:
                self.gm.means_ = Am

            # update GMM covs
            covs_gm = self.covs_cplx.copy()
            covs_gm = np.matmul(np.matmul(A, covs_gm), A.conj().T)
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :, :] = covs_gm[i, :, :] + noise_covariance
            self.gm.covariances_ = covs_gm.copy()  # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(
                covs_gm, covariance_type="full"
            )

            # update GMM feature number
            self.gm.n_features_in_ = A.shape[0]

            y_for_prediction = y
        else:
            raise NotImplementedError(
                f"Estimation for covariance_type = {self.gm.covariance_type} is not implemented."
            )

        # precompute the inverse matrices
        covs_Cy_inv = np.zeros(
            [self.covs_cplx.shape[0], A.shape[0], A.shape[0]], dtype=complex
        )
        for i in range(self.covs_cplx.shape[0]):
            covs_Cy_inv[i, :, :] = np.linalg.pinv(
                A @ self.covs_cplx[i, :, :] @ A.conj().T + noise_covariance
            )

        return y_for_prediction, covs_Cy_inv

    def _lmmse_formula(self, y, mean_h, cov_h, cov_y_inv, mean_y):
        return mean_h + cov_h @ (cov_y_inv @ (y - mean_y))


def compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty(
            (n_components, n_features, n_features), dtype=complex
        )
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = scilinalg.cholesky(covariance, lower=True)
            except scilinalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = scilinalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T.conj()
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances).conj()
    return precisions_chol
