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
                    mean_h = self.means_cplx[labels[yi]]
                    h_est[yi, :] = self._lmmse_formula(
                        y_for_prediction[yi, :],
                        mean_h,
                        self.covs_cplx[labels[yi]] @ A.conj().T,
                        covs_Cy_inv[labels[yi]],
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
                            y_for_prediction[yi, :],
                            mean_h,
                            self.covs_cplx[argproba] @ A.conj().T,
                            covs_Cy_inv[argproba],
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
                        y_for_prediction[yi, :],
                        mean_h,
                        self.covs_cplx[argproba] @ A.conj().T,
                        covs_Cy_inv[argproba],
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
                        y_for_prediction[yi, :],
                        mean_h,
                        self.covs_cplx[argproba] @ A.conj().T,
                        covs_Cy_inv[argproba],
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
        # check if FFT speed-ups and diagonal covs can be used for circulant covariances
        if A.shape[0] == A.shape[1]:
            diag_tf = np.isclose(np.sum(np.abs(A - A[0,0]*np.eye(A.shape[0])) ** 2), 0.0)
            diag_tf = diag_tf and np.isclose(np.sum(np.abs(noise_covariance -
                                               noise_covariance[0,0]*np.eye(noise_covariance.shape[0])) ** 2), 0.0)
        else:
            diag_tf = False
        if (self.params['cov_type'] == 'circulant' or self.params['cov_type'] == 'block-circulant') and diag_tf:
            # update GMM covs by transformation to Fourier domain where cov is diag
            y_for_prediction = self._dft_trafo(y)
            # update means
            self.gm.means_ = np.diag(A) * self.means_cplx
            # update covs
            covs_gm = np.zeros([self.covs_cplx.shape[0], y.shape[-1]], dtype=complex)
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :] = self.covs_cplx[i, :] + np.diag(noise_covariance)
            self.gm.covariances_ = covs_gm.copy()  # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(
                covs_gm, covariance_type="diag"
            )
            # pre-compute inverse covs of observations
            covs_Cy_inv = 1 / covs_gm
            self.params['dft_trafo'] = True

        elif self.gm.covariance_type == "full":
            # update GMM means
            Am = np.squeeze(np.matmul(A, np.expand_dims(self.means_cplx, axis=2)))
            # handle the case of only one GMM component
            if Am.ndim == 1:
                self.gm.means_ = Am[None, :]
            else:
                self.gm.means_ = Am

            # update GMM covs
            #covs_gm = self.covs_cplx.copy()
            covs_gm = np.matmul(np.matmul(A, self.covs_cplx), A.conj().T)
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :, :] = covs_gm[i, :, :] + noise_covariance
            self.gm.covariances_ = covs_gm.copy()  # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(
                covs_gm, covariance_type="full"
            )

            # update GMM feature number
            self.gm.n_features_in_ = A.shape[0]

            y_for_prediction = y.copy()
            # pre-compute inverse covs of observations
            covs_Cy_inv = np.linalg.pinv(covs_gm, hermitian=True)
        else:
            raise NotImplementedError(
                f"Estimation for covariance_type = {self.gm.covariance_type} is not implemented."
            )

        return y_for_prediction, covs_Cy_inv


    def _lmmse_formula(self, y, mean_h, cov_h, cov_y_inv, mean_y):
        len_cy = len(cov_y_inv.shape)
        if self.params['cov_type'] == 'circulant' and len_cy == 1:
            hest = mean_h + cov_h * (cov_y_inv * (y - mean_y))
            return np.fft.ifft(hest, axis=0) * np.sqrt(hest.shape[0])
        elif self.params['cov_type'] == 'block-circulant' and len_cy == 1:
            hest = mean_h + cov_h * (cov_y_inv * (y - mean_y))
            return np.squeeze(self.F2.conj().T @ hest[:, None])
        else:
            hest = mean_h + cov_h @ (cov_y_inv @ (y - mean_y))
            return hest


    def _dft_trafo(self, y):
        """
        If not done yet, transform GMM parameters in Fourier domain via DFT (FFTs) and transform observations.
        """
        if not self.params['dft_trafo']:
            self.gm.covariance_type = "diag"
            if self.params['cov_type'] == 'circulant':
                dft_matrix = np.fft.fft(np.eye(self.means_cplx.shape[-1], dtype=complex)) / np.sqrt(self.means_cplx.shape[-1])
                self.gm.means_ = np.fft.fft(self.means_cplx, axis=1) / np.sqrt(self.means_cplx.shape[-1])
                self.means_cplx = self.gm.means_.copy()
                # diagonalize (block-)circulant cov in Fourier domain
                self.covs_cplx = np.zeros(self.covs_cplx.shape[:-1], dtype=complex)
                for i in range(self.covs_cplx.shape[0]):
                    self.covs_cplx[i] = np.diag(dft_matrix @ self.gm.covariances_[i] @ dft_matrix.conj().T)
                self.gm.covariances_ = self.covs_cplx.copy()
            elif self.params['cov_type'] == 'block-circulant':
                self.gm.means_ = np.squeeze(np.matmul(self.F2, np.expand_dims(self.means_cplx, axis=2)))
                self.means_cplx = self.gm.means_.copy()
                # diagonalize (block-)circulant cov in Fourier domain
                self.covs_cplx = np.zeros(self.covs_cplx.shape[:-1], dtype=complex)
                for i in range(self.covs_cplx.shape[0]):
                    self.covs_cplx[i] = np.diag(self.F2 @ self.gm.covariances_[i] @ self.F2.conj().T)
                self.gm.covariances_ = self.covs_cplx.copy()
            else:
                NotImplementedError('Not implemented!')
        if self.params['cov_type'] == 'circulant':
            y_for_prediction = np.fft.fft(y, axis=1) / np.sqrt(y.shape[-1])
        else:
            y_for_prediction = np.squeeze(self.F2 @ np.expand_dims(y, 2))
        return y_for_prediction

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

