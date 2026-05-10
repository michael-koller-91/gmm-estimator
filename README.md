# gmm-estimator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Package](https://img.shields.io/badge/package-PyPI-informational.svg)](https://pypi.org/project/gmm-estimator/)

GMM-based estimator for complex-valued linear inverse problems.

`gmm-estimator` provides an estimator for noisy linear observation models using a complex-valued Gaussian mixture model (GMM) prior. The package builds on [cplx-gmm](https://pypi.org/project/cplx-gmm/) for fitting the complex-valued GMM prior and adds an estimation layer for problems of the form `y = A h + n`.

The estimator is domain-independent and can be used for linear inverse problems in signal processing, communications, and related applications. Channel estimation is one motivating application and is discussed in the research background section.

## ✨ Highlights

- GMM-based estimator for complex-valued linear inverse problems
- Supports general linear observation models of the form `y = A h + n`
- Uses complex-valued GMM priors fitted with [cplx-gmm](https://pypi.org/project/cplx-gmm/)
- Component-wise LMMSE estimation under the fitted mixture prior
- Posterior component weighting in the observation domain
- Supports identity and rectangular observation matrices
- Supports full posterior mixture estimates or truncated component sums
- Supports full, diagonal, spherical, circulant, block-circulant, Toeplitz, and block-Toeplitz GMM covariance types
- FFT fast paths for circulant and block-circulant priors under scalar identity observation and noise models
- scikit-learn-like workflow via inherited `fit(...)` and added `estimate(...)`
- Modern Python packaging with `pyproject.toml`, `uv`, `pytest`, and `ruff`

## 📌 Citation

If you use `gmm-estimator` in academic work, please cite the package directly:

```bibtex
@software{koller_fesl_gmm_estimator,
  author = {Koller, Michael and Fesl, Benedikt},
  title = {{gmm-estimator}: GMM-based estimator for complex-valued linear inverse problems},
  year = {2026},
  url = {https://github.com/michael-koller-91/gmm-estimator},
  version = {0.1.0}
}
```

Plain-text citation:

> M. Koller and B. Fesl, `gmm-estimator`: GMM-based estimator for complex-valued linear inverse problems, version 0.1.0. Available: https://github.com/michael-koller-91/gmm-estimator

If you use the estimator in the context of channel estimation, please also consider citing the related works listed in the research background section.

## 📦 Installation

Install from PyPI:

```bash
pip install gmm-estimator
```

or with `uv`:

```bash
uv add gmm-estimator
```

`gmm-estimator` depends on [cplx-gmm](https://pypi.org/project/cplx-gmm/) for fitting complex-valued GMM priors. The dependency is installed automatically when installing `gmm-estimator`.

For development, clone the repository and install the development environment:

```bash
git clone https://github.com/michael-koller-91/gmm-estimator.git
cd gmm-estimator
uv sync --group dev
```

## 🚀 Quick Start

```python
import numpy as np

from gmm_estimator import GmmEstimator

rng = np.random.default_rng(0)

h_train = (
    rng.normal(size=(1_000, 8))
    + 1j * rng.normal(size=(1_000, 8))
) / np.sqrt(2.0)

h_val = (
    rng.normal(size=(100, 8))
    + 1j * rng.normal(size=(100, 8))
) / np.sqrt(2.0)

noise = (
    rng.normal(size=(100, 8))
    + 1j * rng.normal(size=(100, 8))
) / np.sqrt(2.0)

# Identity observation model: y = h + n
noise_covariance = np.eye(8, dtype=complex)
y = h_val + noise

estimator = GmmEstimator(
    n_components=4,
    covariance_type="full",
    random_state=0,
    max_iter=100,
    n_init=1,
)

# Fit the complex-valued GMM prior p(h).
estimator.fit(h_train)

# Estimate h from noisy observations y.
h_est = estimator.estimate(
    y=y,
    noise_covariance=noise_covariance,
    observation_matrix=None,
    n_components_or_probability=1.0,
)
```

The estimator follows a two-step pattern:

1. `fit(h_train)` fits the complex-valued GMM prior using the inherited `cplx-gmm` implementation.
2. `estimate(y, noise_covariance, observation_matrix)` estimates the unknown vector from noisy linear observations.

## 🧩 Estimation Model

The package assumes a noisy linear observation model:

```text
y = A h + n
```

where `h` is the unknown complex-valued vector, `A` is the known observation matrix, and `n` is zero-mean complex Gaussian observation noise with known covariance matrix `Cn`.

Using the estimator is a two-step process.

### Step 1: Fit the GMM prior

Training samples of the unknown vector are used to fit a complex-valued GMM prior:

```text
p(h) = sum_k pi_k CN(h; mu_k, C_k)
```

Here, `K` is the number of GMM components, `pi_k` is the mixture weight, `mu_k` is the component mean vector, and `C_k` is the component covariance matrix. After fitting, the GMM approximates the unknown prior distribution of `h`.

This step only needs to be done once and can be performed in an offline training phase.

### Step 2: Estimate from noisy observations

Given the fitted GMM prior, the estimator approximates the MMSE estimate of `h` from `y`. For each mixture component, the observation-domain model is:

```text
y | k ~ CN(A mu_k, A C_k A^H + Cn)
```

The estimator computes posterior component probabilities in the observation domain and combines component-wise LMMSE estimates. Equivalently, it implements the closed-form GMM-based MMSE estimator under the fitted prior.


## 🧠 Estimator API

The main class is:

```python
from gmm_estimator import GmmEstimator
```

`GmmEstimator` inherits from [`cplx_gmm.GaussianMixtureCplx`](https://pypi.org/project/cplx-gmm/) and therefore supports the same prior-fitting API.

Core methods:

| Method | Description |
|---|---|
| `fit(X, y=None)` | Fit the complex-valued GMM prior. Inherited from `cplx-gmm`. |
| `fit_predict(X, y=None)` | Fit the GMM prior and return component labels. Inherited from `cplx-gmm`. |
| `predict(X)` | Predict the most likely GMM prior component. Inherited from `cplx-gmm`. |
| `predict_proba(X)` | Return prior-domain component probabilities. Inherited from `cplx-gmm`. |
| `score_samples(X)` | Return per-sample log-likelihoods. Inherited from `cplx-gmm`. |
| `score(X, y=None)` | Return the mean log-likelihood. Inherited from `cplx-gmm`. |
| `sample(n_samples=1)` | Draw samples from the fitted GMM prior. Inherited from `cplx-gmm`. |
| `estimate(y, noise_covariance, observation_matrix=None, n_components_or_probability=1.0)` | Estimate unknown vectors from noisy linear observations. |

Constructor parameters are inherited from `GaussianMixtureCplx`:

| Parameter | Description |
|---|---|
| `n_components` | Number of mixture components. |
| `covariance_type` | Covariance type of the complex-valued GMM prior. |
| `tol` | EM convergence tolerance. |
| `reg_covar` | Non-negative regularization added to covariance estimates. |
| `max_iter` | Maximum number of EM iterations. |
| `n_init` | Number of initializations. |
| `init_params` | Initialization method, either `"kmeans"` or `"random"`. |
| `random_state` | Random seed or random state. |
| `warm_start` | Reuse previous solution for supported covariance types. |
| `zero_mean` | If `True`, fit zero-mean mixture components. |
| `blocks` | Block dimensions for block-structured covariance types. |

The fitted prior parameters are exposed using trailing-underscore attributes from `cplx-gmm`:

| Attribute | Description |
|---|---|
| `weights_` | Mixture weights of shape `(n_components,)`. |
| `means_` | Component means of shape `(n_components, n_features)`. |
| `covariances_` | Component covariance parameters. Shape depends on the covariance type. |
| `precisions_` | Component precision parameters. |
| `precisions_cholesky_` | Cholesky factors of component precision parameters. |
| `converged_` | Whether EM converged. |
| `n_iter_` | Number of EM iterations used by the best initialization. |
| `lower_bound_` | Final EM lower bound. |
| `means_fft_` | FFT-domain component means for `covariance_type="circulant"`. |
| `covariances_fft_` | FFT-domain diagonal covariance entries for `covariance_type="circulant"`. |
| `means_fft2_` | Two-dimensional FFT-domain component means for `covariance_type="block-circulant"`. |
| `covariances_fft2_` | Two-dimensional FFT-domain diagonal covariance entries for `covariance_type="block-circulant"`. |

## 🔎 Estimation

The main method added by this package is `estimate(...)`:

```python
h_est = estimator.estimate(
    y=y,
    noise_covariance=noise_covariance,
    observation_matrix=observation_matrix,
    n_components_or_probability=1.0,
)
```

Arguments:

| Argument | Description |
|---|---|
| `y` | Observations of shape `(n_samples, n_observations)`. |
| `noise_covariance` | Observation noise covariance of shape `(n_observations, n_observations)`. |
| `observation_matrix` | Observation matrix of shape `(n_observations, n_features)`. If `None`, the identity matrix is used. |
| `n_components_or_probability` | Component-selection rule for the posterior mixture estimate. |

The component-selection parameter can be used in two ways:

| Value | Behavior |
|---|---|
| Integer, e.g. `1` or `5` | Use the corresponding number of most likely posterior components. |
| Float in `(0, 1]`, e.g. `0.9` | Use the fewest most likely components whose cumulative posterior probability reaches the threshold. |
| `1.0` | Use all components. |

Note that `1` and `1.0` intentionally have different meanings: `1` selects one component, while `1.0` selects all components.

## 🧩 Covariance Types

The estimator supports all covariance types provided by `cplx-gmm`:

| `covariance_type` | Description |
|---|---|
| `"full"` | Full covariance matrix for each GMM component. |
| `"diag"` | Diagonal covariance for each GMM component. |
| `"spherical"` | One scalar variance per GMM component. |
| `"circulant"` | Circulant covariance matrix for each component. |
| `"block-circulant"` | Block-circulant covariance matrix. Requires `blocks=(n_1, n_2)`. |
| `"toeplitz"` | Toeplitz covariance matrix for each component. |
| `"block-toeplitz"` | Block-Toeplitz covariance matrix. Requires `blocks=(n_1, n_2)`. |

For block-structured covariance types, pass `blocks` to the constructor:

```python
estimator = GmmEstimator(
    n_components=4,
    covariance_type="block-circulant",
    blocks=(4, 8),
    random_state=0,
)

estimator.fit(h_train)
```

## ⚡ FFT Acceleration

For `covariance_type="circulant"` and `covariance_type="block-circulant"`, `cplx-gmm` fits the prior in a Fourier-domain representation and stores the corresponding fitted FFT-domain parameters.

`gmm-estimator` uses these attributes for fast estimation when the observation model satisfies the required structure:

```text
A = α I
Cn = σ² I
```

where `α` and `σ²` are scalar values. In this case, the posterior component probabilities and component-wise LMMSE estimates can be computed in the Fourier domain using diagonal covariance entries.

If the fast-path assumptions are not satisfied, the estimator automatically falls back to the generic full-covariance implementation.

## 🧪 Examples

Run the example script:

```bash
uv run python examples/gmm_estimator_examples.py --nr 1
```

Available examples:

| Number | Description |
|---|---|
| `1` | Full covariance matrices with identity observation matrix. |
| `2` | Full covariance matrices with selection observation matrix. |
| `3` | Circulant covariance matrices with identity observation matrix. |
| `4` | Circulant covariance matrices with selection observation matrix. |
| `5` | Block-circulant covariance matrices with identity observation matrix. |
| `6` | Toeplitz covariance matrices with identity observation matrix. |
| `7` | Block-Toeplitz covariance matrices with identity observation matrix. |

Run all examples:

```bash
uv run python examples/gmm_estimator_examples.py
```

Use `--help` to inspect the script interface:

```bash
uv run python examples/gmm_estimator_examples.py --help
```

## 📚 Research Background

This repository is joint work of [Michael Koller](https://github.com/michael-koller-91) and [Benedikt Fesl](https://github.com/benediktfesl).

### Papers

The following reference provides more details and properties of the GMM estimator.

- Koller, Fesl, Turan, Utschick, "An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models," *IEEE Trans. Signal. Process.*, 2022. [[IEEEXplore]](https://ieeexplore.ieee.org/document/9842343) [[arXiv]](https://arxiv.org/abs/2112.12499)

The estimator implementation has been used in the following references.

- N. Turan, B. Fesl, M. Grundei, M. Koller, and W. Utschick, “Evaluation of a Gaussian Mixture Model-based Channel Estimator using Measurement Data,” in *Int. Symp. Wireless Commun. Syst. (ISWCS)*, 2022. [[IEEEXplore]](https://ieeexplore.ieee.org/abstract/document/9940363) [[arXiv]](https://arxiv.org/abs/2207.14150)

- B. Fesl, M. Joham, S. Hu, M. Koller, N. Turan, and W. Utschick, “Channel Estimation based on Gaussian Mixture Models with Structured Covariances,” in *56th Asilomar Conf. Signals, Syst., Comput.*, 2022, pp. 533–537.
[[IEEEXplore]](https://ieeexplore.ieee.org/abstract/document/10051921) [[arXiv]](https://arxiv.org/abs/2205.03634)

- B. Fesl, N. Turan, M. Joham, and W. Utschick, “Learning a Gaussian Mixture Model from Imperfect Training Data for Robust Channel Estimation,” *IEEE Wireless Commun. Lett.*, 2023. [[IEEEXplore]](https://ieeexplore.ieee.org/abstract/document/10078293) [[arXiv]](https://arxiv.org/abs/2301.06488)

- M. Koller, B. Fesl, N. Turan and W. Utschick, "An Asymptotically Optimal Approximation of the Conditional Mean Channel Estimator Based on Gaussian Mixture Models," *IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)*, 2022, pp. 5268-5272. [[IEEEXplore]](https://ieeexplore.ieee.org/document/9747226) [[arXiv]](https://arxiv.org/abs/2111.11064)

- B. Fesl, A. Faika, N. Turan, M. Joham, and W. Utschick, “Channel Estimation with Reduced Phase Allocations in RIS-Aided Systems,” in *IEEE 24th Int. Workshop Signal Process. Adv. Wireless Commun. (SPAWC)*, 2023, pp. 161-165. [[IEEEXplore]](https://ieeexplore.ieee.org/document/10304464) [[arXiv]](https://arxiv.org/abs/2211.07552)

- N. Turan, B. Fesl, M. Koller, M. Joham and W. Utschick, "A Versatile Low-Complexity Feedback Scheme for FDD Systems via Generative Modeling," in *IEEE Trans. Wireless Commun.*, vol. 23, no. 6, pp. 6251-6265, 2024. [[IEEEXplore]](https://ieeexplore.ieee.org/document/10318056) [[arXiv]](https://arxiv.org/abs/2304.14373)

- N. Turan, B. Fesl, and W. Utschick, "Enhanced Low-Complexity FDD System Feedback with Variable Bit Lengths via Generative Modeling," in *57th Asilomar Conf. Signals, Syst., Comput.*, 2023. [[IEEEXplore]](https://ieeexplore.ieee.org/document/10477075) [[arXiv]](https://arxiv.org/abs/2305.03427)

- N. Turan, M. Koller, B. Fesl, S. Bazzi, W. Xu and W. Utschick, "GMM-based Codebook Construction and Feedback Encoding in FDD Systems," in *56th Asilomar Conf. Signals, Syst., Comput.*, 2022, pp. 37-42. [[IEEEXplore]](https://ieeexplore.ieee.org/abstract/document/10052020) [[arXiv]](https://arxiv.org/abs/2205.12002)

- M. Koller, “Asymptotically Optimal Channel Estimation and Learning Suitable Compressive Sensing Matrices,” *Ph.D. dissertation, Technical University of Munich*, 2022. [[TUM]](https://mediatum.ub.tum.de/1659309)

- B. Fesl, “Generative Model-Aided Channel Estimation Design and Optimality Analysis,” *Ph.D. dissertation, Technical University of Munich*, 2025. [[TUM]](https://mediatum.ub.tum.de/?id=1748775)

## 🧪 Development

Install the development environment with `uv`:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Format code:

```bash
uv run ruff format .
```

Run the examples:

```bash
uv run python examples/gmm_estimator_examples.py
```

Build the package:

```bash
uv run python -m build
```

Check the package distribution:

```bash
uv run twine check dist/*
```

## ✅ Test Coverage

The test suite covers:

- package imports
- integration with the `cplx-gmm` package
- estimation with identity observation matrices
- estimation with rectangular observation matrices
- estimation with complex-valued observation matrices
- real-valued input handling
- posterior probability normalization
- component-count selection
- cumulative-probability selection
- compact covariance representations
- full estimator output against manual reference implementations
- FFT fast-path equivalence for circulant covariance matrices
- FFT fast-path equivalence for block-circulant covariance matrices
- unfitted-estimator behavior
- invalid input validation
- example script execution

## 📄 License

This project is licensed under the [BSD 3-Clause License](LICENSE).
