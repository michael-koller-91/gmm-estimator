"""Examples for GMM-based estimator."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from gmm_estimator import GmmEstimator

ComplexArray = npt.NDArray[np.complexfloating]


def mse(x: ComplexArray, y: ComplexArray) -> float:
    """Compute the mean squared error between two complex-valued arrays."""
    return float(np.mean(np.abs(x - y) ** 2))


def standard_normal_cplx(
    rng: np.random.Generator,
    n_samples: int,
    n_dim: int,
) -> ComplexArray:
    """Draw standard circularly symmetric complex Gaussian samples."""
    return (
        rng.standard_normal((n_samples, n_dim))
        + 1j * rng.standard_normal((n_samples, n_dim))
    ) / np.sqrt(2)


def selection_matrix(
    rng: np.random.Generator,
    n_observations: int,
    n_features: int,
) -> ComplexArray:
    """Create a random row-selection matrix."""
    selected_indices = np.sort(
        rng.choice(n_features, size=n_observations, replace=False)
    )

    matrix = np.zeros((n_observations, n_features), dtype=complex)
    matrix[np.arange(n_observations), selected_indices] = 1.0

    return matrix


def evaluate_estimator(
    estimator: GmmEstimator,
    x_eval: ComplexArray,
    y_eval: ComplexArray,
    noise_covariance: ComplexArray,
    observation_matrix: ComplexArray | None = None,
) -> None:
    """Evaluate an estimator with all components and with the top three components."""
    tic = time.perf_counter()

    x_est = estimator.estimate(
        y=y_eval,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
        n_components_or_probability=1.0,
    )
    print(f"MSE with n_components_or_probability=1.0: {mse(x_est, x_eval):.6f}")

    x_est = estimator.estimate(
        y=y_eval,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
        n_components_or_probability=3,
    )
    print(f"MSE with n_components_or_probability=3: {mse(x_est, x_eval):.6f}")

    toc = time.perf_counter()
    print(f"Estimation done. ({toc - tic:.3f} s)")


def fit_estimator(
    x_train: ComplexArray,
    covariance_type: str,
    blocks: tuple[int, int] | None = None,
    max_iter: int = 100,
) -> GmmEstimator:
    """Fit a GMM estimator."""
    tic = time.perf_counter()

    estimator = GmmEstimator(
        n_components=16,
        covariance_type=covariance_type,
        blocks=blocks,
        random_state=2,
        max_iter=max_iter,
        n_init=1,
    )
    estimator.fit(x_train)

    toc = time.perf_counter()
    print(f"Training done. ({toc - tic:.3f} s)")

    return estimator


def run_identity_observation_example(
    title: str,
    covariance_type: str,
    n_features: int,
    blocks: tuple[int, int] | None = None,
    max_iter: int = 100,
) -> None:
    """Run an example with identity observation matrix."""
    print(title)

    rng = np.random.default_rng(1235428719812346)
    n_train = 1_000
    n_eval = 100

    x_train = standard_normal_cplx(rng, n_train, n_features)
    estimator = fit_estimator(
        x_train=x_train,
        covariance_type=covariance_type,
        blocks=blocks,
        max_iter=max_iter,
    )

    x_eval = standard_normal_cplx(rng, n_eval, n_features)
    noise = standard_normal_cplx(rng, n_eval, n_features)
    noise_covariance = np.eye(n_features, dtype=complex)

    y_eval = x_eval + noise

    evaluate_estimator(
        estimator=estimator,
        x_eval=x_eval,
        y_eval=y_eval,
        noise_covariance=noise_covariance,
    )


def run_selection_observation_example(
    title: str,
    covariance_type: str,
    n_features: int,
    n_observations: int,
    blocks: tuple[int, int] | None = None,
    max_iter: int = 100,
) -> None:
    """Run an example with a row-selection observation matrix."""
    print(title)

    rng = np.random.default_rng(1235428719812346)
    n_train = 1_000
    n_eval = 100

    observation_matrix = selection_matrix(
        rng=rng,
        n_observations=n_observations,
        n_features=n_features,
    )

    x_train = standard_normal_cplx(rng, n_train, n_features)
    estimator = fit_estimator(
        x_train=x_train,
        covariance_type=covariance_type,
        blocks=blocks,
        max_iter=max_iter,
    )

    x_eval = standard_normal_cplx(rng, n_eval, n_features)
    noise = standard_normal_cplx(rng, n_eval, n_observations)
    noise_covariance = np.eye(n_observations, dtype=complex)

    y_eval = x_eval @ observation_matrix.T + noise

    evaluate_estimator(
        estimator=estimator,
        x_eval=x_eval,
        y_eval=y_eval,
        noise_covariance=noise_covariance,
        observation_matrix=observation_matrix,
    )


def example1() -> None:
    """Run full covariance example with identity observation matrix."""
    run_identity_observation_example(
        title="Full covariance matrices with A = I.",
        covariance_type="full",
        n_features=10,
    )


def example2() -> None:
    """Run full covariance example with selection observation matrix."""
    run_selection_observation_example(
        title="Full covariance matrices with selection matrix A.",
        covariance_type="full",
        n_features=10,
        n_observations=5,
    )


def example3() -> None:
    """Run circulant covariance example with identity observation matrix."""
    run_identity_observation_example(
        title="Circulant covariance matrices with A = I.",
        covariance_type="circulant",
        n_features=10,
    )


def example4() -> None:
    """Run circulant covariance example with selection observation matrix."""
    run_selection_observation_example(
        title="Circulant covariance matrices with selection matrix A.",
        covariance_type="circulant",
        n_features=10,
        n_observations=5,
    )


def example5() -> None:
    """Run block-circulant covariance example with identity observation matrix."""
    run_identity_observation_example(
        title="Block-circulant covariance matrices with A = I.",
        covariance_type="block-circulant",
        n_features=12,
        blocks=(4, 3),
    )


def example6() -> None:
    """Run Toeplitz covariance example with identity observation matrix."""
    run_identity_observation_example(
        title="Toeplitz covariance matrices with A = I.",
        covariance_type="toeplitz",
        n_features=10,
    )


def example7() -> None:
    """Run block-Toeplitz covariance example with identity observation matrix."""
    run_identity_observation_example(
        title="Block-Toeplitz covariance matrices with A = I.",
        covariance_type="block-toeplitz",
        n_features=12,
        blocks=(3, 4),
        max_iter=200,
    )


EXAMPLES: dict[int, Callable[[], None]] = {
    1: example1,
    2: example2,
    3: example3,
    4: example4,
    5: example5,
    6: example6,
    7: example7,
}


def main() -> None:
    """Run one example or all examples."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nr",
        help="Run a specific example. Runs all examples by default.",
        type=int,
        choices=[0, *EXAMPLES.keys()],
        default=0,
    )
    args = parser.parse_args()

    if args.nr > 0:
        print(f"Running example {args.nr}.")
        EXAMPLES[args.nr]()
        return

    for nr, example in EXAMPLES.items():
        if nr > 1:
            print()
        print(f"Running example {nr}.")
        example()


if __name__ == "__main__":
    main()
