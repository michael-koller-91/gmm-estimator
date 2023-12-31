import time
import numpy as np
from gmm_estimator import GmmEstimator
import random


def mse(x, y):
    """Compute the mean square error between x and y."""
    return np.sum(np.abs(x - y) ** 2) / x.size


def standard_normal_cplx(n_samples, n_dim, rng=np.random.default_rng()):
    """
    Standard complex normal random numbers of shape (n_samples, n_dim).
    """
    return (
        rng.standard_normal((n_samples, n_dim))
        + 1j * rng.standard_normal((n_samples, n_dim))
    ) / np.sqrt(2)


def example1():
    """
    Full covariance matrices with A = I.
    """
    print('Full covariance matrices with A = I.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 10

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type="full",
    )
    gmm_estimator.fit(h_train)

    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_covariance = np.eye(n_dim, dtype=complex)
    # noisy observations, y = h + n
    y_eval = h_eval + noise_eval

    #
    # evaluate GMM estimator
    #
    tic = time.time()

    # use all components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    # use the top 3 components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 1


def example2():
    """
    Full covariance matrices with selection matrix A.
    """
    print('Full covariance matrices with selection matrix A.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 10
    n_dim_obs = 5

    #
    # create a random selection matrix
    #
    A = np.zeros([n_dim_obs, n_dim])
    pattern_vec = random.sample(range(n_dim), n_dim_obs)
    pattern_vec.sort()
    for i, val in enumerate(pattern_vec):
        A[i, val] = 1

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type="full",
    )
    gmm_estimator.fit(h_train)
    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim_obs, rng)
    noise_covariance = np.eye(n_dim_obs, dtype=complex)
    # noisy observations, y = A h + n
    y_eval = np.squeeze(np.matmul(A, np.expand_dims(h_eval, 2))) + noise_eval

    #
    # GMM evaluation
    #
    tic = time.time()

    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, A=A, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, A=A, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 2


def example3():
    """
    Circulant covariance matrices with A = I.
    """
    print('Circulant covariance matrices with A = I.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 10

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type="circulant",
    )
    gmm_estimator.fit(h_train)

    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_covariance = np.eye(n_dim, dtype=complex)
    # noisy observations, y = h + n
    y_eval = h_eval + noise_eval

    #
    # evaluate GMM estimator
    #
    tic = time.time()

    # use all components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    # use the top 3 components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 3


def example4():
    """
    Circulant covariance matrices with selection matrix A.
    """
    print('Circulant covariance matrices with selection matrix A.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 10
    n_dim_obs = 5

    #
    # create a random selection matrix
    #
    A = np.zeros([n_dim_obs, n_dim])
    pattern_vec = random.sample(range(n_dim), n_dim_obs)
    pattern_vec.sort()
    for i, val in enumerate(pattern_vec):
        A[i, val] = 1

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type="circulant",
    )
    gmm_estimator.fit(h_train)
    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim_obs, rng)
    noise_covariance = np.eye(n_dim_obs, dtype=complex)
    # noisy observations, y = A h + n
    y_eval = np.squeeze(np.matmul(A, np.expand_dims(h_eval, 2))) + noise_eval

    #
    # GMM evaluation
    #
    tic = time.time()

    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, A=A, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, A=A, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 4


def example5():
    """
    Block-circulant covariance matrices with A = I.
    """
    print('Block-circulant covariance matrices with A = I.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 12

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type="block-circulant",
    )
    gmm_estimator.fit(h_train, blocks=(4, 3))

    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_covariance = np.eye(n_dim, dtype=complex)
    # noisy observations, y = h + n
    y_eval = h_eval + noise_eval

    #
    # evaluate GMM estimator
    #
    tic = time.time()

    # use all components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    # use the top 3 components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 5


def example6():
    """
    Toeplitz covariance matrices with A = I.
    """
    print('Toeplitz covariance matrices with A = I.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 10

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type="toeplitz",
    )
    gmm_estimator.fit(h_train)

    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_covariance = np.eye(n_dim, dtype=complex)
    # noisy observations, y = h + n
    y_eval = h_eval + noise_eval

    #
    # evaluate GMM estimator
    #
    tic = time.time()

    # use all components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    # use the top 3 components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 6


def example7():
    """
    Block-Toeplitz covariance matrices with A = I.
    """
    print('Block-Toeplitz covariance matrices with A = I.')
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_eval = 100
    n_dim = 12

    #
    # training data
    #
    h_train = standard_normal_cplx(n_train, n_dim, rng)

    #
    # train GMM estimator
    #
    tic = time.time()

    gmm_estimator = GmmEstimator(
        n_components=16,
        random_state=2,
        max_iter=200,
        n_init=1,
        covariance_type="block-toeplitz",
    )
    gmm_estimator.fit(h_train, blocks=(3, 4))

    toc = time.time()
    print(f"training done. ({toc - tic:.3f} s)")

    #
    # data to evaluate the GMM estimator
    #
    h_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_eval = standard_normal_cplx(n_eval, n_dim, rng)
    noise_covariance = np.eye(n_dim, dtype=complex)
    # noisy observations, y = h + n
    y_eval = h_eval + noise_eval

    #
    # evaluate GMM estimator
    #
    tic = time.time()

    # use all components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=1.0
    )
    print("MSE with n_components_or_probability=1.0:", mse(h_est, h_eval))
    del h_est

    # use the top 3 components for the evaluation
    h_est = gmm_estimator.estimate(
        y_eval, noise_covariance, n_dim, n_components_or_probability=3
    )
    print("MSE with n_components_or_probability=3:", mse(h_est, h_eval))
    del h_est

    toc = time.time()
    print(f"estimation done. ({toc - tic:.3f} s)")

    return 7


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nr",
        help="Run a specific example or run all examples by default.",
        type=int,
        default=0,
    )
    parargs = parser.parse_args()

    if parargs.nr > 0:
        # run a specific example
        print(f"Running example {parargs.nr}.")
        eval(f"example{parargs.nr}()")
    else:
        # run all examples
        for nr in range(1, 8):
            if nr > 1:
                print()
            print(f"Running example {nr}.")
            eval(f"example{nr}()")