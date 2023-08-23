# Gaussian Mixture Model (GMM) Estimator

This repository provides an estimator for the problem of recovering the channel vector $\mathbf{h} \in \mathbb{C}^N$ from the observation $$\mathbf{y} = \mathbf{A} \mathbf{h} + \mathbf{n}$$ where $\mathbf{A} \in \mathbb{C}^{m\times N}$ is the observation matrix and $\mathbf{n} \sim \mathcal{N}_{\mathbb{C}}(\mathbf{0}, \mathbf{\Sigma})$ is a realization of a complex Gaussian zero-mean random variable with known covariance matrix $\mathbf{\Sigma} \in \mathbb{C}^{m\times m}$.
It is assumed that the channel $\mathbf{h} \sim f_{\mathbf{h}}$ is a realization of a random variable with unknown probability density function (PDF) $f_{\mathbf{h}}$.

Using the estimator is a two-step process.

### Step 1: Train the estimator
Training data $\{\mathbf{h}_t\}_{t=1}^{T}$ is used to fit a GMM $$f^{(K)}(\mathbf{h}) = \sum_{k=1}^K p(k) \mathcal{N}_{\mathbb{C}}(\mathbf{h}; \mathbf{\mu}_k, \mathbf{C}_k)$$
where $K$ is the number of GMM components and $p(k)$, $\mathbf{\mu}_k$, and $\mathbf{C}_k$ are the mixing coefficient, mean vector, and covariance matrix of component $k$, respectively.
After the fitting process is done, the GMM $f^{(K)}$ approximates the unknown PDF $f_{\mathbf{h}}$ of the channels.

Step 1 only needs to be done once and it can be done in a preparatory offline training phase.

### Step 2: Estimate channels
Assuming that $f^{(K)}$ is a good approximation of $f_{\mathbf{h}}$, the GMM estimator is the function $g$ which minimizes the mean square error $$\mathrm{E}\big[\| \mathbf{h} - g(\mathbf{y}) \|^2\big].$$
The GMM estimator $g$ can be found in closed form and is implemented in [gmm_estimator.py](https://github.com/michael-koller-91/gmm-estimator/blob/main/gmm_estimator.py).
The file [examples.py](https://github.com/michael-koller-91/gmm-estimator/blob/main/examples.py) demonstrates how [gmm_estimator.py](https://github.com/michael-koller-91/gmm-estimator/blob/main/gmm_estimator.py) can be used. See also the next section below.

For more details, kindly take a look at the first two references below.

## How to use the estimator
The file [examples.py](https://github.com/michael-koller-91/gmm-estimator/blob/main/examples.py) demonstrates the use of the GMM estimator in various settings.
Invoking
```
python example.py
```
runs all examples. Alternatively,
```
python example.py --nr n
```
runs example $n$.
The examples demonstrate the following:
1. The observation matrix $\mathbf{A} = \mathbf{I}$ is the identity matrix and full GMM covariance matrices are used.
2. The observation matrix is a selection matrix and full GMM covariance matrices are used.

## References
The first following two references provide more details and properties of the GMM estimator.

- Koller, Fesl, Turan, Utschick, "An Asymptotically Optimal Approximation of the Conditional Mean Channel Estimator Based on Gaussian Mixture Models," ICASSP, 2022  - [[IEEEXplore](https://ieeexplore.ieee.org/document/9747226)] - [[arXiv](https://arxiv.org/abs/2111.11064)]
- Koller, Fesl, Turan, Utschick, "An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models," IEEE Trans. Signal. Process., 2022 - [[IEEEXplore]](https://ieeexplore.ieee.org/document/9842343) - [[arXiv]](https://arxiv.org/abs/2112.12499)

The estimator has been used in all the following references.
