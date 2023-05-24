import scipy
import numpy as np
from numpy.random import default_rng
from scipy.spatial import distance


def exponentiated_quadratic(xa, xb, l=3.0, sigma=1.0):
    """Exponentiated quadratic  with l and sigma"""
    # L2 distance (Squared Euclidian)
    # l = 3.0
    # sigma = 1.0
    sq_norm = -0.5 * distance.cdist(xa, xb, 'sqeuclidean') / (2 * l ** 2)
    # sq_norm = -0.5 * ()
    return (sigma ** 2) * np.exp(sq_norm)


# Gaussian process posterior
def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    sig11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    sig12 = kernel_func(X1, X2)
    # Solve
    solved = scipy.linalg.solve(sig11, sig12, assume_a='pos').T
    # Compute posterior mean
    mu2 = solved @ y1
    # mu2 = np.zeros_like(X2[:, 0])
    # Compute the posterior covariance
    sig22 = kernel_func(X2, X2)
    sig2 = sig22 - (solved @ sig12)
    return mu2[:, 0], sig2  # mean, covariance


def sample_gp(x, y):
    # Compute the posterior mean and covariance
    X1 = np.expand_dims(np.array([x[0], x[-1]]), axis=1)
    y1 = np.expand_dims(y, axis=1)
    X2 = np.expand_dims(x[1:-1], axis=1)

    # Compute posterior mean and covariance
    mu2, sig2 = GP(X1, y1, X2, exponentiated_quadratic)

    # Compute the standard deviation at the test points to be plotted
    sig2 = np.sqrt(sig2)

    # Draw samples of the posterior
    y2 = np.random.multivariate_normal(mean=mu2, cov=sig2, size=X2.shape[1])[0]
    X2 = X2[:, 0]

    X = np.concatenate((X1[0], X2, X1[1]))
    y = np.concatenate((y1[0], y2, y1[1]))
    return X, y


def sample_gp_prior(x, t, size):
    xx, tt = np.meshgrid(x, t)
    inputs = np.stack((xx, tt), axis=2)
    inputs = inputs.reshape((inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
    sigma = exponentiated_quadratic(inputs, inputs, l=10)
    mu = np.zeros(sigma.shape[0])
    rng = default_rng()
    y = rng.multivariate_normal(mean=mu, cov=sigma, check_valid='ignore', size=(sigma.shape[1], size), method='eigh')[0]
    y = y.reshape((size, t.shape[0], x.shape[0]))
    return y


def sample_gp_2d(x, y, size):
    xx, yy = np.meshgrid(x, y)
    inputs = np.stack((xx, yy), axis=2)
    inputs = inputs.reshape((inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
    sigma = exponentiated_quadratic(inputs, inputs, l=0.1)
    mu = np.zeros(sigma.shape[0])
    rng = default_rng()
    res = rng.multivariate_normal(mean=mu, cov=sigma, check_valid='ignore', size=(sigma.shape[1], size), method='eigh')[0]
    res = res.reshape((size, x.shape[0], y.shape[0]))
    return res