import numpy as np
import scipy.special as spe
import scipy.linalg as lnl
from kinematic_model import stats


def find_min_z_to_bounds(mu, sd, ab):
    """
    Find absolute z score of mu of each mixture k to its closest bound
    Assume bounds are independent of k, and depend only on dim
    - take minimum over (dimension, bound)
    :param mu: k, d
    :param sd: k, d
    :param ab: d, 2
    :return:
        z: k, | lowest absolute z score from mixture to any bound along any dim
    """
    z = (ab[np.newaxis] - mu[..., np.newaxis]) / sd[..., np.newaxis]
    return np.abs(z).min(axis=(1, 2))


def sample_mixture_gaussians_2d(w, mu, L, n_samples):
    """
    Given k mixtures' parameters, sample
    :param w: mixing probabilities summing to 1
    :param mu: k, 2
    :param L: k, 2, 2 | LL^T = var
        - so (L^-1)' L^-1 = var^-1
        - hence ||L^-1(x-mu)||^2 = (x-mu)'var^-1(x-mu)
    :param n_samples:
    :return:
        inds: n_samples,
        x: n_samples, 2
        nll: n_samples | for importance sampling
    """
    L_inv = make_L_inv_2d(L)
    det_var = (L[:, 0, 0] * L[:, 1, 1]) ** 2

    inds = np.random.choice(w.size, n_samples, p=w)
    noise = np.random.randn(2, n_samples)
    x = mu[inds] + np.einsum('kij,jk->ki', L[inds], noise)

    dif = x[:, np.newaxis] - mu[np.newaxis]  # n_samples, k, 2
    z = np.einsum('kij,skj->ski', L_inv, dif)  # n_samples, k, 2
    nlog_p = .5 * ((z ** 2).sum(axis=2) + np.log(det_var))  # n_samples, k
    # nll = -np.log((np.exp(-log_p) * w).sum(axis=1))
    nll = -spe.logsumexp(-nlog_p, b=w, axis=1)
    return inds, x, nll


def sample_mixture_gaussians_nd(w, mu, L, n_samples, L_inv=None):
    """
    Given k mixtures' parameters, sample
    :param w: mixing probabilities summing to 1
    :param mu: k, n | k n-dimensional gaussian mixture components
    :param L: k, n, n | LL^T = var
        - so (L^-1)' L^-1 = var^-1
        - hence ||L^-1(x-mu)||^2 = (x-mu)'var^-1(x-mu)
    :param n_samples:
    :return:
        inds: n_samples,
        x: n_samples, n
        nll: n_samples | for importance sampling
    """
    if L_inv is None:
        L_inv = make_L_inv_nd(L)
    n = L.shape[1]
    log_det_var = 2 * np.log(L[:, np.arange(n), np.arange(n)]).sum(axis=1)  # k,

    inds = np.random.choice(w.size, n_samples, p=w)
    noise = np.random.randn(n, n_samples)
    x = mu[inds] + np.einsum('kij,jk->ki', L[inds], noise)

    dif = x[:, np.newaxis] - mu[np.newaxis]  # n_samples, k, n
    z = np.einsum('kij,skj->ski', L_inv, dif)  # n_samples, k, n
    nlog_p = .5 * ((z ** 2).sum(axis=2) + log_det_var)  # n_samples, k
    # nll = -np.log((np.exp(-log_p) * w).sum(axis=1))
    nll = -spe.logsumexp(-nlog_p, b=w, axis=1)
    return inds, x, nll


def sample_constrained_mixture(w, mu, sd, ab, n_samples):
    """
    Sample from k-mixture's dimensions independently and use truncated
    univariate normal for bounded dimensions.
    For variance, use only diagonal terms of provided structure.
    Bounds are assumed to depend only on the dimension, and not on the
    mixture component.
    :param w: k,
    :param mu: k, n_dim | dimensions ordered by [:n_con] are constrained, rest are not
    :param sd: k, n_dim | instead of LL^T = var, use only the diagonals (independent approx)
    :param ab: n_con, 2 | [i] = dimension i's bounds [low, high]
        - has n_con >= 1 number of constrained dimensions
        - dimensions [:n_con] treated as constrained, rest are unconstrained
        - (should use normal-only sampler if n_con = 0)
    :param n_samples:
    :return:
        inds: n_samples,
        x: n_samples, n_dim
        nll: n_samples | for importance sampling
    """
    n_k = w.size
    n_dim = mu.shape[1]
    n_con = ab.shape[0]
    n_unc = n_dim - n_con
    inds = np.random.choice(n_k, n_samples, p=w)
    x = np.zeros((n_samples, n_dim))
    if n_unc > 0:
        noise = np.random.randn(n_samples, n_unc)
        x[:, n_con:] = mu[inds, n_con:] + sd[inds, n_con:] * noise  # n_samples, n_unc

    # draw n_con * n_samples truncated rv
    x[:, :n_con] = stats.generate_truncnorm_multi(
        np.broadcast_to(ab[np.newaxis], (n_samples, n_con, 2)).reshape(-1, 2),
        mu[inds, :n_con].reshape(-1), sd[inds, :n_con].reshape(-1)).reshape(n_samples, n_con)

    # need to calculate logpdf for *all* k
    nlog_p = np.zeros((n_samples, n_k, n_dim))
    nlog_p[:, :, :n_con] = -stats.logpdf_truncnorm_multi(
        np.broadcast_to(x[:, np.newaxis, :n_con], (n_samples, n_k, n_con)).reshape(-1),
        np.broadcast_to(ab[np.newaxis, np.newaxis], (n_samples, n_k, n_con, 2)).reshape(-1, 2),
        np.broadcast_to(mu[np.newaxis, :, :n_con], (n_samples, n_k, n_con)).reshape(-1),
        np.broadcast_to(sd[np.newaxis, :, :n_con], (n_samples, n_k, n_con)).reshape(-1)
    ).reshape(n_samples, n_k, n_con)
    if n_unc > 0:
        dif = x[:, np.newaxis, n_con:] - mu[np.newaxis, :, n_con:]  # n_samples, k, n_unc
        z = dif / sd[np.newaxis, :, n_con:]
        nlog_p[:, :, n_con:] = .5 * (z ** 2) + np.log(sd[np.newaxis, :, n_con:])
    nlog_p = nlog_p.sum(axis=2)
    nll = -spe.logsumexp(-nlog_p, b=w, axis=1)
    return inds, x, nll


def sample_gaussians_nd(mu, L, n_samples):
    """
    Given parameters, sample
    :param mu: n | n-dimensional mean vector
    :param L: n, n | LL^T = var
    :param n_samples:
    :return:
        x: n_samples, n
        nll: n_samples | for importance sampling
    """
    n = L.shape[1]
    log_det_L = 2 * np.log(L[np.arange(n), np.arange(n)]).sum()
    z = np.random.randn(n, n_samples)
    x = mu + (L @ z).T  # n_samples, n
    nll = .5 * ((z.T ** 2).sum(axis=1) + log_det_L)  # n_samples,
    return x, nll


def make_L_inv_2d(L):
    """
    :param L: k, 2, 2 | k lower triangular matrices
    :return:
        L_inv: k, 2, 2 | (L^-1), useful for calculating (x-mu)'var^-1(x-mu)
    """
    L_inv = np.zeros_like(L)
    L_inv[:, 0, 0] = 1/L[:, 0, 0]
    L_inv[:, 1, 0] = -L[:, 1, 0]/(L[:, 0, 0] * L[:, 1, 1])
    L_inv[:, 1, 1] = 1/L[:, 1, 1]
    return L_inv


def make_L_inv_nd(L):
    """
    :param L: k, n, n | k lower triangular matrices
    :return:
        L_inv: k, n, n | (L^-1), useful for calculating (x-mu)'var^-1(x-mu)
    """
    # L_inv = np.linalg.inv(L)
    L_inv = np.zeros_like(L)
    for i in range(L.shape[0]):
        L_inv[i] = lnl.solve_triangular(L[i], np.eye(L.shape[1]), lower=True)
    return L_inv


def project_box_constraints(mu, L_inv, con_ind, bounds):
    """
    Project means to box constraints, accounting for covariance structure.
    Assume each constrained dimension has constraints independent of phase k.
    :param mu: k, n | original possibly infeasible means
    :param L_inv: k, n, n | matrices such that
        ||L^-1(x - mu)||^2 = (x - mu)'var^-1(x - mu)
    :param con_ind: n_con, | indices corresponding to constrained dimensions
    :param bounds: n_con, 2 | [low, high] box constraints for constrained dimensions
    :return:
        x_feasible_opt: k, n | optimal feasible means
        dist_opt: k, | .5 * squared Mahalanobis distance of feasible mean from original
    """
    k, n = mu.shape
    n_con = con_ind.size
    con_dim_choices = np.zeros((k, n_con, 3))
    con_dim_choices[..., 0] = mu[:, con_ind]
    con_dim_choices[..., 1:] = bounds[np.newaxis]
    x_test_con = make_combinations_multi_v0(con_dim_choices)  # k, n_con, m
    x_test_con = x_test_con[..., np.isfinite(x_test_con).all((0, 1))]  # k, n_con, m'
    is_feasible = (bounds[np.newaxis, :, [0]] <= x_test_con) & \
                  (x_test_con <= bounds[np.newaxis, :, [1]])
    is_feasible = is_feasible.all(axis=1)  # k, m'
    x_test = np.zeros((k, n, x_test_con.shape[-1]))
    x_test[:] = mu[..., np.newaxis]
    x_test[:, con_ind] = x_test_con
    z = np.einsum('kij,kjm->kim', L_inv, x_test - mu[:, :, np.newaxis])  # k, n, m'
    dist = 0.5 * (z ** 2).sum(axis=1)  # k, m'
    dist[~is_feasible] = np.inf
    min_feasible_inds = np.argmin(dist, axis=1)  # k,
    x_feasible_opt = x_test[np.arange(k), :, min_feasible_inds]  # k, n
    dist_opt = dist[np.arange(k), min_feasible_inds]
    return x_feasible_opt, dist_opt


def make_combinations(x):
    """
    Return all combinations resulting from each possible value at each dimension
    :param x: n_dim, n_choices
    :return:
        c: n_dim, n_choices^n_dim
    """
    n_dim, n_choices = x.shape
    c = np.stack(np.meshgrid(*x[np.arange(n_dim)], indexing='ij'))\
        .reshape(n_dim, -1)
    return c


def make_combinations_multi_v0(x):
    """
    Return combinations of each dimension's choice of value,
    separately along each k.
    :param x: k, n_dim, n_choices
    :return:
        c: k, n_dim, n_choices^n_dim
    """
    k, n_dim, n_choices = x.shape
    c = np.zeros((k, n_dim, n_choices ** n_dim))
    for i in range(k):
        c[i] = make_combinations(x[i])
    return c


# xf, d = project_box_constraints(
#     np.array([[-1., -1.], [-1, 0], [1, 1.]]),
#     np.array([np.eye(2), np.eye(2), np.eye(2)]),
#     np.array([0, ]),
#     np.array([[0, np.inf], ]),
# )
# print(xf)
# print(d)
