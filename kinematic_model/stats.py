import numpy as np
from scipy import special
from scipy import stats as ss


EPS = 1e-8


def generate_truncnorm_multi(ab, mu, sd):
    """
    Use inverse-cdf method.
    Note:
    - difficult to generate x for which mu is many sd away (same sign) from [a, b]
    - in this case, around 5 sd, cdf_ab = (0, 0) or (1, 1)
    - best solution: avoid such inputs
    :param ab: n, 2 | [i] = [a_i, b_i], and a_i < b_i, finite
    :param mu: n,
    :param sd: n,
    :return:
        x: n, | [i] = sample from truncated normal(mu_i, sd_i, ab_i)
    """
    p = np.random.rand(mu.size)
    cdf_ab = ss.norm.cdf(ab, mu[:, np.newaxis], sd[:, np.newaxis])
    z = cdf_ab[:, 0] + p * (cdf_ab[:, 1] - cdf_ab[:, 0])
    z[1-EPS < z] = 1-EPS
    z[z < EPS] = EPS
    x = mu + sd * np.sqrt(2) * special.erfinv(2 * z - 1)
    return x


def logpdf_truncnorm_multi(x, ab, mu, sd):
    """

    :param x: n,
    :param ab: n, 2 | [i] = [a_i, b_i], and a_i < b_i, finite
    :param mu: n,
    :param sd: n,
    :return:
    """
    cdf_ab = ss.norm.cdf(ab, mu[:, np.newaxis], sd[:, np.newaxis])
    delta = cdf_ab[:, 1] - cdf_ab[:, 0]
    delta[delta < EPS] = EPS
    return ss.norm.logpdf(x, mu, sd) - np.log(delta)


def generate_truncnorm(ab, mu, sd, n_samples):
    """
    Use inverse-cdf method in which inv-cdf of truncated normal can be written
    in terms of the normal's cdf and inv-cdf.
    see: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    :param ab: 2, | [a, b], and a < b, finite
    :param mu: scalar
    :param sd: scalar
    :param n_samples:
    :return:
        x: n_samples,
    """
    p = np.random.rand(n_samples)
    cdf_ab = ss.norm.cdf(ab, mu, sd)
    z = cdf_ab[0] + p * (cdf_ab[1] - cdf_ab[0])
    # x = ss.norm.ppf(z, mu, sd)  # t = 0.000403
    x = mu + sd * np.sqrt(2) * special.erfinv(2 * z - 1)  # t = 0.00018
    return x


if __name__ == '__main__':
    from timeit import timeit

    # print(generate_truncnorm([0, 2], 1.5, .5, 10))
    ab = np.array([[0, 1], [1, 2], [2, 3]])
    mu = np.array([0, 0, 0])
    sd = np.array([1, 1, 1])
    x = generate_truncnorm_multi(ab, mu, sd)
    nll = -logpdf_truncnorm_multi(x, ab, mu, sd)
    print(x)
    print(nll)
    print()

    args = ([0, 2], 1, .5, 1000)
    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=generate_truncnorm, args=args))/n_tries)
