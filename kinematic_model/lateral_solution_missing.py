"""
Lateral solution for missing data case (observe only some p_i)
- __ -> v2: update sampling used to account for constraints earlier

theta = (x_1; ...; x_n; p_merge)
"""
import numpy as np
import kinematic_model.mixture_sampling as smp
import kinematic_model.kalman as ka

DT = 0.1
LARGE_V = 1e3  # [m/s] for ~ inf in sampling truncated normals
LARGE_NLL = 1e8
MAX_Z_DIST = 4.
KF_EPS = 1e-8
KF_LARGE_VAR = 1e8


def main_sample_solutions():
    seed = np.random.randint(1000)
    seed = 461
    print('seed: {}'.format(seed))
    np.random.seed(seed)

    k = 120
    p_merge = 3.
    p0 = 0.
    v0 = 0.0
    sd_dv = 0.05
    n_samples = 200

    n_obs = 30
    k_min = 5
    k_str = 100
    p_merge_bounds = p_merge + np.array([-2, 2])
    mu_p = p_merge
    sd_p = 1.5
    n_steps = 200

    k_grid = np.mgrid[k_min:130+1:5]
    # k_grid = np.array([80, 120])
    # is_obs = (np.arange(n_obs) < 6)
    is_obs = (np.arange(n_obs) < 6) | (25 < np.arange(n_obs))
    # is_obs = (np.arange(n_obs) < 10) | (20 < np.arange(n_obs))
    # is_obs = (np.arange(n_obs) < 15) | (17 < np.arange(n_obs))
    is_obs[-1] = True  # ensure last is observed (this is assumed for prop, and in sampling)

    x_true = np.zeros((1 + n_steps, 2))
    x_true[0] = [p0, v0]
    x_true[1:] = forward_kpv_samples_v0(
        p0, np.array([[k, p_merge, v0]]), sd_dv, k_str, 1, n_steps)[..., 0]
    p_obs = x_true[:n_obs, 0].copy()
    p_obs[~is_obs] = np.nan  # bad value - 'not seen', but [n_obs-1] is

    kpv_samples, nll = sample_merge_kpv_v0(
        p_obs, is_obs, k_grid, sd_dv, mu_p, sd_p, p_merge_bounds, k_str, n_samples)
    x_hat = forward_kpv_samples_v0(
        p_obs[-1], kpv_samples, sd_dv, k_str, n_obs, n_steps)

    np.set_printoptions(suppress=True)
    print(np.hstack((kpv_samples, nll[:, np.newaxis])).round(2)[:22])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(np.arange(x_true.shape[0]), x_true[:, 0], alpha=0.5, color='black', marker='+', label='true')
    ax.plot(np.arange(n_obs)[is_obs], x_true[:n_obs, 0][is_obs],
            alpha=0.8, color='orange', marker='+', label='observed', ls='')
    ax.plot(n_obs + np.arange(x_hat.shape[0]), x_hat[:, 0], alpha=0.1, color='blue')
    plt.show()


def forward_kpv_samples_v0(p0, kpv_samples, sd_dv, k_str, n_obs, n_steps):
    """
    - quasi-switching: once initial k ends, switch to k_str
    :param p0: last observed lat p
    :param kpv_samples: n_samples, 3 | [i] = [k, p_merge, v]
        - v at last observed step
    :param sd_dv:
    :param k_str: if k < 3 switch to straight with constant duration k_str
    :param n_obs:
    :param n_steps:
    :return:
        x_hat: n_steps, 2, n_samples | [i, :, j] = lat [p, v]
    """
    n_samples = kpv_samples.shape[0]
    x_hat = np.zeros((1 + n_steps, 2, n_samples))
    x_hat[0, 0] = p0
    x_hat[0, 1] = kpv_samples[:, 2]
    for i in range(n_steps):
        k_i = kpv_samples[:, 0] - (n_obs - 1) - i
        k_i[k_i < 3] = k_str
        AAt = np.zeros((4, n_samples))
        At0_AAt_inv = np.zeros((2, n_samples))
        set_At0_AAt_inv_from_k(AAt, k_i, At0_AAt_inv)
        b = np.zeros((2, n_samples))
        b[0] = (kpv_samples[:, 1] - x_hat[i, 0]) / DT - k_i * x_hat[i, 1]
        b[1] = -x_hat[i, 1]
        dv = (At0_AAt_inv * b).sum(axis=0)
        x_hat[i + 1, 0] = x_hat[i, 0] + DT * x_hat[i, 1]
        x_hat[i + 1, 1] = x_hat[i, 1] + dv + sd_dv * np.random.randn(n_samples)
    return x_hat[1:]


def multi_forward_kpv_samples(pkpv_samples, sd_dv_lat, k_str, n_obs, n_steps):
    """

    :param pkpv_samples: n_samples, 4 | [i] = [p_n, k, p_merge, v_n]
    :param sd_dv_lat:
    :param k_str:
    :param n_obs: n_samples,
    :param n_steps: n_samples,
    :return:
        x_hat: n_max, 2, n_samples | [i, :, j] = lat [p, v]
    """
    n_samples = pkpv_samples.shape[0]
    n_max = n_steps.max()
    start_inds = n_max - n_steps
    x_hat = np.zeros((1 + n_max, 2, n_samples))
    x_hat[start_inds, 0, np.arange(n_samples)] = pkpv_samples[:, 0]
    x_hat[start_inds, 1, np.arange(n_samples)] = pkpv_samples[:, -1]
    for i in range(n_max):
        mask = start_inds <= i
        n_mask = mask.sum()
        k_i = pkpv_samples[mask, 1] - (n_obs[mask] - 1) - i
        k_i[k_i < 3] = k_str
        AAt = np.zeros((4, n_mask))
        At0_AAt_inv = np.zeros((2, n_mask))
        set_At0_AAt_inv_from_k(AAt, k_i, At0_AAt_inv)
        b = np.zeros((2, n_mask))
        b[0] = (pkpv_samples[mask, 2] - x_hat[i, 0, mask]) / DT - k_i * x_hat[i, 1, mask]
        b[1] = -x_hat[i, 1, mask]
        dv = (At0_AAt_inv * b).sum(axis=0)
        x_hat[i + 1, 0, mask] = x_hat[i, 0, mask] + DT * x_hat[i, 1, mask]
        x_hat[i + 1, 1, mask] = x_hat[i, 1, mask] + dv + sd_dv_lat * np.random.randn(n_mask)
    return x_hat[1:]


def multi_forward_kpv_samples_v1(pkpv_samples, sd_dv_lat, k_str, n_obs, n_steps):
    """
    Calculate each A(k) variable once
    :param pkpv_samples: n_samples, 4 | [i] = [p_n, k, p_merge, v_n]
    :param sd_dv_lat:
    :param k_str:
    :param n_obs: n_samples,
    :param n_steps: n_samples,
    :return:
        x_hat: n_max, 2, n_samples | [i, :, j] = lat [p, v]
    """
    n_samples = pkpv_samples.shape[0]
    n_max = n_steps.max()
    start_inds = n_max - n_steps
    x_hat = np.zeros((1 + n_max, 2, n_samples))
    x_hat[start_inds, 0, np.arange(n_samples)] = pkpv_samples[:, 0]
    x_hat[start_inds, 1, np.arange(n_samples)] = pkpv_samples[:, -1]
    K_MAX = 120
    AAt = np.zeros((4, K_MAX))
    At0_AAt_inv = np.zeros((2, K_MAX))
    k_offset = 3
    k_save = np.arange(K_MAX) + k_offset
    set_At0_AAt_inv_from_k(AAt, k_save, At0_AAt_inv)
    for i in range(n_max):
        mask = start_inds <= i
        n_mask = mask.sum()
        k_i = pkpv_samples[mask, 1].astype(np.int) - (n_obs[mask] - 1) - i
        k_i[k_i < 3] = k_str
        b = np.zeros((2, n_mask))
        b[0] = (pkpv_samples[mask, 2] - x_hat[i, 0, mask]) / DT - k_i * x_hat[i, 1, mask]
        b[1] = -x_hat[i, 1, mask]
        dv = (At0_AAt_inv[:, k_i - k_offset] * b).sum(axis=0)
        x_hat[i + 1, 0, mask] = x_hat[i, 0, mask] + DT * x_hat[i, 1, mask]
        x_hat[i + 1, 1, mask] = x_hat[i, 1, mask] + dv + sd_dv_lat * np.random.randn(n_mask)
    return x_hat[1:]


def sample_merge_kpv_v0(p, is_obs, k_grid, sd_dv, mu_p, sd_p, bds_p, k_str, n_samples):
    """
    Solve with Kalman filtering

    - filtering -> nll for filtered values = unconstrained optima
    - set true f = according to filtered mu, var, but constrained
    - sample via proposal
    :param p: n, | lateral positions
    :param is_obs: n, | m true values for each of observed positions
    :param k_grid: n_k, |
    :param sd_dv:
    :param mu_p: mean of prior for p_merge
    :param sd_p: sd of prior for p_merge
    :param bds_p: 2, | [lb, ub] bounds for p_merge
    :param k_str:
    :param n_samples:
    :return:
        kpv_samples: n_samples, 3 | [i] = [k, p_merge, v_n]
        - sample (always unknown) v_n since p_n assumed observed
        nll: n_samples,
    """
    n = p.size
    x0 = np.array([0, 0, mu_p])
    P0 = np.diag((KF_LARGE_VAR, KF_LARGE_VAR, sd_p ** 2))
    A = make_kf_process_matrix(n, k_grid, k_str)  # n, n_k, 3, 3
    # - make A accounting for dv
    # - should be 3 x 3
    G = np.array([[0], [1.], [0]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = KF_EPS
    # (t, n_k, 3), (t, n_k, 3, 3)
    x_hat, P_hat, nll_f_opt = ka.batch_filter_tva_select_single(
        p, x0, P0, A, G, R, c_ind, Q)
    vp_opt = x_hat[-1, :, 1:]  # n_k, 2 = [v_n, p_merge]
    var_half = np.linalg.cholesky(P_hat[-1, :, 1:, 1:])  # n_k, 2, 2
    nlog_det_var = 2 * np.log(var_half[:, np.arange(2), np.arange(2)]).sum(axis=1)  # n_k,
    var_half_inv = smp.make_L_inv_2d(var_half)
    # true nll_f(sample) = posterior + marginal - prior
    # print(nll_f_opt)
    # print(vp_opt)

    # sampling:
    w = nll_f_opt - nll_f_opt.min()
    w = np.exp(-w)
    w /= w.sum()
    inds, vp_samples, nll_proposal = smp.sample_mixture_gaussians_2d(
        w, vp_opt, var_half, n_samples)

    kpv_samples = np.zeros((n_samples, 3))
    kpv_samples[:, 0] = k_grid[inds]
    kpv_samples[:, 1:] = vp_samples[:, [1, 0]]

    z = np.einsum('sij,sj->si', var_half_inv[inds], vp_samples - vp_opt[inds])
    nll_true = nll_f_opt[inds]\
        + 0.5 * ((z ** 2).sum(axis=1) + nlog_det_var[inds])
    nll = nll_true - nll_proposal
    return kpv_samples, nll


def sample_multi_merge_kpv_v0(p, k_grid, sd_dv, mu_p, sd_p, k_str, n_samples):
    """
    Solve with Kalman filtering

    - filtering -> nll for filtered values = unconstrained optima
    - set true f = according to filtered mu, var, but constrained
    - sample via proposal
    :param p: n, | lateral positions, nan if not observed
    :param k_grid: n_k, |
    :param sd_dv:
    :param mu_p: n_p, | mean of prior for p_merge
    :param sd_p: sd of prior for p_merge
    :param k_str:
    :param n_samples:
    :return:
        kpv_samples: n_samples, 3 | [i] = [k, p_merge, v_n]
        - sample (always unknown) v_n since p_n assumed observed
        nll: n_samples,
    """
    n = p.size
    n_p, n_k = mu_p.size, k_grid.size
    x0 = np.zeros((n_p * n_k, 3))
    x0[:, 2] = np.broadcast_to(mu_p[:, np.newaxis], (n_p, n_k)).reshape(-1)
    P0 = np.diag((KF_LARGE_VAR, KF_LARGE_VAR, sd_p ** 2))
    A = make_kf_process_matrix(n, k_grid, k_str)  # n, n_k, 3, 3
    A = np.broadcast_to(A[:, np.newaxis], (n, n_p, n_k, 3, 3)).reshape((n, n_p * n_k, 3, 3))
    G = np.array([[0], [1.], [0]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = KF_EPS
    # (t, n_pk, 3), (t, n_pk, 3, 3)
    x_hat, P_hat, nll_f_opt = ka.batch_filter_tva_select_single(
        p, x0, P0, A, G, R, c_ind, Q)
    vp_opt = x_hat[-1, :, 1:]  # n_k, 2 = [v_n, p_merge]
    var_half = np.linalg.cholesky(P_hat[-1, :, 1:, 1:])  # n_k, 2, 2
    nlog_det_var = 2 * np.log(var_half[:, np.arange(2), np.arange(2)]).sum(axis=1)  # n_k,
    var_half_inv = smp.make_L_inv_2d(var_half)
    # print(nll_f_opt)
    # print(vp_opt)

    # sampling:
    w = nll_f_opt - nll_f_opt.min()
    w = np.exp(-w)
    w /= w.sum()
    inds, vp_samples, nll_proposal = smp.sample_mixture_gaussians_2d(
        w, vp_opt, var_half, n_samples)

    k_grid_rep = np.broadcast_to(k_grid[np.newaxis, :], (n_p, n_k)).reshape(-1)
    kpv_samples = np.zeros((n_samples, 3))
    kpv_samples[:, 0] = k_grid_rep[inds]
    kpv_samples[:, 1:] = vp_samples[:, [1, 0]]

    z = np.einsum('sij,sj->si', var_half_inv[inds], vp_samples - vp_opt[inds])
    nll_true = nll_f_opt[inds]\
        + 0.5 * ((z ** 2).sum(axis=1) + nlog_det_var[inds])
    nll = nll_true - nll_proposal
    return kpv_samples, nll


def make_kf_process_matrix(n, k_grid, k_str):
    """
    x = (p; v; p_merge)
    x_{t+1} = A_t x_t describes the equations:

    p_merge_{t+1} = p_merge_t (does not change)
    p_{t+1} = p_t + t * v_t + dv part
    v_{t+1} = v_t + dv part

    dv part:

     ( 0    =  b_i(k) [ -1/t  -k  1/t   ( x_i
       dv )               0   -1   0  ]   p_merge )
    where b_i(k): 2, 2
                = (0 0; At0_AAt_inv)

    :param n:
    :param k_grid: n_k,
    :param k_str:
    :return:
        A: n, n_k, 3, 3 | process matrix for lane merging
            - at each time for each initial horizon
    """
    n_k = k_grid.size
    A = np.zeros((n, n_k, 3, 3))
    # [p v] part
    A[:, :, 0, 0] = 1.
    A[:, :, 0, 1] = DT
    A[:, :, 1, 1] = 1.
    # constant for p_merge
    A[:, :, 2, 2] = 1.
    # dv part - this A is abuse of notation
    At0_AAt_inv, k_i = make_At_AA_inv_vec(k_grid, k_str, n)
    A[:, :, 1, 0] += -At0_AAt_inv[0].T / DT
    A[:, :, 1, 1] += -k_i.T * At0_AAt_inv[0].T - At0_AAt_inv[1].T
    A[:, :, 1, 2] += At0_AAt_inv[0].T / DT
    return A


def make_At_AA_inv_vec(k, k_str, n):
    """
    Calculate vector that produces lateral control:

    At0_AAt_inv = A(k)^T[0](A(k)A(k)^T)^-1

    - which is a 2,1 vector given a k
    - like rate terms in pid
    Do this for each combination of k and step,
    where horizon k decreases each step, and when k = 1,
    switches to constant horizon k_str
    :param k: n_k,
    :param k_str:
    :param n:
    :return:
        At0_AAt_inv: 2, n_k, n |
        k_i: n_k, n | k used to compute each vector
    """
    n_k = k.size
    k_i = k[:, np.newaxis] - np.arange(n)  # n_k, n
    k_i[k_i < 3] = k_str
    AAt = np.zeros((4, n_k, n))
    At0_AAt_inv = np.zeros((2, n_k, n))
    set_At0_AAt_inv_from_k(AAt, k_i, At0_AAt_inv)
    return At0_AAt_inv, k_i


def set_At0_AAt_inv_from_k(AAt, k_i, At0_AAt_inv):
    AAt[0] = (k_i - 1) * k_i * (2 * k_i - 1) / 6
    AAt[1] = (k_i - 1) * k_i / 2
    AAt[2] = AAt[1]
    AAt[3] = k_i
    detAAt = AAt[0] * AAt[3] - AAt[1] ** 2
    AAt_inv = AAt[[3, 1, 2, 0]]
    AAt_inv[[1, 2]] *= -1
    AAt_inv /= detAAt
    At0_AAt_inv[0] = (k_i - 1) * AAt_inv[0] + AAt_inv[2]
    At0_AAt_inv[1] = (k_i - 1) * AAt_inv[1] + AAt_inv[3]


if __name__ == '__main__':
    main_sample_solutions()
