"""
Longitudinal solution for missing data case (observe only some p_i)
- drop constraint on v*
- sampling_v1: treat v*-v_n as standard prior - since v_n is linear function of x_{1|0}, f

"""
import numpy as np
import kinematic_model.mixture_sampling as smp
from kinematic_model import smoothing as smo
from kinematic_model import lanes as lan
import kinematic_model.kalman as ka

DT = 0.1
SMALL_V = 0.1  # [m/s] for sampling small velocities
LARGE_V = 1e3  # [m/s] for ~ inf in sampling truncated normals
LARGE_NLL = 1e8
MAX_Z_DIST = 4.
EPS_VAR = 1e-6  # small var to avoid degenerate distribution for p_n^i
MIN_LEAD_OBS = 10  # min #frames needed at 10Hz

KF_EPS = 1e-8
KF_LARGE_VAR = 1e8


def main_sample_solutions():
    from time import time

    seed = np.random.randint(1000)
    seed = 0
    print('seed: {}'.format(seed))
    np.random.seed(seed)

    n_steps = 100
    # v_i = 10. + 2 * np.sin(np.arange(n_steps)/10)
    # v_i = 10 + (.5 * np.random.randn(n_steps)).cumsum()
    v_i = 10 + (0. * np.random.randn(n_steps)).cumsum()
    g0 = 10.
    p0 = 0.
    v0 = 10.0
    k = 120
    sd_dv = 0.1
    vk = 12.
    gk = 8.

    n_samples = 100
    n_obs = 30

    sd_v_prior = 2.
    sd_gv_prior = np.array([10, 2.])
    k_hat = k + 0
    # is_obs_ego = (np.arange(n_obs) < 6) | (25 < np.arange(n_obs))
    # is_obs_ego = (np.arange(n_obs) < 10) | (20 < np.arange(n_obs))
    is_obs_ego = (np.arange(n_obs) < 15) | (17 < np.arange(n_obs))
    # is_obs = (np.arange(n_obs) < 16) | (25 < np.arange(n_obs))
    is_obs = (10 < np.arange(n_obs))
    is_obs = is_obs.reshape(-1, 1)

    x_true = np.zeros((1 + n_steps, 2))
    x_true[0] = [p0, v0]
    x_true[1:] = forward_gv_samples_given_vi(
        p0, v_i.reshape((-1, 1)), np.array([[v0, g0 + p0, v_i[0], gk, vk]]), sd_dv, k, n_steps, n_obs)[..., 0]
    x_surround = np.zeros((v_i.size, 1, 2))
    x_surround[:, 0, 1] = v_i
    x_surround[:, 0, 0] = p0 + g0
    x_surround[1:, 0, 0] += DT * v_i.cumsum()[:-1]
    p_obs_ego = x_true[:n_obs, 0].copy()
    p_obs_ego[~is_obs_ego] = np.nan  # bad value - 'not seen', but [n_obs-1] is
    x_surround_hat = smo.smooth_surround_LS_v0(x_surround[:n_obs, :, 0], is_obs, sd_dv)

    is_test_gv_prior = True
    time_0 = time()
    # - [g* v*] when leads exist
    if is_test_gv_prior:
        is_no_g_mask = ()
        v_x_gv_samples, nll = sample_lon_gv_v1(
            p_obs_ego, is_obs_ego, x_surround_hat, sd_gv_prior, sd_dv, k_hat, n_samples)
    else:
        # - [v*] only when no leads exist
        is_no_g_mask = np.ones(n_samples) == 1
        v_x_gv_samples, nll = sample_lon_v_v1(
            p_obs_ego, is_obs_ego, sd_v_prior, sd_dv, k_hat, n_samples)
    print('elapsed: {:.5f}s'.format(time() - time_0))

    x_hat = forward_gv_samples(p_obs_ego[-1], v_x_gv_samples, sd_dv, k, n_steps, is_no_g_mask=is_no_g_mask)

    np.set_printoptions(suppress=True)
    print(np.hstack((v_x_gv_samples, nll[:, np.newaxis])).round(2)[:20])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(np.arange(x_true.shape[0]), x_true[:, 0], alpha=0.5, color='black', marker='+', label='true')
    ax.plot(np.arange(n_obs)[is_obs_ego], x_true[:n_obs, 0][is_obs_ego],
            alpha=0.8, color='orange', marker='+', label='observed', ls='')
    ax.plot(n_obs + np.arange(x_hat.shape[0]), x_hat[:, 0], alpha=0.1, color='blue')
    plt.show()


def sample_lon_v_x_gv(p_ego, is_obs_ego, x_surround, is_obs,
                      lane_edges, lane_ind, sd_gv_prior, sd_dv, k, n_samples):
    """

    :param p_ego: n, | lon p
    :param is_obs_ego: n,
    :param x_surround: n, n_vic, 3 | estimated lon x_i = [p v] of possible the n_vic lead vehicles
        [..., 2] = lat p_i of lead vehicles, but this is not estimated; only is_obs values are valid
    :param is_obs: n, n_vic | True iff [i, j] of x_surround is observed
    :param lane_edges:
    :param lane_ind: lane of x_ego at t_0
    :param sd_gv_prior: 2, | prior for [g* v*]
    :param sd_dv:
    :param k
    :param n_samples:
    :return:
        v_x_gv_samples: 3, n_samples, 5 | [r] = samples of [v_n x_n^i g* v*] for maneuver r
        nll_samples: 3, n_samples
        is_no_g_mask: 3, n_samples
    """
    n, n_vic = x_surround.shape[:2]
    tau_merge_lon = -10  # [m]
    tau_max_lon = 50  # [m]
    r2lane_ind_add = np.array([0, 1, -1])  # offset for [str, +merge, -merge]
    r2min_lon_offset = np.array([0., tau_merge_lon, tau_merge_lon])
    v_x_gv_samples = np.zeros((3, n_samples, 5))
    nll_samples = np.zeros((3, n_samples))
    is_no_g_mask = np.ones((3, n_samples)) == 0

    # uniform prior over lanes -> can just sample n_samples from each
    lane_inds = -10 + np.zeros((n, n_vic))
    lane_inds[is_obs] = lan.lane_from_lat(x_surround[..., 2][is_obs], lane_edges)
    for i in range(3):
        # valid = vehicles with *any* *observed* values in lane
        # - this covers merges in/out of target lane
        # valid lon: vehicles with *any* valid values (based on observed p_ego)
        lane_mask = (lane_inds == lane_ind + r2lane_ind_add[i]).any(axis=0)
        lon_mask = (p_ego[is_obs_ego, np.newaxis] - r2min_lon_offset[i] <= x_surround[is_obs_ego, :, 0]).any(axis=0)
        mask = lane_mask & lon_mask & (x_surround[is_obs_ego, :, 0] <= p_ego[is_obs_ego, np.newaxis] + tau_max_lon).any(axis=0)
        if not mask.any():
            is_no_g_mask[i] = True
            v_x_gv_samples[i], nll_samples[i] = sample_lon_v_v1(
                p_ego, is_obs_ego, sd_gv_prior[1], sd_dv, k, n_samples)
        else:
            v_x_gv_samples[i], nll_samples[i] = sample_lon_gv_v1(
                p_ego, is_obs_ego, x_surround, sd_gv_prior, sd_dv, k, n_samples)
    return v_x_gv_samples.reshape(-1, 5), nll_samples.reshape(-1), is_no_g_mask.reshape(-1)


def forward_gv_samples_given_vi(p0, vi, v_x_gv_samples, sd_dv, k, n_steps, n_obs):
    """
    Forward samples for testing, where future trajectory of lead is known
    :param p0: last observed lon p
    :param vi: n_t, n_samples | observed future lead velocity, vi
    :param v_x_gv_samples: n_samples, 5 | [i] = [v0, p_i0, v_i0, g*, v*]
        - p_i0, v_i0 estimated for timestep [n_obs - 1]
    :param sd_dv:
    :param k: constant horizon
    :param n_steps:
    :return:
        x_hat: n_steps, 2, n_samples | [i, j] = lon [p, v]
    """
    n_t = vi.shape[0]
    n_samples = v_x_gv_samples.shape[0]
    x_hat = np.zeros((1 + n_steps, 2, n_samples))
    x_hat[0, 0] = p0
    x_hat[0, 1] = v_x_gv_samples[:, 0].copy()
    gv_i = v_x_gv_samples[:, 1:3].T.copy()  # 2, n_samples
    gv_i[0] -= p0
    At0_AAt_inv = make_A_vars(k)
    for i in range(n_steps):
        b = np.zeros((2, n_samples))
        b[0] = (v_x_gv_samples[:, 3] - gv_i[0]) / DT - k * (gv_i[1] - x_hat[i, 1])
        b[1] = v_x_gv_samples[:, 4] - x_hat[i, 1]
        # note (-) since change is based on (dv_i - dv) so ego part is negative
        dv = (At0_AAt_inv[:, np.newaxis] * -b).sum(axis=0)
        x_hat[i + 1, 0] = x_hat[i, 0] + DT * x_hat[i, 1]
        noise = sd_dv * np.random.randn(n_samples)
        # enforce v >= 0 by just take max(v, 0)
        v_next = x_hat[i, 1] + dv + noise
        v_next[v_next < 0] = 0
        x_hat[i + 1, 1] = v_next
        gv_i[0] += DT * (gv_i[1] - x_hat[i, 1])
        if i+1 + (n_obs - 2) < n_t:
            gv_i[1] = vi[i + 1]
        # gv_i[1] += 0
    return x_hat[1:]


def forward_gv_samples(p0, v_x_gv_samples, sd_dv, k, n_steps, is_no_g_mask=(), is_ignore=()):
    """
    :param p0: last observed lon p
    :param v_x_gv_samples: n_samples, 5 | [i] = [v0, p_i0, v_i0, g*, v*]
        - p_i0, v_i0 estimated for timestep [n_obs - 1]
    :param sd_dv:
    :param k: constant horizon
    :param n_steps:
    :param is_no_g_mask: n_samples, | if True, ignore g component of dv
        - for when sample corresponds to "no existing lead"
        - default = all False
    :param is_ignore:  n_samples, | default = all False
        - for ensuring v >= 0 for all samples, ignoring eg "large nll" samples
    :return:
        x_hat: n_steps, 2, n_samples | [i, j] = lon [p, v]
    """
    n_samples = v_x_gv_samples.shape[0]
    x_hat = np.zeros((1 + n_steps, 2, n_samples))
    x_hat[0, 0] = p0
    x_hat[0, 1] = v_x_gv_samples[:, 0].copy()
    gv_i = v_x_gv_samples[:, 1:3].T.copy()  # 2, n_samples
    gv_i[0] -= p0
    is_no_g_mask = is_no_g_mask if len(is_no_g_mask) > 0 else np.ones(n_samples) == 0
    is_ignore = is_ignore if len(is_ignore) > 0 else np.ones(n_samples) == 0
    At0_AAt_inv = make_A_vars(k)
    for i in range(n_steps):
        b = np.zeros((2, n_samples))
        b[0] = (v_x_gv_samples[:, 3] - gv_i[0]) / DT - k * (gv_i[1] - x_hat[i, 1])
        b[0, is_no_g_mask] = 0.
        b[1] = v_x_gv_samples[:, 4] - x_hat[i, 1]
        dv = (At0_AAt_inv[:, np.newaxis] * -b).sum(axis=0)
        dv[is_ignore] = 0.
        x_hat[i + 1, 0] = x_hat[i, 0] + DT * x_hat[i, 1]
        noise = sd_dv * np.random.randn(n_samples)
        # --
        # ab = np.zeros((n_samples, 2))
        # ab[:, 0] = -x_hat[i, 1]
        # ab[:, 1] = LARGE_V
        # noise = stats.generate_truncnorm_multi(ab, 0*ab[:, 0], sd_dv + 0*ab[:, 0])
        # --
        # enforce v >= 0 by just take max(v, 0)
        v_next = x_hat[i, 1] + dv + noise
        v_next[v_next < 0] = 0
        x_hat[i + 1, 1] = v_next
        gv_i[0] += DT * (gv_i[1] - x_hat[i, 1])
        gv_i[1] += sd_dv * np.random.randn(n_samples)
    return x_hat[1:]


def multi_forward_gv_samples(
        pv_x_gv_samples, sd_dv, k, n_steps, is_no_g_mask=(), is_ignore=()):
    """
    :param pv_x_gv_samples: n_samples, 6 | [i] = [p0, v0, p_i0, v_i0, g*, v*]
        - p_i0, v_i0 estimated for timestep [n_obs - 1]
    :param sd_dv:
    :param k: constant horizon
    :param n_steps: n_samples, | number of steps this sample needs to
        take to cover prediction horizon
    :param is_no_g_mask: n_samples, | if True, ignore g component of dv
        - for when sample corresponds to "no existing lead"
        - default = all False
    :param is_ignore:  n_samples, | default = all False
        - for ensuring v >= 0 for all samples, ignoring eg "large nll" samples
    :return:
        x_hat: n_steps, 2, n_samples | [i, j] = lon [p, v]
    """
    n_samples = pv_x_gv_samples.shape[0]
    n_max = n_steps.max()
    start_inds = n_max - n_steps
    x_hat = np.zeros((1 + n_max, 2, n_samples))
    x_hat[start_inds, 0, np.arange(n_samples)] = pv_x_gv_samples[:, 0].copy()
    x_hat[start_inds, 1, np.arange(n_samples)] = pv_x_gv_samples[:, 1].copy()
    gv_i = pv_x_gv_samples[:, 2:4].T.copy()  # 2, n_samples
    gv_i[0] -= pv_x_gv_samples[:, 0]
    is_no_g_mask = is_no_g_mask if len(is_no_g_mask) > 0 else np.ones(n_samples) == 0
    is_ignore = is_ignore if len(is_ignore) > 0 else np.ones(n_samples) == 0
    At0_AAt_inv = make_A_vars(k)
    for i in range(n_max):
        mask = start_inds <= i
        n_mask = mask.sum()
        b = np.zeros((2, n_mask))
        b[0] = (pv_x_gv_samples[mask, 4] - gv_i[0, mask]) / DT \
            - k * (gv_i[1, mask] - x_hat[i, 1, mask])
        b[0, is_no_g_mask[mask]] = 0.
        b[1] = pv_x_gv_samples[mask, 5] - x_hat[i, 1, mask]
        dv = (At0_AAt_inv[:, np.newaxis] * -b).sum(axis=0)
        dv[is_ignore[mask]] = 0.
        x_hat[i + 1, 0, mask] = x_hat[i, 0, mask] + DT * x_hat[i, 1, mask]
        noise = sd_dv * np.random.randn(n_mask)
        # enforce v >= 0 by just take max(v, 0)
        v_next = x_hat[i, 1, mask] + dv + noise
        v_next[v_next < 0] = 0
        x_hat[i + 1, 1, mask] = v_next
        gv_i[0, mask] += DT * (gv_i[1, mask] - x_hat[i, 1, mask])
        gv_i[1, mask] += sd_dv * np.random.randn(n_mask)
    return x_hat[1:]


def sample_lon_gv_v0(p_ego, is_obs_ego, x_surround, sd_gvprior, sd_dv, k, n_samples):
    """
    Solve with Kalman filtering:
    x = (p, v, g*, v*)' with g*, v* as unknown constants
    Handle prior on final state v_n after last step in forward pass.
    - since both p_n^i and p_n assumed to be observed, can apply this through
      usual prior as g*_prior = p_n^i - p_n
    :param p_ego: n,
    :param is_obs_ego: n,
    :param x_surround: n, n_mx, 2 | estimated lon x_i = [p v] of possible the n_mx lead vehicles
    :param sd_gvprior: 2,
    :param sd_dv:
    :param k:
    :param n_samples:
    :return:
        v_x_gv_samples: n_samples, 4
        nll: n_samples,
    """
    n, n_mx = x_surround.shape[:2]
    mu_g_prior = x_surround[-1, :, 0] - p_ego[-1]
    x0 = np.zeros((n_mx, 4))
    x0[:, 2] = mu_g_prior
    P0 = np.zeros((n_mx, 4, 4))
    P0[:] = np.eye(4) * KF_LARGE_VAR
    P0[:, 2, 2] = sd_gvprior[0] ** 2
    A, f = make_kf_gv_process_terms(x_surround, k)
    G = np.array([[0], [1.], [0], [0]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = KF_EPS
    prior_c = np.array([0, 1, 0, -1])
    prior_Q = sd_gvprior[1] ** 2
    prior_y = 0.
    x_hat, P_hat, nll_f_opt = ka.multi_batch_filter_prior_ti_select_single(
        np.broadcast_to(p_ego[:, np.newaxis], (n, n_mx)),
        x0, P0, A, G, R, c_ind, Q, prior_c, prior_Q, prior_y, f=f)
    v_gv_opt = x_hat[-1, :, 1:]  # n_k, 3 = [v_n, g*, v*]
    var_half = np.linalg.cholesky(P_hat[-1, :, 1:, 1:])  # n_k, 3, 3
    nlog_det_var = 2 * np.log(var_half[:, np.arange(3), np.arange(3)]).sum(axis=1)  # n_k,
    var_half_inv = smp.make_L_inv_nd(var_half)
    # print(nll_f_opt)
    # print(v_gv_opt)

    # sampling:
    w = nll_f_opt - nll_f_opt.min()
    w = np.exp(-w)
    w /= w.sum()
    inds, v_gv_samples, nll_proposal = smp.sample_mixture_gaussians_nd(
        w, v_gv_opt, var_half, n_samples, L_inv=var_half_inv)

    z = np.einsum('sij,sj->si', var_half_inv[inds], v_gv_samples - v_gv_opt[inds])
    nll_true = nll_f_opt[inds]\
        + 0.5 * ((z ** 2).sum(axis=1) + nlog_det_var[inds])\
        + 0.5 * ((v_gv_samples[:, 2] - v_gv_samples[:, 0]) / sd_gvprior[1]) ** 2 + np.log(sd_gvprior[1])
    nll = nll_true - nll_proposal
    v_x_gv_samples = np.zeros((n_samples, 5))
    v_x_gv_samples[:, [0, 3, 4]] = v_gv_samples
    v_x_gv_samples[:, 1] = x_surround[-1, inds, 0]
    v_x_gv_samples[:, 2] = x_surround[-1, inds, 1]
    return v_x_gv_samples, nll


def sample_lon_v_v0(p_ego, is_obs, sd_v_prior, sd_dv, k, n_samples):
    """
    Solve with Kalman filtering:
    x = (p, v, v*)' with v* as an unknown constant
    - No g* term as no lead vehicles are present
    Handle prior on final state v_n after last step in forward pass.
    :param p_ego: n, | lon over n timesteps
    :param is_obs: n, | whether lon [i] actually observed
    :param sd_v_prior:
    :param sd_dv:
    :param k:
    :param n_samples:
    :return:
        v_x_gv_samples: n_samples, 4 |
            - [1, 2, 3] part for x_n^i, g* is filled with dummy values
        nll: n_samples,
    """
    x0 = np.zeros(3)
    P0 = np.eye(3) * KF_LARGE_VAR
    A = make_kf_v_process_terms(k)
    G = np.array([[0], [1.], [0]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = KF_EPS
    prior_c = np.array([0, 1, -1])
    prior_Q = sd_v_prior ** 2
    prior_y = 0.
    x_hat, P_hat, nll_f_opt = ka.multi_batch_filter_prior_ti_select_single(
        p_ego.reshape(-1, 1),
        x0.reshape(1, 3),
        P0.reshape(1, 3, 3),
        A, G, R, c_ind, Q, prior_c, prior_Q, prior_y)
    x_hat = x_hat[:, 0]  # t, 2
    P_hat = P_hat[:, 0]  # t, 2, 2
    nll_f_opt = nll_f_opt[0]
    v_v_opt = x_hat[-1, 1:]  # 2, = [v_n, v*]
    # print(v_v_opt)
    var_half = np.linalg.cholesky(P_hat[-1, 1:, 1:])  # 2, 2
    # nlog_det_var = 2 * np.log(var_half[np.arange(2), np.arange(2)]).sum()
    # var_half_inv = smp.make_L_inv_2d(var_half[np.newaxis])[0]

    # sampling:
    v_v_samples, nll_proposal = smp.sample_gaussians_nd(v_v_opt, var_half, n_samples)

    nll_true = nll_f_opt\
        + 0.5 * ((v_v_samples[:, 1] - v_v_samples[:, 0]) / sd_v_prior) ** 2 + np.log(sd_v_prior)
    nll = nll_true - nll_proposal
    v_x_gv_samples = np.zeros((n_samples, 5))
    v_x_gv_samples[:, [0, 4]] = v_v_samples
    v_x_gv_samples[:, 1] = 1e5
    v_x_gv_samples[:, 2] = 10
    return v_x_gv_samples, nll


def sample_lon_gv_v1(p_ego, is_obs_ego, x_surround, sd_gvprior, sd_dv, k, n_samples):
    """
    Solve with Kalman filtering
    x = (p, v, g*, v*)' with g*, v* as unknown constants
    Handle prior on final state v_n after last step in forward pass.
    - since both p_n^i and p_n assumed to be observed, can apply this through
      usual prior as g*_prior = p_n^i - p_n
    - handle v* - v_n with v_n = linear function of x0, f
    :param p_ego: n,
    :param is_obs_ego: n,
    :param x_surround: n, n_mx, 2 | estimated lon x_i = [p v] of possible the n_mx lead vehicles
    :param sd_gvprior: 2,
    :param sd_dv:
    :param k:
    :param n_samples:
    :return:
        v_x_gv_samples: n_samples, 4
        nll: n_samples,
    """
    n, n_mx = x_surround.shape[:2]
    mu_g_prior = x_surround[-1, :, 0] - p_ego[-1]
    x0 = np.zeros((n_mx, 4))
    x0[:, 2] = mu_g_prior
    P0 = np.zeros((n_mx, 4, 4))
    P0[:] = np.eye(4) * KF_LARGE_VAR
    P0[:, 2, 2] = sd_gvprior[0] ** 2

    # v* - v_n prior
    A, f = make_kf_gv_process_terms(x_surround, k)
    c_prior, d_prior = make_kf_gv_vstar_prior_terms(A, f)
    x0, P0 = ka.step_update(x0, P0, c_prior, d_prior, sd_gvprior[1])

    G = np.array([[0], [1.], [0], [0]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = KF_EPS
    x_hat, P_hat, nll_f_opt = ka.multi_batch_filter_prior_ti_select_single(
        np.broadcast_to(p_ego[:, np.newaxis], (n, n_mx)),
        x0, P0, A, G, R, c_ind, Q, 0, 0, 0, f=f)
    v_gv_opt = x_hat[-1, :, 1:]  # n_k, 3 = [v_n, g*, v*]
    var_half = np.linalg.cholesky(P_hat[-1, :, 1:, 1:])  # n_k, 3, 3
    nlog_det_var = 2 * np.log(var_half[:, np.arange(3), np.arange(3)]).sum(axis=1)  # n_k,
    var_half_inv = smp.make_L_inv_nd(var_half)
    # print(nll_f_opt)
    # print(v_gv_opt)

    # sampling:
    w = nll_f_opt - nll_f_opt.min()
    w = np.exp(-w)
    w /= w.sum()
    inds, v_gv_samples, nll_proposal = smp.sample_mixture_gaussians_nd(
        w, v_gv_opt, var_half, n_samples, L_inv=var_half_inv)

    z = np.einsum('sij,sj->si', var_half_inv[inds], v_gv_samples - v_gv_opt[inds])
    nll_true = nll_f_opt[inds]\
        + 0.5 * ((z ** 2).sum(axis=1) + nlog_det_var[inds])
    nll = nll_true - nll_proposal
    v_x_gv_samples = np.zeros((n_samples, 5))
    v_x_gv_samples[:, [0, 3, 4]] = v_gv_samples
    v_x_gv_samples[:, 1] = x_surround[-1, inds, 0]
    v_x_gv_samples[:, 2] = x_surround[-1, inds, 1]
    return v_x_gv_samples, nll


def sample_lon_v_v1(p_ego, is_obs, sd_v_prior, sd_dv, k, n_samples):
    """
    Solve with Kalman filtering:
    x = (p, v, v*)' with v* as an unknown constant
    - No g* term as no lead vehicles are present
    Handle prior on final state v_n after last step in forward pass.
    :param p_ego: n, | lon over n timesteps
    :param is_obs: n, | whether lon [i] actually observed
    :param sd_v_prior:
    :param sd_dv:
    :param k:
    :param n_samples:
    :return:
        v_x_gv_samples: n_samples, 4 |
            - [1, 2, 3] part for x_n^i, g* is filled with dummy values
        nll: n_samples,
    """
    x0 = np.zeros(3)
    P0 = np.eye(3) * KF_LARGE_VAR
    # v* - v_n prior
    A = make_kf_v_process_terms(k)
    c_prior, d_prior = make_kf_v_vstar_prior_terms(A, p_ego.size)
    x0, P0 = ka.step_update(x0.reshape(1, 3), P0.reshape(1, 3, 3), c_prior, d_prior + np.zeros(1), sd_v_prior)

    G = np.array([[0], [1.], [0]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = KF_EPS
    prior_c = np.array([0, 1, -1])
    prior_Q = sd_v_prior ** 2
    prior_y = 0.
    x_hat, P_hat, nll_f_opt = ka.multi_batch_filter_prior_ti_select_single(
        p_ego.reshape(-1, 1),
        x0,  # .reshape(1, 3),
        P0,  # .reshape(1, 3, 3),
        A, G, R, c_ind, Q, prior_c, prior_Q, prior_y)
    x_hat = x_hat[:, 0]  # t, 2
    P_hat = P_hat[:, 0]  # t, 2, 2
    nll_f_opt = nll_f_opt[0]
    v_v_opt = x_hat[-1, 1:]  # 2, = [v_n, v*]
    # print(v_v_opt)
    var_half = np.linalg.cholesky(P_hat[-1, 1:, 1:])  # 2, 2

    # sampling:
    v_v_samples, nll_proposal = smp.sample_gaussians_nd(v_v_opt, var_half, n_samples)
    nll_true = nll_f_opt
    nll = nll_true - nll_proposal
    v_x_gv_samples = np.zeros((n_samples, 5))
    v_x_gv_samples[:, [0, 4]] = v_v_samples
    v_x_gv_samples[:, 1] = 1e5
    v_x_gv_samples[:, 2] = 10
    return v_x_gv_samples, nll


def make_kf_gv_process_terms(x_surround, k):
    """
    x = (p; v; g*, v*)
    x_{t+1} = A_t x_t + f_t describes the equations:
    0) the constants g* v* do not change
    1) double integrator dynamics
    p_{t+1} = p_t + t * v_t
    v_{t+1} = v_t + dv part
    2) dv part:
    (0   = b_i(k) [ 1/t    k  -1/t  -k  1/t  0   ( x_t
     dv)              0   -1    0    0   0   1 ]   x_t^i
                                                    g*
                                                    v*  )
    where b_i(k): 2, 2
                = (0 0; -At0_AAt_inv)
    - and x_t^i products are put into f_t

    :param x_surround: t, n_mx, 2 | estimated lon x_i = [p v] of possible the n_mx lead vehicles
    :param k: scalar | time horizon used to make longitudinal control
        - the A here is abuse of notation
    :return:
        A: 4, 4 | process matrix
        f: t, n_mx, 4 | constant offset due to known lead at each time
    """
    t, n_mx = x_surround.shape[:2]
    f = np.zeros((t, n_mx, 4))
    A = np.eye(4)
    A[0, 1] = DT
    At0_AAt_inv = make_A_vars(k)
    A[1, 0] += -At0_AAt_inv[0] / DT
    A[1, 1] += -At0_AAt_inv[0] * k + At0_AAt_inv[1]
    A[1, 2] += -At0_AAt_inv[0] / DT
    A[1, 3] += -At0_AAt_inv[1]
    f[..., 1] += x_surround[..., 0] * At0_AAt_inv[0] / DT
    f[..., 1] += x_surround[..., 1] * At0_AAt_inv[0] * k
    return A, f


def make_A_kv_gv(k):
    A = np.eye(4)
    A[0, 1] = DT
    At0_AAt_inv = make_A_vars(k)
    A[1, 0] += -At0_AAt_inv[0] / DT
    A[1, 1] += -At0_AAt_inv[0] * k + At0_AAt_inv[1]
    A[1, 2] += -At0_AAt_inv[0] / DT
    A[1, 3] += -At0_AAt_inv[1]
    return A


def make_Ap_kf_gv(A, t):
    w, v = np.linalg.eig(A)
    v_inv = np.linalg.inv(v)
    # t, 4, 4 : [0] = S^t-1, ..., [-2] = S, [-1] = I
    S = np.diag(w)[np.newaxis] ** np.arange(t-1, -1, -1)[:, np.newaxis, np.newaxis]
    S[-1] = np.eye(4)
    Ap = v @ S @ v_inv
    return Ap


def make_Ap_kf_v(A, t):
    w, v = np.linalg.eig(A)
    v_inv = np.linalg.inv(v)
    # t, 3, 3 : [0] = S^t-1, ..., [-2] = S, [-1] = I
    S = np.diag(w)[np.newaxis] ** np.arange(t-1, -1, -1)[:, np.newaxis, np.newaxis]
    S[-1] = np.eye(3)
    Ap = v @ S @ v_inv
    return Ap


def make_kf_v_process_terms(k):
    """
    x = (p; v; v*)
    x_{t+1} = A_t x_t + f_t describes the equations:
    0) the constant v* does not change
    1) double integrator dynamics
    p_{t+1} = p_t + t * v_t
    v_{t+1} = v_t + dv part
    2) dv part:
    (0   = b_i(k) [   0   -1   1 ] ( x_t      =  ( 0
     dv)                             v*   )        (v* - v_t)/k )

    where b_i(k): 2,
                = (0; 1/k)

    :param k: scalar | time horizon used to make longitudinal control
    :return:
        A: 3, 3 | process matrix
    """
    A = np.eye(3)
    A[0, 1] = DT
    A[1, 1] += -1/k
    A[1, 2] += 1/k
    return A


def make_kf_gv_vstar_prior_terms(A, f):
    """

    v* - v_n ~ N(0, sd^2)
        = (0 0 0 1)x_{1,0} - (0 1 0 0)[A^{t-1}x_{1,0} + A^{t-2}f[0] + ... + f[t-1]]
        = c_1 x_{1,0} - (c_2 x_{1,0} + d)
        = c x_{1,0} - d

    So prior => (c_1 - c_2) x_{1,0} = d + sd * N(0,1)
    :param A: 4, 4
    :param f: t, n_mx, 4
    :return:
        c: 4,
        d: n_mx,
    """
    c1 = np.array([0, 0, 0, 1.])

    # --
    # w, v = np.linalg.eig(A)
    # v_inv = np.linalg.inv(v)
    # t = f.shape[0]
    # # t, 4, 4 : [0] = S^t-1, ..., [-2] = S, [-1] = I
    # S = np.diag(w)[np.newaxis] ** np.arange(t-1, -1, -1)[:, np.newaxis, np.newaxis]
    # S[-1] = np.eye(4)
    # Ap = v @ S @ v_inv
    # --
    t = f.shape[0]
    Ap = AP_KF_GV[T_MAX - t:]
    # --

    Af_sum = np.einsum('tij,tkj->tki', Ap[1:], f[:-1])
    d = Af_sum[:, :, 1].sum(axis=0)  # n_k
    c2 = Ap[0, 1]
    c = c1 - c2
    # w, v = np.linalg.eig(A)
    # (v @ np.diag(w) @ np.linalg.inv(v)).round(2) = A
    return c, d


def make_kf_v_vstar_prior_terms(A, t):
    c1 = np.array([0, 0, 1.])
    # --
    # w, v = np.linalg.eig(A)
    # v_inv = np.linalg.inv(v)
    # S = np.diag(w) ** (t-1)
    # Ap = v @ S @ v_inv
    # --
    Ap = AP_KF_V[T_MAX - t]
    # --
    c2 = Ap[1]
    c = c1 - c2
    d = 0
    return c, d


def make_A_vars(k):
    # At0_AAt_inv: 2,
    AAt = np.zeros(4)
    AAt[0] = (k - 1) * k * (2 * k - 1) / 6
    AAt[1] = (k - 1) * k / 2
    AAt[2] = AAt[1]
    AAt[3] = k
    detAAt = AAt[0] * AAt[3] - AAt[1] ** 2
    AAt_inv = AAt[[3, 1, 2, 0]]
    AAt_inv[[1, 2]] *= -1
    AAt_inv /= detAAt
    At0_AAt_inv = np.zeros(2)
    At0_AAt_inv[0] = (k - 1) * AAt_inv[0] + AAt_inv[2]
    At0_AAt_inv[1] = (k - 1) * AAt_inv[1] + AAt_inv[3]
    return At0_AAt_inv


T_MAX = 30
K_LON = 100
A_KF_GV = make_A_kv_gv(K_LON)
AP_KF_GV = make_Ap_kf_gv(A_KF_GV, T_MAX)

A_KF_V = make_kf_v_process_terms(K_LON)
AP_KF_V = make_Ap_kf_v(A_KF_V, T_MAX)


if __name__ == '__main__':
    main_sample_solutions()
