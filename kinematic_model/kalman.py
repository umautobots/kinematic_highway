import numpy as np


EPS = 1e-8


def batch_filter_ti(y, x0, P0, A, G, R, C, Q):
    """
    Apply Kalman filtering to time-invariant system given by

    x_{t+1} = Ax_t + Ge_t
    y_t = Cx_t + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, m | observations over 1,...,t
    :param x0: n, | mean of prior on x0
    :param P0: n, n  | variance of prior on x0
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param C: m, n | observation matrix
    :param Q: m, m | observation noise
    :return:
        x: t, n | filtered estimates of x (x_i | y_1:i)
        P: t, n, n | variance of filtered estimates
    """
    t, m = y.shape
    n = x0.size
    x = np.zeros((t, n))
    P = np.zeros((t, n, n))
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        PCt = P_m1.dot(C.T)
        CPC_Q = C.dot(PCt) + Q  # assert CPC_Q pos def
        K = np.linalg.solve(CPC_Q, PCt.T).T
        x[i] = x_m1 + K @ (y[i] - C @ x_m1)
        P[i] = P_m1 - K @ C @ P_m1
        x_m1 = A @ x[i]
        P_m1 = A @ P[i] @ A.T + GRG
    return x, P


def multi_batch_filter_ti(y, x0, P0, A, G, R, C, Q):
    """
    Apply Kalman filtering to time-invariant system given by

    x_{t+1} = Ax_t + Ge_t
    y_t = Cx_t + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, n_k, m | observations over 1,...,t for n_k systems
    :param x0: n_k, n, | mean of prior on x0 (or can broadcast to given shape)
    :param P0: n_k, n, n  | variance of prior on x0
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param C: m, n | observation matrix
    :param Q: m, m | observation noise
    :return:
        x: t, n_k, n | filtered estimates of x (x_i | y_1:i)
        P: t, n_k, n, n | variance of filtered estimates
    """
    is_obs = ~np.isnan(y).any(axis=-1)
    t, n_k = y.shape[:2]
    n = A.shape[0]
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    x_m1 = np.broadcast_to(x0, (n_k, n))
    P_m1 = np.broadcast_to(P0, (n_k, n, n))
    GRG = G @ R @ G.T
    for i in range(t):
        x[i] = x_m1
        P[i] = P_m1
        # is_obs[i]: ki are True
        PCt = P_m1[is_obs[i]] @ C.T  # ki, n, m
        CPC_Q = C @ PCt + Q  # ki, m, m
        K = np.linalg.solve(CPC_Q, PCt.transpose((0, 2, 1))) \
            .transpose((0, 2, 1))
        y_dif = y[i, is_obs[i]] - np.einsum('ij,kj->ki', C, x_m1[is_obs[i]])
        x[i, is_obs[i]] += np.einsum('knm,km->kn', K, y_dif)
        P[i, is_obs[i]] -= K @ C @ P_m1[is_obs[i]]
        x_m1 = x[i] @ A.T
        P_m1 = A @ P[i] @ A.T + GRG
    return x, P


def batch_smooth_ti(y, x0, P0, A, G, R, C, Q):
    """
    Apply Kalman smoothing to time-invariant system given by

    x_{t+1} = Ax_t + Ge_t
    y_t = Cx_t + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, m | observations over 1,...,t | may be nan for unobserved
    :param x0: n, | mean of prior on x0
    :param P0: n, n  | variance of prior on x0
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param C: m, n | observation matrix
    :param Q: m, m | observation noise
    :return:
        x: t, n | smoothed estimates of x (x_i | y_1:t)
        P: t, n, n | variance of smoothed estimates
    """
    is_obs = ~np.isnan(y).any(axis=1)
    t, m = y.shape
    n = x0.size
    x = np.zeros((t, n))
    P = np.zeros((t, n, n))
    P_m1_arr = np.zeros((t, n, n))
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        if is_obs[i]:
            PCt = P_m1.dot(C.T)
            CPC_Q = C.dot(PCt) + Q  # assert CPC_Q pos def
            K = np.linalg.solve(CPC_Q, PCt.T).T
            x[i] = x_m1 + K @ (y[i] - C @ x_m1)
            P[i] = P_m1 - K @ C @ P_m1  # 'V_i'
        else:
            x[i] = x_m1
            P[i] = P_m1
        x_m1 = A @ x[i]
        P_m1 = A @ P[i] @ A.T + GRG  # 'P_i'
        P_m1_arr[i] = P_m1

    eye_n = np.eye(n)
    for i in range(t-2, -1, -1):
        # C = P[i] @ A.T @ np.linalg.inv(P_m1_arr[i] + EPS * eye_n)
        C = np.linalg.solve(P_m1_arr[i] + EPS * eye_n, A @ P[i]).T
        x[i] = x[i] + C @ (x[i + 1] - A @ x[i])
        P[i] = P[i] + C @ (P[i + 1] - P[i]) @ C.T
    return x, P


def batch_smooth_ti_select_single(y, x0, P0, A, G, R, c_ind, Q):
    """
    Apply Kalman smoothing to time-invariant system given by

    x_{t+1} = Ax_t + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, | observations over 1,...,t | may be nan for unobserved
    :param x0: n, | mean of prior on x0
    :param P0: n, n  | variance of prior on x0
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :return:
        x: t, n | smoothed estimates of x (x_i | y_1:t)
        P: t, n, n | variance of smoothed estimates
    """
    is_obs = ~np.isnan(y)
    t = y.size
    n = x0.size
    x = np.zeros((t, n))
    P = np.zeros((t, n, n))
    P_m1_arr = np.zeros((t, n, n))
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        if is_obs[i]:
            PCt = P_m1[:, c_ind]
            CPC_Q = P_m1[c_ind, c_ind] + Q
            K = PCt / CPC_Q  # n,
            x[i] = x_m1 + K * (y[i] - x_m1[c_ind])
            KCP_m1 = (P_m1[:, [c_ind]] * K[np.newaxis]).T
            P[i] = P_m1 - KCP_m1  # 'V_i'
        else:
            x[i] = x_m1
            P[i] = P_m1
        x_m1 = A @ x[i]
        P_m1 = A @ P[i] @ A.T + GRG  # 'P_i'
        P_m1_arr[i] = P_m1
    # eps in case initial indices in P_m1_arr happen to be improper prior
    C = np.linalg.solve(
        P_m1_arr + EPS * np.eye(n)[np.newaxis],
        A @ P
    ).transpose((0, 2, 1))  # t-2, n, n
    for i in range(t-2, -1, -1):
        # C = P[i] @ A.T @ np.linalg.inv(P_m1_arr[i] + EPS * eye_n)
        # C = np.linalg.solve(P_m1_arr[i] + EPS * eye_n, A @ P[i]).T
        x[i] = x[i] + C[i] @ (x[i + 1] - A @ x[i])
        P[i] = P[i] + C[i] @ (P[i + 1] - P[i]) @ C[i].T
    return x, P


def multi_batch_smooth_ti_select_single(y, x0, P0, A, G, R, c_ind, Q, f=None, is_P=False):
    """
    Smooth n_k separate observation sequences (that may be unobserved in
    different places) at once for the same time-invariant system.
    Apply Kalman smoothing to time-invariant system given by

    x_{t+1} = Ax_t + f + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, n_k | observations over 1,...,t; may be nan for unobserved
    :param x0: n_k, n | mean of prior on x0 for each of n_k series
    :param P0: n_k, n, n  | variance of prior on x0 for each of n_k series
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :param f: t, n_k, n | known constants
    :return:
        x: t, n_k, n | smoothed estimates of x (x_i | y_1:t)
        P: t, n_k, n, n | variance of smoothed estimates
    """
    is_obs = ~np.isnan(y)
    t, n_k = y.shape
    if f is None:
        f = np.zeros(t)
    n = x0.shape[1]
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    P_m1_arr = np.zeros((t, n_k, n, n))
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        x[i] = x_m1
        P[i] = P_m1
        # is_obs[i]: ki are True
        PCt = P_m1[is_obs[i], :, c_ind]  # ki, n
        CPC_Q = P_m1[is_obs[i], c_ind, c_ind] + Q  # ki,
        K = PCt / CPC_Q[:, np.newaxis]  # ki, n
        x[i, is_obs[i]] += K * (y[i, is_obs[i]] - x_m1[is_obs[i], c_ind])[:, np.newaxis]
        # (ki, n, 1), (ki, 1, n) -> (ki, n, n), then ^T
        KCP_m1 = (P_m1[is_obs[i], :, c_ind][..., np.newaxis]
                  * K[:, np.newaxis]).transpose((0, 2, 1))
        P[i, is_obs[i]] += -KCP_m1
        x_m1 = x[i] @ A.T + f[i]
        P_m1 = A[np.newaxis] @ P[i] @ A.T[np.newaxis] + GRG  # 'P_i'
        P_m1_arr[i] = P_m1
    # eps in case initial indices in P_m1_arr happen to be improper P_m1[is_obs[i], 0, 0]prior
    C = np.linalg.solve(
        P_m1_arr + EPS * np.eye(n)[np.newaxis, np.newaxis],
        A[np.newaxis, np.newaxis] @ P
    ).transpose((0, 1, 3, 2))  # t, n_k, n, n
    for i in range(t-2, -1, -1):
        x[i] += np.einsum('kij,kj->ki', C[i], x[i + 1] - x[i] @ A.T - f[i])
    if is_P:
        for i in range(t - 2, -1, -1):
            P[i] = P[i] + C[i] @ (P[i + 1] - P[i]) @ C[i].transpose((0, 2, 1))
    return x, P


def batch_smooth_tva_select_single(y, x0, P0, A, G, R, c_ind, Q):
    """
    Apply Kalman smoothing to n_k separate time-varying systems given by

    x_{t+1} = A_t x_t + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q
    and A_t is known in advance.
    :param y: t, | observations over 1,...,t | may be nan for unobserved
    :param x0: n, | mean of prior on x0
    :param P0: n, n  | variance of prior on x0
    :param A: t, n_k, n, n | process transition matrix for each (time, system)
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :return:
        x: t, n_k, n | smoothed estimates of x (x_i | y_1:t)
        P: t, n_k, n, n | variance of smoothed estimates
    """
    is_obs = ~np.isnan(y)
    t, n_k, n = A.shape[:3]
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    P_m1_arr = np.zeros((t, n_k, n, n))
    x_m1 = np.broadcast_to(x0, (n_k, n))
    P_m1 = np.broadcast_to(P0, (n_k, n, n))
    GRG = G @ R @ G.T
    for i in range(t):
        x[i] = x_m1
        P[i] = P_m1
        if is_obs[i]:
            PCt = P_m1[:, :, c_ind]  # ki, n
            CPC_Q = P_m1[:, c_ind, c_ind] + Q  # ki,
            K = PCt / CPC_Q[:, np.newaxis]  # ki, n
            x[i] += K * (y[i] - x_m1[:, c_ind])[:, np.newaxis]
            # (ki, n, 1), (ki, 1, n) -> (ki, n, n), then ^T
            KCP_m1 = (P_m1[:, :, c_ind][..., np.newaxis]
                      * K[:, np.newaxis]).transpose((0, 2, 1))
            P[i] += -KCP_m1
        x_m1 = np.einsum('kij,kj->ki', A[i], x[i])
        P_m1 = A[i] @ P[i] @ A[i].transpose((0, 2, 1)) + GRG
        P_m1_arr[i] = P_m1
    # eps in case initial indices in P_m1_arr happen to be improper prior
    C = np.linalg.solve(
        P_m1_arr + EPS * np.eye(n)[np.newaxis, np.newaxis],
        A @ P
    ).transpose((0, 1, 3, 2))  # t, n_k, n, n
    Ax = np.einsum('tkij,tkj->tki', A, x)  # t, n_k, n
    for i in range(t-2, -1, -1):
        x[i] += np.einsum('kij,kj->ki', C[i], x[i + 1] - Ax[i])
        P[i] = P[i] + C[i] @ (P[i + 1] - P[i]) @ C[i].transpose((0, 2, 1))
    return x, P


def batch_filter_tva_select_single(y, x0, P0, A, G, R, c_ind, Q):
    """
    Apply Kalman filtering to n_k separate time-varying systems given by

    x_{t+1} = A_t x_t + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q
    and A_t is known in advance.
    :param y: t, | observations over 1,...,t | may be nan for unobserved
    :param x0: n, | mean of prior on x0
    :param P0: n, n  | variance of prior on x0
    :param A: t, n_k, n, n | process transition matrix for each (time, system)
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :return:
        x: t, n_k, n | filtered estimates of x (x_i | y_1:t)
        P: t, n_k, n, n | variance of filtered estimates
        nll: n_k, | -log likelihood of marginal probability p(y_{observed})
    """
    is_obs = ~np.isnan(y)
    t, n_k, n = A.shape[:3]
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    nll = np.zeros(n_k)
    x_m1 = np.broadcast_to(x0, (n_k, n))
    P_m1 = np.broadcast_to(P0, (n_k, n, n))
    GRG = G @ R @ G.T
    for i in range(t):
        x[i] = x_m1
        P[i] = P_m1
        if is_obs[i]:
            # calculate nll
            var_r = P_m1[:, c_ind, c_ind] + Q
            r = y[i] - x_m1[:, c_ind]
            nll += .5 * ((r ** 2) / var_r + np.log(var_r))

            # update
            PCt = P_m1[:, :, c_ind]  # ki, n
            CPC_Q = P_m1[:, c_ind, c_ind] + Q  # ki,
            K = PCt / CPC_Q[:, np.newaxis]  # ki, n
            x[i] += K * (y[i] - x_m1[:, c_ind])[:, np.newaxis]
            # (ki, n, 1), (ki, 1, n) -> (ki, n, n), then ^T
            KCP_m1 = (P_m1[:, :, c_ind][..., np.newaxis]
                      * K[:, np.newaxis]).transpose((0, 2, 1))
            P[i] += -KCP_m1

        x_m1 = np.einsum('kij,kj->ki', A[i], x[i])
        P_m1 = A[i] @ P[i] @ A[i].transpose((0, 2, 1)) + GRG
    return x, P, nll


def multi_batch_smooth_prior_ti_select_single(
        y, x0, P0, A, G, R, c_ind, Q, prior_c, prior_Q, prior_y, f=None, is_nll=False):
    """
    Smooth n_k separate observation sequences (that may be unobserved in
    different places) at once for the same time-invariant system.
    Apply Kalman smoothing to time-invariant system given by

    x_{t+1} = Ax_t + f_t + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q
    with prior on last state specified by
    prior_y = prior_c @ x_T + eps_t
    var(eps_t) = prior_Q

    :param y: t, n_k | observations over 1,...,t; may be nan for unobserved
    :param x0: n_k, n | mean of prior on x0 for each of n_k series
    :param P0: n_k, n, n  | variance of prior on x0 for each of n_k series
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :param prior_c: n, | row vector observation matrix for prior
    :param prior_Q: scalar | prior's var
    :param prior_y: scalar | prior's mean
    :param f: t, n_k, n | known constants
    :return:
        x: t, n_k, n | smoothed estimates of x (x_i | y_1:t)
        P: t, n_k, n, n | variance of smoothed estimates
    """
    is_obs = ~np.isnan(y)
    t, n_k = y.shape
    if f is None:
        f = np.zeros(t)
    n = x0.shape[1]
    nll = np.zeros(n_k)
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    P_m1_arr = np.zeros((t, n_k, n, n))
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        x[i] = x_m1
        P[i] = P_m1
        # calculate nll
        var_r = P_m1[is_obs[i], c_ind, c_ind] + Q
        r = y[i, is_obs[i]] - x_m1[is_obs[i], c_ind]
        nll[is_obs[i]] += .5 * ((r ** 2) / var_r + np.log(var_r))
        # is_obs[i]: ki are True
        PCt = P_m1[is_obs[i], :, c_ind]  # ki, n
        CPC_Q = P_m1[is_obs[i], c_ind, c_ind] + Q  # ki,
        K = PCt / CPC_Q[:, np.newaxis]  # ki, n
        x[i, is_obs[i]] += K * (y[i, is_obs[i]] - x_m1[is_obs[i], c_ind])[:, np.newaxis]
        # (ki, n, 1), (ki, 1, n) -> (ki, n, n), then ^T
        KCP_m1 = (P_m1[is_obs[i], :, c_ind][..., np.newaxis]
                  * K[:, np.newaxis]).transpose((0, 2, 1))
        P[i, is_obs[i]] += -KCP_m1
        x_m1 = x[i] @ A.T + f[i]
        P_m1 = A[np.newaxis] @ P[i] @ A.T[np.newaxis] + GRG  # 'P_i'
        P_m1_arr[i] = P_m1
    # prior on final value step
    PCt = P[-1] @ prior_c  # n_k, n
    CPC_Q = PCt @ prior_c + prior_Q  # n_k,
    K = PCt / CPC_Q[:, np.newaxis]
    x[-1] += K * (prior_y - x[-1] @ prior_c)[:, np.newaxis]
    KCP_m1 = (PCt[..., np.newaxis]
              * K[:, np.newaxis]).transpose((0, 2, 1))
    P[-1] += -KCP_m1
    # eps in case initial indices in P_m1_arr happen to be improper P_m1[is_obs[i], 0, 0]prior
    C = np.linalg.solve(
        P_m1_arr + EPS * np.eye(n)[np.newaxis, np.newaxis],
        A[np.newaxis, np.newaxis] @ P
    ).transpose((0, 1, 3, 2))  # t, n_k, n, n
    for i in range(t-2, -1, -1):
        x[i] += np.einsum('kij,kj->ki', C[i], x[i + 1] - x[i] @ A.T - f[i])
        P[i] = P[i] + C[i] @ (P[i + 1] - P[i]) @ C[i].transpose((0, 2, 1))
    if is_nll:
        return x, P, nll
    return x, P


def multi_batch_smooth_nll_ti_select_single(
        y, x0, P0, A, G, R, c_ind, Q, f=None):
    """
    Smooth n_k separate observation sequences (that may be unobserved in
    different places) at once for the same time-invariant system.
    Apply Kalman smoothing to time-invariant system given by

    x_{t+1} = Ax_t + f_t + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, n_k | observations over 1,...,t; may be nan for unobserved
    :param x0: n_k, n | mean of prior on x0 for each of n_k series
    :param P0: n_k, n, n  | variance of prior on x0 for each of n_k series
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :param f: t, n_k, n | known constants
    :return:
        x: t, n_k, n | smoothed estimates of x (x_i | y_1:t)
        P: t, n_k, n, n | variance of smoothed estimates
    """
    is_obs = ~np.isnan(y)
    t, n_k = y.shape
    if f is None:
        f = np.zeros(t)
    n = x0.shape[1]
    nll = np.zeros(n_k)
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    P_m1_arr = np.zeros((t, n_k, n, n))
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        # P_m1_arr[i] = P_m1
        x[i] = x_m1
        P[i] = P_m1
        # calculate nll
        var_r = P_m1[is_obs[i], c_ind, c_ind] + Q
        r = y[i, is_obs[i]] - x_m1[is_obs[i], c_ind]
        nll[is_obs[i]] += .5 * ((r ** 2) / var_r + np.log(var_r))
        # is_obs[i]: ki are True
        PCt = P_m1[is_obs[i], :, c_ind]  # ki, n
        CPC_Q = P_m1[is_obs[i], c_ind, c_ind] + Q  # ki,
        K = PCt / CPC_Q[:, np.newaxis]  # ki, n
        x[i, is_obs[i]] += K * (y[i, is_obs[i]] - x_m1[is_obs[i], c_ind])[:, np.newaxis]
        # (ki, n, 1), (ki, 1, n) -> (ki, n, n), then ^T
        KCP_m1 = (P_m1[is_obs[i], :, c_ind][..., np.newaxis]
                  * K[:, np.newaxis]).transpose((0, 2, 1))
        P[i, is_obs[i]] += -KCP_m1
        x_m1 = x[i] @ A.T + f[i]
        P_m1 = A[np.newaxis] @ P[i] @ A.T[np.newaxis] + GRG  # 'P_i'
        P_m1_arr[i] = P_m1
    # eps in case initial indices in P_m1_arr happen to be improper P_m1[is_obs[i], 0, 0]prior
    C = np.linalg.solve(
        P_m1_arr + EPS * np.eye(n)[np.newaxis, np.newaxis],
        A[np.newaxis, np.newaxis] @ P
    ).transpose((0, 1, 3, 2))  # t, n_k, n, n
    for i in range(t-2, -1, -1):
        x[i] += np.einsum('kij,kj->ki', C[i], x[i + 1] - x[i] @ A.T - f[i])
        P[i] = P[i] + C[i] @ (P[i + 1] - P[i]) @ C[i].transpose((0, 2, 1))
    return x, P, nll


def multi_batch_filter_prior_ti_select_single(
        y, x0, P0, A, G, R, c_ind, Q, prior_c, prior_Q, prior_y, f=None):
    """
    Filter n_k separate observation sequences (that may be unobserved in
    different places) at once for the same time-invariant system.
    Apply Kalman filtering to time-invariant system given by

    x_{t+1} = Ax_t + f_t + Ge_t
    y_t = x_t[c_ind] + w_t

    where {e_t}, {w_t} are uncorrelated zero-mean
    white Gaussian noise processes with
    var(e_t) = R
    var(w_t) = Q

    :param y: t, n_k | observations over 1,...,t; may be nan for unobserved
    :param x0: n_k, n | mean of prior on x0 for each of n_k series
    :param P0: n_k, n, n  | variance of prior on x0 for each of n_k series
    :param A: n, n | process transition matrix
    :param G: n, p | matrix for applying process noise
    :param R: p, p | process noise variance
    :param c_ind: index | ind for observation matrix that selects single element from x
    :param Q: scalar | observation noise's variance
    :param prior_c: n, | row vector observation matrix for prior
    :param prior_Q: scalar | prior's var
    :param prior_y: scalar | prior's mean
    :param f: t, n_k, n | known constants
    :return:
        x: t, n_k, n | filtered estimates of x (x_i | y_1:t)
        P: t, n_k, n, n | variance of filtered estimates
        nll: n_k, | -log likelihood of marginal probability p(y_{observed})
    """
    is_obs = ~np.isnan(y)
    t, n_k = y.shape
    if f is None:
        f = np.zeros(t)
    n = x0.shape[1]
    x = np.zeros((t, n_k, n))
    P = np.zeros((t, n_k, n, n))
    nll = np.zeros(n_k)
    x_m1 = x0
    P_m1 = P0
    GRG = G @ R @ G.T
    for i in range(t):
        x[i] = x_m1
        P[i] = P_m1

        # calculate nll
        var_r = P_m1[is_obs[i], c_ind, c_ind] + Q
        r = y[i, is_obs[i]] - x_m1[is_obs[i], c_ind]
        nll[is_obs[i]] += .5 * ((r ** 2) / var_r + np.log(var_r))

        # is_obs[i]: ki are True
        PCt = P_m1[is_obs[i], :, c_ind]  # ki, n
        CPC_Q = P_m1[is_obs[i], c_ind, c_ind] + Q  # ki,
        K = PCt / CPC_Q[:, np.newaxis]  # ki, n
        x[i, is_obs[i]] += K * (y[i, is_obs[i]] - x_m1[is_obs[i], c_ind])[:, np.newaxis]
        # (ki, n, 1), (ki, 1, n) -> (ki, n, n), then ^T
        KCP_m1 = (P_m1[is_obs[i], :, c_ind][..., np.newaxis]
                  * K[:, np.newaxis]).transpose((0, 2, 1))
        P[i, is_obs[i]] += -KCP_m1
        x_m1 = x[i] @ A.T + f[i]
        P_m1 = A[np.newaxis] @ P[i] @ A.T[np.newaxis] + GRG  # 'P_i'
    return x, P, nll


def batch_nll_tva_select_single(x_hat, z_hat, A, noise_inds, sd_x, mu_z, sd_z):
    """
    Calculate negative log likelihood of estimated x and z
    - variables are partitioned into x (time varying) and z (constant)
    :param x_hat: t, n_k, n | estimates of x_{1:t}
    :param z_hat: n_k, n_z | estimates of the n_z constant variables
    :param A: t, n_k, n, n+n_z | process matrix by (time, system)
        - mapping (free variables x_t + constants z) -> x_{t+1}
    :param noise_inds: n_i, | select subset of (x_{t+1} - A_t x_t) indices to evaluate
        - for when some components of x have zero noise, to avoid evaluating those
    :param sd_x: n_i, | standard dev for evaluating noisy components of x
    :param mu_z: n_z, | prior mean on z
    :param sd_z: n_z, | prior sd on z
    :return:
        nll: n_k,
    """
    n = x_hat.shape[2]
    Ax = np.einsum('tkij,tkj->tki', A[:-1, :, :, :n], x_hat[:-1])  # t-1, n_k, n
    Ax += np.einsum('tkij,kj->tki', A[:-1, :, :, n:], z_hat)
    resid = x_hat[1:] - Ax
    nll = 0.5 * ((resid[..., noise_inds] / sd_x) ** 2).sum(axis=(0, 2))  # n_k,
    resid_z = (z_hat - mu_z) / sd_z
    nll_z = 0.5 * (resid_z ** 2).sum(axis=1)
    return nll + nll_z


def batch_nll_ti_select_single(x_hat, z_hat, A, noise_inds, sd_x, mu_z, sd_z, f=None):
    """
    Calculate negative log likelihood of estimated x and z
    - variables are partitioned into x (time varying) and z (constant)
    :param x_hat: t, n_k, n | estimates of x_{1:t}
    :param z_hat: n_k, n_z | estimates of the n_z constant variables
    :param A: n, n+n_z | process matrix
        - mapping (free variables x_t + constants z) -> x_{t+1}
    :param noise_inds: n_i, | select subset of (x_{t+1} - A_t x_t) indices to evaluate
        - for when some components of x have zero noise, to avoid evaluating those
    :param sd_x: n_i, | standard dev for evaluating noisy components of x
    :param mu_z: n_k, n_z | prior mean on z
    :param sd_z: n_z, | prior sd on z
    :param f: t, n_k, n+n_z | known forcing terms
    :return:
        nll: n_k,
    """
    if f is None:
        f = np.zeros((2, 1, 1))
    n = x_hat.shape[2]
    Ax = np.einsum('ij,tkj->tki', A[:, :n], x_hat[:-1])  # t-1, n_k, n
    Ax += np.einsum('ij,kj->ki', A[:, n:], z_hat)
    resid = x_hat[1:] - (Ax + f[:-1, :, :n])
    nll = 0.5 * ((resid[..., noise_inds] / sd_x) ** 2).sum(axis=(0, 2))  # n_k,
    resid_z = (z_hat - mu_z) / sd_z
    nll_z = 0.5 * (resid_z ** 2).sum(axis=1)
    return nll + nll_z


def step_update(x, P, a, b, sd):
    """
    Apply 'observation' of form
    a'x = b + N(0, sd^2)
    to obtain new x, P, useful for building priors
    :param x: n_k, n
    :param P: n_k, n, n
    :param a: n
    :param b: n_k,
    :param sd:
    :return:
    """
    PCt = P @ a  # n_k, n
    CPC_Q = PCt @ a + sd ** 2  # n_k,
    K = PCt / CPC_Q[:, np.newaxis]
    x_next = x + K * (b - x @ a)[:, np.newaxis]
    KCP_m1 = (PCt[..., np.newaxis]
              * K[:, np.newaxis]).transpose((0, 2, 1))
    P_next = P - KCP_m1
    return x_next, P_next
