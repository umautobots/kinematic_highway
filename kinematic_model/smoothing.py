import numpy as np
from kinematic_model import kalman as ka

DT = 0.1
EPS = 1e-8


def smooth_surround_LS_v0(p, is_obs, sd_dv):
    """
    Smooth via LS on each vic in a loop.
    :param p: n, n_vic
    :param is_obs: n, n_vic
    :param sd_dv:
    :return:
        x: n, n_vic, 2
    """
    n, n_vic = p.shape
    x = np.zeros((n, n_vic, 2))
    for i in range(n_vic):
        x[:, i] = smooth_single_vic_LS_v0(p[:, i], is_obs[:, i], sd_dv)
    return x


def smooth_surround_LS_v1(p, is_obs, sd_dv):
    """
    Smooth via LS on each vic in a loop.
    :param p: n, n_vic
    :param is_obs: n, n_vic
    :param sd_dv:
    :return:
        x: n, n_vic, 2
    """
    n, n_vic = p.shape
    x0 = np.zeros((n_vic, 2))
    P0 = np.zeros((n_vic, 2, 2))
    P0[:] = np.eye(2) * 1e8
    A = np.array([[1, DT], [0, 1]])
    G = np.array([[EPS], [1.]])
    R = np.array([[sd_dv ** 2]])
    c_ind = 0
    Q = EPS
    x_hat = ka.multi_batch_smooth_ti_select_single(
        p, x0, P0, A, G, R, c_ind, Q)[0]
    return x_hat


def smooth_single_vic_LS_v0(p, is_obs, sd_dv):
    """
    theta = (x_1; ...; x_n) : (2n,)
    :param p: n,
    :param is_obs: n,
    :param sd_dv:
    :return:
        x: n, 2
    """
    n = p.size
    n_th = 2*n
    # selecting (p_2, ..., p_n): A_2
    A2 = np.zeros((n - 1, n_th))
    A2[np.arange(n - 1), 2 * np.arange(1, n)] = 1.
    # selecting (v_2, ..., v_n): A_4
    A4 = np.zeros((n - 1, n_th))
    A4[np.arange(n - 1), 2 * np.arange(1, n) + 1] = 1.
    # dynamics: predicting (x_2; ...; x_n) from previous x_i
    A1 = np.zeros((n - 1, n_th))
    A3 = np.zeros((n - 1, n_th))
    # x_{i+1} = (1 t; 0 1) x_i part
    A1[np.arange(n-1), 2 * np.arange(n-1)] = 1.
    A1[np.arange(n-1), 2 * np.arange(n-1) + 1] = DT
    A3[np.arange(n-1), 2 * np.arange(n-1) + 1] = 1.
    A = (A3 - A4) / sd_dv
    AtA = A.T @ A
    m = is_obs.sum()
    C = np.zeros((n-1+m, n_th))
    C[:n-1, :n_th] = A1 - A2
    C[n-1 + np.arange(m), 2 * np.arange(n)[is_obs]] = 1.

    n_c = n-1+m
    L1 = np.zeros((n_th + n_c, n_th + n_c))
    L1[:n_th, :n_th] = AtA
    L1[n_th:, :n_th] = C[:n_c, :]
    L1[:n_th, n_th:] = C[:n_c, :].T
    L2 = np.zeros(n_th + n_c)
    L2[-m:] = p[is_obs]
    L1_inv = np.linalg.pinv(L1)
    theta_z_hat = L1_inv.dot(L2)
    x = theta_z_hat[:n_th].reshape(-1, 2)
    return x


def smooth_single_vic_LS_v1(p, is_obs, sd_dv):
    """
    theta = (x_1; ...; x_n) : (2n,)
    :param p: n,
    :param is_obs: n,
    :param sd_dv:
    :return:
        x: n, 2
    """
    x0 = np.array([p[is_obs][0], 0.])
    P0 = np.eye(2) * 1e8
    A = np.array([[1, DT], [0, 1]])
    G = np.array([[EPS], [1.]])
    R = np.array([[sd_dv]])
    C = np.array([[1., 0]])
    Q = np.array([[EPS]])
    x_hat, P_hat = ka.batch_smooth_ti(p.reshape(-1, 1), x0, P0, A, G, R, C, Q)
    return x_hat


def smooth_single_vic_LS_v2(p, is_obs, sd_dv):
    """
    theta = (x_1; ...; x_n) : (2n,)
    :param p: n,
    :param is_obs: n,
    :param sd_dv:
    :return:
        x: n, 2
    """
    x0 = np.array([0, 0.])
    P0 = np.eye(2) * 1e8
    A = np.array([[1, DT], [0, 1]])
    G = np.array([[EPS], [1.]])
    R = np.array([[sd_dv]])
    c_ind = 0
    Q = EPS
    x_hat, P_hat = ka.batch_smooth_ti_select_single(p, x0, P0, A, G, R, c_ind, Q)
    return x_hat


def main_smooth_single_vic_LS():
    seed = np.random.randint(0, 1000)
    # seed = 1
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    m = 60
    sd_dv = 0.05
    p = (sd_dv * np.random.randn(m)).cumsum()
    is_obs = ~(np.arange(m) % 3 == 0)
    p[~is_obs] = np.nan

    # p = np.array([np.nan, 0., 1, np.nan, np.nan, 4., np.nan])
    # is_obs = ~np.isnan(p)

    args = (p, is_obs, sd_dv)
    x_true = smooth_single_vic_LS_v0(*args)
    x_hat = smooth_single_vic_LS_v2(*args)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))
    # print(x_true)
    # print(x_hat)

    from timeit import timeit

    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=smooth_single_vic_LS_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=smooth_single_vic_LS_v2, args=args))/n_tries)


def main_smooth_surround_LS():
    seed = np.random.randint(0, 1000)
    # seed = 1
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    m = 30
    n_vic = 5
    sd_dv = 0.05
    p = (sd_dv * np.random.randn(m, n_vic)).cumsum(axis=0)
    is_obs = np.random.rand(m, n_vic) < 0.8
    p[~is_obs] = np.nan

    args = (p, is_obs, sd_dv)
    x_true = smooth_surround_LS_v0(*args)
    x_hat = smooth_surround_LS_v1(*args)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))
    # print(x_true)
    # print(x_hat)

    from timeit import timeit

    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=smooth_surround_LS_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=smooth_surround_LS_v1, args=args))/n_tries)


if __name__ == '__main__':
    # main_smooth_single_vic_LS()
    main_smooth_surround_LS()














