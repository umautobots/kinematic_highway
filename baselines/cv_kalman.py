import numpy as np
import kinematic_model.kalman as ka
from time import time


DT = 0.1  # [s]
SD_DV_LAT = 0.05  # [m/s]
SD_DV_LON = 0.2  # [m/s]
SD_OBS = 0.1  # [m]
KF_EPS = 1e-8
KF_LARGE_VAR = 1e8
N_SAMPLES = 100


def predict(p, dataset_id, datafile_id, n_steps=0, **kwargs):
    """
    Predict assuming each vehicle moves at constant velocity subject
    to additive error with the given distributions.

    x_t = (p_t^{lat}, v^lat, p_t^{lon}, v^lon)

    :param p: t, n_agents, 2 | p [lat lon], both will be nan if not observed
    :param dataset_id:
    :param datafile_id:
    :param n_steps:
    :param kwargs:
    :return:
        p_samples: n_steps, n_vic, 2, n_samples
        w: n_vic, n_samples
    """
    t0 = time()
    n_vic = p.shape[1]
    x0 = np.zeros((n_vic, 4))
    P0 = np.zeros((n_vic, 4, 4))
    P0[:] = np.eye(4) * KF_LARGE_VAR
    A = np.array([
        [1, DT, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, DT],
        [0, 0, 0, 1],
    ])
    G = np.array([[0., 0], [1, 0], [0, 0], [0, 1]])
    R = np.diag([SD_DV_LAT**2, SD_DV_LON**2])
    C = np.array([
        [1., 0, 0, 0],
        [0, 0, 1, 0],
    ])
    Q = np.eye(2) * SD_OBS**2
    x_filtered, P_filtered = ka.multi_batch_filter_ti(
        p, x0, P0, A, G, R, C, Q)
    x_hat = np.zeros((1 + n_steps, n_vic, 4))
    P_hat = np.zeros((1 + n_steps, n_vic, 4, 4))
    x_hat[0] = x_filtered[-1]
    P_hat[0] = P_filtered[-1]
    GRG = G @ R @ G.T
    for i in range(n_steps):
        x_hat[i + 1] = x_hat[i] @ A.T
        P_hat[i + 1] = A @ P_hat[i] @ A.T + GRG
    # sample the normals
    L = np.linalg.cholesky(
        P_hat[1:, :, [0, 0, 2, 2], [0, 2, 0, 2]].reshape(n_steps, -1, 2, 2))
    p_samples = x_hat[1:, :, [0, 2], np.newaxis] + \
        L @ np.random.randn(n_steps, n_vic, 2, N_SAMPLES)
    w = np.ones((n_vic, N_SAMPLES)) / N_SAMPLES
    return p_samples, w, {'duration': time()-t0}
