"""
No-interaction version of kin_model.py for ablation study
"""
import numpy as np
from kinematic_model import lanes as lan, lateral_solution_missing as lat_model, lon_solution_missing as lon_model
from time import time


DT = 0.1
SD_DV_LON = 0.2  # [m/s] variation in v for lon
SD_DV_LAT = 0.05  # [m/s] for lateral dv deviation
SD_P_MERGE = 1.5  # [m] for prior on lateral target
K_STR = 100
SD_GV_PRIOR = np.array([2, 2.])
K_LON = K_STR


def predict_all(p, lane_edges, n_samples, n_steps):
    """

    :param p: n_obs, n_vic, 2 | nan where not observed (for all of last dimension)
    :param lane_edges:
    :param n_samples:
    :param n_steps:
    :return:
        p_hat: n_steps, n_vic, 2, 3n_samples | [i, :, j] = [lat lon] predicted by particle j
        w: n_vic, 3n_samples, | [j] = pr(particle j), sum over all particles equals 1
        dict:
    """
    t0 = time()
    is_obs_all = ~np.isnan(p[..., 0])
    n_obs, n_vic = p.shape[:2]
    n_samples_out = 3 * n_samples
    p_hat = np.zeros((n_steps, n_vic, 2, n_samples_out))
    w = np.zeros((n_vic, n_samples_out))

    n_steps_adjusted = np.zeros(n_vic, dtype=np.int)
    n_obs_adjusted = np.zeros(n_vic, dtype=np.int)
    pkpv_samples = np.zeros((n_vic, n_samples_out, 4))
    pv_x_gv_samples = np.zeros((n_vic, n_samples_out, 6))
    is_no_g_mask = np.zeros((n_vic, n_samples_out), dtype=np.bool)
    is_ignore_mask = np.zeros((n_vic, n_samples_out), dtype=np.bool)

    x_surround = np.zeros((n_obs, 0, 4))  # [lat p, lat v, lon p, lon v]
    # x_surround[:, :, 0] = p[..., 0]
    # x_surround[:, :, 2:] = smo.smooth_surround_LS_v1(p[..., 1], is_obs_all, SD_DV_LON)
    for i in range(n_vic):
        # other_vic_inds = np.hstack((np.arange(i), np.arange(i+1, n_vic)))
        other_vic_inds = np.array([], dtype=np.int)
        start_ind, end_ind = find_observation_period(is_obs_all[:, i])
        n_steps_adjusted[i] = n_steps + (p.shape[0] - end_ind)
        n_obs_adjusted[i] = end_ind - start_ind
        predict_single(
            p[start_ind:end_ind, i], is_obs_all[start_ind:end_ind, i],
            x_surround[start_ind:end_ind, other_vic_inds],
            is_obs_all[start_ind:end_ind, other_vic_inds], lane_edges, n_samples,
            w[i], pkpv_samples[i], pv_x_gv_samples[i], is_no_g_mask[i], is_ignore_mask[i]
        )

    n_steps_adjusted_br = np.broadcast_to(
        n_steps_adjusted[:, np.newaxis], (n_vic, n_samples_out)).reshape(-1)
    p_hat[:, :, 1] = lon_model.multi_forward_gv_samples(
        pv_x_gv_samples.reshape(-1, 6),
        SD_DV_LON, K_STR,
        n_steps_adjusted_br,
        is_no_g_mask.reshape(-1),
        is_ignore_mask.reshape(-1),
    )[-n_steps:, 0, :].reshape(n_steps, n_vic, n_samples_out)

    p_hat[:, :, 0] = lat_model.multi_forward_kpv_samples_v1(
        pkpv_samples.reshape(-1, 4),
        SD_DV_LAT, K_STR,
        np.broadcast_to(n_obs_adjusted[:, np.newaxis], (n_vic, n_samples_out)).reshape(-1),
        n_steps_adjusted_br,
    )[-n_steps:, 0, :].reshape(n_steps, n_vic, n_samples_out)
    return p_hat, w, {'duration': time()-t0}


def predict_single(
        p_ego, is_obs_ego, x_surround, is_obs, lane_edges, n_samples,
        total_w, pkpv_samples, pv_x_gv_samples, is_no_g_mask, is_ignore_mask):
    """
    Modify total_w, ..., is_ignore_mask to set for forward propagation.

    :param p_ego: n_obs, 2 | [lat lon] observations, both nan where not observed
        - assume: [0] and [-1] observed
    :param is_obs_ego: n_obs, | True iff p_ego[i] observed
    :param x_surround: n_obs, n_vic, 4 | [lat p, lat v, lon p, lon v]
        - lat is nan where not observed
        - lon is assumed estimated for entire duration
    :param is_obs: n_obs, n_vic | True iff x_surround[i, j] observed
    :param lane_edges:
    :param n_samples:
    :param n_steps:
    :return:
    """
    ind_str, ind_merge_pos, ind_merge_neg = lan.get_lane_targets(p_ego[0, 0], lane_edges)
    p_straight, p_merge_pos, p_merge_neg = lane_edges[[ind_str, ind_merge_pos, ind_merge_neg], 1]

    k_min = 0
    k_max = 120
    k_stride = 5
    k_grid = np.mgrid[k_min:k_max+1:k_stride]

    lat_nll = 0 * total_w
    pkpv_samples[:, 0] = p_ego[-1, 0]
    pkpv_samples[:, 1:], lat_nll[:] = lat_model.sample_multi_merge_kpv_v0(
        p_ego[:, 0], k_grid, SD_DV_LAT, np.array([p_straight, p_merge_pos, p_merge_neg]),
        SD_P_MERGE, K_STR, 3*n_samples
    )

    pv_x_gv_samples[:, 0] = p_ego[-1, 1]
    pv_x_gv_samples[:, 1:], lon_nll, is_no_g_mask[:] = lon_model.sample_lon_v_x_gv(
        p_ego[:, 1], is_obs_ego, x_surround[:, :, [2, 3, 0]], is_obs, lane_edges, ind_str,
        SD_GV_PRIOR, SD_DV_LON, K_LON, n_samples
    )
    mask = pv_x_gv_samples[:, 1] < 0
    pv_x_gv_samples[mask, 1] = 0.
    is_ignore_mask[:] = (lon_nll.reshape(-1) >= lon_model.LARGE_NLL)

    w = lat_nll + lon_nll
    w -= w.min()
    w = np.exp(-w)
    w /= w.sum()
    total_w[:] = w


def find_observation_period(is_obs):
    """
    Find indices such that is_obs[start:end] = [True, ..., True],
    with these indices being the first and last true values.
    Assume: such indices exist, and start < end
    :param is_obs: n,
    :return:
        start_ind
        end_ind
    """
    start_ind = np.arange(is_obs.size)[is_obs.cumsum() == 1][0]
    end_ind = np.arange(is_obs.size)[::-1][is_obs[::-1].cumsum() == 1][0] + 1
    return start_ind, end_ind
