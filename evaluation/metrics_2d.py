import numpy as np


def get_point_quantile_expected_dist_by_time_fcns(quantile=0.2, select_inds=np.arange(9, 60, 10)):
    # Base on ordering of overall *point* distance rather than
    # *trajectory* distance at each selected time
    # (point allows for switching best sample index at each timestep).
    # q: quantile \in [0, 1) | to find the distance at the qth
    #   quantile (lower = closer)
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of qth percentile distances from previous examples
            list[i] = si, n_agents
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1 along n_samples,
        :param y_true: n_steps, n_agents, 2
        :return:
        """
        # si, n_agents, 2, n_samples
        difs = y_hats[select_inds, ...] - np.expand_dims(y_true[select_inds, ...], -1)
        dists = np.sqrt((difs ** 2).sum(axis=2))  # si, n_agents, n_samples

        sort_inds = np.argsort(dists, axis=-1)  # si, n_agents, n_samples
        dists_sorted = np.take_along_axis(dists, sort_inds, axis=-1)
        si, n_agents, n_samples = sort_inds.shape
        j_inds = np.broadcast_to(np.arange(n_agents)[np.newaxis, :, np.newaxis],
                                 sort_inds.shape).ravel()
        p_sorted = p[j_inds, sort_inds.ravel()].reshape(si, n_agents, n_samples)

        gt_q_mask = p_sorted.cumsum(axis=-1) >= quantile
        selector_mask = gt_q_mask.cumsum(axis=-1) == 1  # one True per [i, j]
        q_dists = dists_sorted[selector_mask].reshape(si, n_agents)
        accumulator.append(q_dists)

    def reduce_fcn(accumulator):
        q_dists = np.concatenate(accumulator, axis=1)  # si, total agents
        return q_dists.mean(axis=1)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_expected_dist_by_time_fcns(select_inds=np.arange(9, 60, 10)):
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of average distances from previous examples
            list[i] = si, n_agents
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1
        :param y_true: n_steps, n_agents, 2
        :return:
        """
        # si, n_agents, 2, n_samples
        difs = y_hats[select_inds, ...] - np.expand_dims(y_true[select_inds, ...], -1)
        dists = np.sqrt((difs ** 2).sum(axis=2))  # si, n_agents, n_samples
        expected_dist = np.einsum('ijk,jk->ij', dists, p)  # si, n_agents
        accumulator.append(expected_dist)

    def reduce_fcn(accumulator):
        expected_dists = np.concatenate(accumulator, axis=1)  # si, total agents
        return expected_dists.mean(axis=1)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_rmse_by_time_fcns(select_inds=np.arange(9, 60, 10)):
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of squared distances from previous examples
            list[i] = si, n_agents
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1
        :param y_true: n_steps, n_agents, 2
        :return:
        """
        # si, n_agents, 2, n_samples
        difs = y_hats[select_inds, ...] - np.expand_dims(y_true[select_inds, ...], -1)
        dists = (difs ** 2).sum(axis=2)  # si, n_agents, n_samples
        expected_dist = np.einsum('ijk,jk->ij', dists, p)  # si, n_agents
        accumulator.append(expected_dist)

    def reduce_fcn(accumulator):
        expected_dists = np.concatenate(accumulator, axis=1)  # si, total agents
        return np.sqrt(expected_dists.mean(axis=1))
    return init_fcn, accumulate_fcn, reduce_fcn


def get_timing_fcns():
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, duration=np.nan, **kwargs):
        """
        :param accumulator: list of (duration, n_agents)
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1
        :param y_true: n_steps, n_agents, 2
        :param duration: time taken
        :return:
        """
        accumulator.append((duration, y_true.shape[1]))

    def reduce_fcn(accumulator):
        duration_n_agents = np.array(accumulator)
        return duration_n_agents[:, 0].mean()
    return init_fcn, accumulate_fcn, reduce_fcn
