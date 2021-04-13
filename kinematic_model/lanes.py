import numpy as np


def lane_from_lat(lat, lane_edges):
    """
    Retrieve lane indices for lateral coordinates
    :param lat: n,
    :param lane_edges: m+1, 2 | for m lanes
        [i] = lat for top (low x) edge for lane i [m], lat of center line for lane i [m]
        - lane edges increase in lateral value with i
    :return:
        lane_inds: n, | ind of center line lateral for each x
    """
    lane_inds = np.searchsorted(lane_edges[:, 0], lat, side='left') - 1
    return lane_inds


def get_lane_targets(lat0, lane_edges):
    """
    Retrieve lane index of current lane, and those above and below
    :param lat0: lateral position at initial time
    :param lane_edges: m+1, 2 | for m lanes
        [i] = lat for top (low x) edge for lane i [m], lat of center line for lane i [m]
        - lane edges increase in lateral value with i
    :return:
        p_merge_pos, p_merge_neg | lateral values for merging in
            positive and negative lateral directions
    """
    lane_ind = np.searchsorted(lane_edges[:, 0], lat0, side='left') - 1
    assert (lane_ind > 0) and (lane_ind < lane_edges.shape[0] - 1)
    return lane_ind, lane_ind + 1, lane_ind - 1
