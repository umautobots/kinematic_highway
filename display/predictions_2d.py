import numpy as np
from scipy import stats as ss
import matplotlib.pyplot as plt


def display_prediction_density(ax, p, w):
    # p: 2, n_samples | [lon, lat] position
    p_min = 1e-2
    jitter = np.random.randn(*p.shape) * np.array([[.02, 0]]).T
    kernel = ss.gaussian_kde(p + jitter, weights=w)
    xmin = p[0, :].min() - 6
    xmax = p[0, :].max() + 6
    ymin = p[1, :].min() - 6
    ymax = p[1, :].max() + 6
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel.evaluate(pos).T, X.shape)
    masked = np.ma.masked_where(Z < p_min, Z)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ctr = ax.imshow(
        np.rot90(masked), cmap='viridis',
        extent=[xmin, xmax, ymin, ymax], zorder=10,
        vmin=1e-2, aspect='auto',
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ctr


def display_predictions(
        p_obs, target_ind, p_true, prediction_methods,
        prediction_ind=-1,
        ego_ind=np.inf,
        data_title='', datafile_path=''):
    """

    :param p_obs: n_obs, n_vic, 2 | [lat lon]
    :param target_ind: index of target vehicle predicted in {0, ..., n_vic-1}
    :param p_true: n_pred, n_vic, 2 |
    :param prediction_methods: [i] -> for scenario i, the
        p_pred: n_pred, n_agents, 2, n_samples | predicted positions
        w: n_agents, n_samples | weight for each outcome summing to one
        dict: |
    :param prediction_ind: index of prediction made by method to use (-1 == last)
    :param ego_ind: index used as ego to produce occlusion scenario (if finite)
    :param data_title:
    :param datafile_path:
    :return:
    """
    n_methods = len(prediction_methods)
    n_vic = p_obs.shape[1]
    non_target_inds = np.hstack((np.arange(target_ind), np.arange(target_ind+1, n_vic)))
    # fig, ax = plt.subplots(n_methods, 1)
    fig, ax = plt.subplots(n_methods, 1, sharex='all', sharey='all')
    if n_methods == 1:
        ax = [ax]
    for axi in ax:
        axi.grid()
        axi.plot(p_obs[:, target_ind, 1], p_obs[:, target_ind, 0],
                 alpha=0.8, color='black', ls='', marker='+')
        axi.plot(p_true[:, target_ind, 1], p_true[:, target_ind, 0],
                 alpha=0.4, color='black', ls='', marker='+')

        if np.isfinite(ego_ind):
            axi.plot(p_obs[:, ego_ind, 1] + 1e-3, p_obs[:, ego_ind, 0] + 1e-3,
                     alpha=0.5, color='magenta', ls='', marker='+')

    ax[0].set_title(data_title)
    ax[0].set_xlabel('longitude [m]')
    ax[0].set_ylabel('lateral [m]')
    ax[0].plot(p_obs[:, non_target_inds, 1], p_obs[:, non_target_inds, 0],
               alpha=0.1, marker='x', color='blue')
    if np.isnan(p_obs).any():
        is_nan_endpoint = np.isnan(p_obs)
        is_nan_endpoint[:-1] |= is_nan_endpoint[1:]
        is_nan_endpoint[1:] |= is_nan_endpoint[:-1]
        p_obs_nan = p_obs.copy()
        p_obs_nan[~is_nan_endpoint] = np.nan
        ax[0].plot(p_obs_nan[:, non_target_inds, 1], p_obs_nan[:, non_target_inds, 0],
                   alpha=0.1, marker='o', ls='', color='blue')

    for i in range(len(prediction_methods)):
        p_pred, w_pred, _ = prediction_methods[i][prediction_ind]
        n_samples = w_pred.shape[1]

        p_pred = p_pred[:, target_ind]
        w_pred = w_pred[target_ind]
        p_pred = p_pred[..., w_pred > .5 / n_samples]
        w_pred = w_pred[w_pred > .5 / n_samples]
        w_pred /= w_pred.sum()

        alpha = 1. if w_pred.size == 1 else .1
        ax[i].plot(p_pred[:, 1], p_pred[:, 0],
                   alpha=alpha, color='blue', label=prediction_methods[i].name)

        if w_pred.size > 2:
            display_prediction_density(ax[i], p_pred[-1, [1, 0], :], w_pred)
            ax[i].plot(p_true[-1, target_ind, 1], p_true[-1, target_ind, 0],
                       alpha=0.8, marker='+', color='white', zorder=20)
    plt.show()


def format_example_title(i, df_path):
    return 'Scenario {} | {}'.format(i, df_path)
