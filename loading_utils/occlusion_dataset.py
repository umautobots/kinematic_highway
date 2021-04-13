import os
import numpy as np
import pandas as pd
import loading_utils.tt_dataset as tt


MIN_OBS = 10  # [frames at 10Hz]
MAX_LON_DIST = 50  # [m]
AGENT_OCCLUSION_RADIUS = 2.  # [m]


class DataframeInfo(object):

    def __init__(self, df, start_frame_ego2valid_ids_p_occluded,
                 dataset_id, datafile_path):
        """

        :param df:
        :param start_frame_ego2valid_ids_p_occluded: dict
            (start_frame, ego_id) ->
                valid_ids: n_agents, | sorted order
                p_occluded: n_obs, n_agents, 2 | positions of valid agents in observation window
                    - filled with nan where "occluded"
        :param dataset_id:
        :param datafile_path:
        """
        self.df = df
        self.start_frame_ego2valid_ids_p_occluded = start_frame_ego2valid_ids_p_occluded
        self.keys = list(start_frame_ego2valid_ids_p_occluded.keys())
        self.dataset_id = dataset_id
        self.datafile_path = datafile_path


class OccludedTrajectoryTypeDataset(object):
    """
    For loading dataframes with columns [frame_id agent_id x y]
    and iterating over selections of the data.
    Properties (within dataframe):
    - each (frame_id, agent_id) is unique
    """
    def __init__(self, data_dir, n_obs, n_pred, sep=',', dataset_id=0,
                 dataset_file_ids=None, valid_ids_kwargs=None,
                 df_loader_fcn=None, **kwargs):
        """
        :param data_dir: absolute path to data directory
        :param n_obs:
        :param n_pred:
        :param sep:
        :param dataset_id:
        :param dataset_file_ids:
        :param is_fill_nan:
        :param valid_ids_kwargs: dict
        :param df_loader_fcn:
        """
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.n_seq_frames = n_obs + n_pred
        self.df_list = []
        df_loader_fcn = df_loader_fcn if df_loader_fcn else tt.load_named_df

        data_names = [name for name in os.listdir(data_dir) if ('.txt' in name) or ('.csv' in name)]
        print('Begin processing of\n{}\nat {}\n'.format(data_names, data_dir))
        for datafile_id, data_name in enumerate(data_names):
            if dataset_file_ids and datafile_id not in dataset_file_ids:
                continue
            print('Processing {}'.format(data_name))
            data_path = os.path.join(data_dir, data_name)
            df = df_loader_fcn(data_path, sep=sep)
            tt.format_dataframe(df, is_raise=False, is_set_index=True)
            df = df[['agent_id', 'x', 'y']]  # 'frame_id' is the index
            start_frame_ego2valid_ids_is_occluded = \
                build_start_frame_ego2valid_ids_p_occluded_indexed(
                    df, n_obs, self.n_seq_frames)
            df_info = DataframeInfo(
                df, start_frame_ego2valid_ids_is_occluded, dataset_id, data_path)
            self.df_list.append(df_info)
        self.n_batch_per_df = np.array([len(df_info.keys) for df_info in self.df_list])
        self.n_csum = np.hstack((0, self.n_batch_per_df.cumsum()))

    def __len__(self):
        return self.n_csum[-1]

    def get_df(self, index):
        """
        Include partially observed agents, filling unobserved frames with nan
        - all agents must be recorded for entirety of prediction window
        - occlusions (or non-recording) may be present during observation window
        :param index:
        :return:
        - agent_data: n_frames, n_agents, 2
        - dataset_id:
        - datafile_id:
        """
        df_ind = np.searchsorted(self.n_csum[1:], index, side='right')
        df_info = self.df_list[df_ind]
        i = index - self.n_csum[df_ind]
        key = df_info.keys[i]
        frame, ego_id = key
        seq_df = df_info.df.loc[frame:frame + self.n_seq_frames - 1, :]
        valid_agent_ids, p_occluded = df_info.start_frame_ego2valid_ids_p_occluded[key]
        seq_df = seq_df[seq_df['agent_id'].isin(valid_agent_ids)]
        n_agents = len(valid_agent_ids)
        agent_data = np.zeros((self.n_seq_frames, n_agents, 2))
        agent_data[:self.n_obs] = p_occluded
        agent_data[self.n_obs:] = tt.sequence_df2array(
            seq_df.loc[frame + self.n_obs:, ['agent_id', 'x', 'y']], self.n_pred, n_agents)
        return agent_data, df_info.dataset_id, df_ind

    def get_frame_info(self, index):
        df_ind = np.searchsorted(self.n_csum[1:], index, side='right')
        df_info = self.df_list[df_ind]
        i = index - self.n_csum[df_ind]
        frame, ego_id = df_info.keys[i]
        valid_agent_ids = df_info.start_frame_ego2valid_ids_p_occluded[(frame, ego_id)][0]
        ego_ind = np.arange(valid_agent_ids.size)[valid_agent_ids == ego_id][0]
        return df_info.datafile_path, frame, ego_id, ego_ind


def build_start_frame_ego2valid_ids_p_occluded_indexed(df, n_obs, n_seq_frames):
    """
    Include partially observed agents, filling unobserved frames with nan
    - all agents must be recorded for entirety of prediction window
    - occlusions (or non-recording) may be present during observation window
    - must be non-occluded for a minimum number of frames
    :param df:
    :param n_obs:
    :param n_seq_frames:
    :return:

    """
    start_frame_ego2valid_ids_p_occluded = {}
    all_agent_ids = np.unique(df['agent_id'].values)
    n_added = 0
    for ego_id in all_agent_ids:
        ego_frames = np.unique(df[df.agent_id == ego_id].index)
        for frame in ego_frames[:-n_seq_frames]:
            seq_df = df.loc[frame:frame + n_seq_frames - 1, :]
            # sorted values
            frames = np.unique(seq_df.index.values)
            agent_ids = np.unique(seq_df['agent_id'].values)
            agent_ids2id_index = np.zeros(agent_ids[-1] + 1 - agent_ids[0], dtype=np.int)
            agent_ids2id_index[agent_ids - agent_ids[0]] = np.arange(agent_ids.size)
            # make occluded_array: n_seq, n_agents, 2 | may have nan at any position
            nan_df = pd.DataFrame(np.nan, index=range(n_seq_frames * agent_ids.size),
                                  columns=['x', 'y'])
            # index of existing data in full nan df
            # - sorted by ['frame_id', 'agent_id']
            inds = agent_ids2id_index[seq_df.agent_id.values - agent_ids[0]] * n_seq_frames +\
                (seq_df.index.values - frames[0])
            nan_df.loc[inds] = seq_df.values[:, 1:]
            occluded_data = nan_df.values.reshape((agent_ids.size, n_seq_frames, 2))
            occluded_data = np.transpose(occluded_data, axes=(1, 0, 2))
            original_nan_mask = np.isnan(occluded_data)
            occluded_data[original_nan_mask] = -10e5  # avoid occlusions for sight filtering

            # filter by lon: n_obs, n_agents
            ego_ind = agent_ids2id_index[ego_id - agent_ids[0]]
            is_lon_out = np.abs(occluded_data[:n_obs, :, 1] -
                                occluded_data[:n_obs, [ego_ind], 1]) > MAX_LON_DIST

            # filter by sight: n_obs, n_agents
            is_occluded = _time_separated_pt2line_min_dists(
                occluded_data[:n_obs],
                np.broadcast_to(occluded_data[:n_obs, [ego_ind], :], (n_obs, agent_ids.size, 2)),
                occluded_data[:n_obs],
            ) < AGENT_OCCLUSION_RADIUS  # n_obs, n_agents, n_agents
            # - no self-occlusions
            is_occluded[:, np.mgrid[:agent_ids.size], np.mgrid[:agent_ids.size]] = False
            # - no occlusion by ego
            is_occluded[:, ego_ind, :] = False
            is_occluded = is_occluded.any(axis=1)
            is_occluded[:, ego_ind] = False  # other vehicles can never occlude ego
            # print('occ nan:', agent_ids.size, is_occluded.sum())

            mask = np.zeros((n_seq_frames, agent_ids.size), dtype=np.bool)
            mask[:n_obs] = is_lon_out | is_occluded
            occluded_data[mask] = np.nan
            occluded_data[original_nan_mask] = np.nan

            is_preds_recorded = (~np.isnan(occluded_data[n_obs:, :, 0])).all(axis=0)
            is_min_frames = (~np.isnan(occluded_data[:n_obs, :, 0])).sum(axis=0) > MIN_OBS
            valid_agent_mask = is_preds_recorded & is_min_frames  # n_agents,
            if not valid_agent_mask.any():
                continue

            valid_ids = agent_ids[valid_agent_mask]
            occluded_data = occluded_data[:, valid_agent_mask]
            p_occluded = occluded_data[:n_obs]
            key = (frame, ego_id)
            start_frame_ego2valid_ids_p_occluded[key] = (valid_ids, p_occluded)

            # print(len(valid_ids), agent_ids.size)
            n_added += 1
            if n_added % 200 == 0:
                print('    Processed {} scenarios'.format(n_added))
            # if n_added > 200:
            #     break  # deb
        # if n_added > 200:
        #     break  # deb
    return start_frame_ego2valid_ids_p_occluded


def _time_separated_pt2line_min_dists(axy, b0xy, b1xy):
    """
    Find distances only between points and lines with same t index
    For example:
        axy = agent positions (t, n_agents, 2)
        b0xy = ego position at t broadcasted to #agents (t, n_agents, 2)
        b1xy = agent positions (t, n_agents, 2)
    :param axy: t, n, 2
    :param b0xy: t, k, 2
    :param b1xy: t, k, 2
    :return:
        dists: t, n, k | [i, j] = distance from point i to segment j
    """
    line_vectors = b1xy - b0xy  # t, k, 2
    pt2start_vectors = axy[:, :, np.newaxis] - b0xy[:, np.newaxis]  # t, n, k, 2
    dots = (line_vectors[:, np.newaxis] * pt2start_vectors).sum(axis=3)  # t, n, k  #deb
    mags = (line_vectors ** 2).sum(axis=2)  # t, k
    mags[mags < 1e-8] = 1e-8
    t = dots / mags[:, np.newaxis]  # t, n, k
    np.clip(t, 0, 1, out=t)
    proj = b0xy[:, np.newaxis] + t[..., np.newaxis] * line_vectors[:, np.newaxis]  # t, n, k, 2
    rej = axy[:, :, np.newaxis] - proj
    sq_dist = (rej ** 2).sum(axis=3)  # t, n, k
    return np.sqrt(sq_dist)


def _pt2line_min_dists(axy, b0xy, b1xy):
    """
    Find distance of each point to each of k line segments
    :param axy: n, 2
    :param b0xy: k, 2
    :param b1xy: k, 2
    :return:
        dists: n, k | [i, j] = distance from point i to segment j
    """
    line_vectors = b1xy - b0xy                      # k, 2
    pt2start_vectors = axy - b0xy[:, np.newaxis]    # k, n, 2
    dots = (line_vectors[:, np.newaxis] * pt2start_vectors).sum(axis=2)  # k, n
    mags = (line_vectors ** 2).sum(axis=1)  # k,
    mags[mags < 1e-8] = 1e-8  # for zero length line segments
    t = dots.T/mags     # n, k
    t = np.clip(t, 0, 1)
    proj = b0xy.T + (t[:, np.newaxis] * line_vectors.T)  # n, 2, k
    rej = axy[:, :, np.newaxis] - proj                 # n, 2, k
    sq_dists = (rej ** 2).sum(axis=1)  # n, k
    return np.sqrt(sq_dists)
