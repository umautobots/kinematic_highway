import numpy as np
import pandas as pd
import os


def load_named_df(p, sep=','):
    df = pd.read_csv(p, sep=sep)
    df = df.astype({
        'frame_id': np.int,
        'agent_id': np.int,
    })
    return df


class AgentType(object):
    ped = 0
    vic = 1
    bicycle = 2
    type_id2name = {
        ped: 'ped',
        vic: 'vic',
        bicycle: 'bicycle',
    }
    type_list = [ped, vic, bicycle]


class DataframeInfo(object):

    def __init__(self, df, start_frame2valid_ids, frame_keys,
                 dataset_id, datafile_path):
        self.df = df
        self.start_frame2valid_ids = start_frame2valid_ids
        self.frame_keys = frame_keys
        self.dataset_id = dataset_id
        self.datafile_path = datafile_path


class TrajectoryTypeDataset(object):
    """
    For loading dataframes with columns [frame_id agent_id x y]
    and iterating over selections of the data.
    Properties (within dataframe):
    - each (frame_id, agent_id) is unique
    """
    def __init__(self, data_dir, n_obs, n_pred, sep=',', dataset_id=0,
                 dataset_file_ids=None, is_fill_nan=False, valid_ids_kwargs=None,
                 df_loader_fcn=None):
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
        df_loader_fcn = df_loader_fcn if df_loader_fcn else load_named_df

        data_names = [name for name in os.listdir(data_dir) if ('.txt' in name) or ('.csv' in name)]
        print('Begin processing of\n{}\nat {}\n'.format(data_names, data_dir))
        n_added = 0  # deb
        n_vic = 0  # deb
        for datafile_id, data_name in enumerate(data_names):
            if dataset_file_ids and datafile_id not in dataset_file_ids:
                continue
            print('Processing {}'.format(data_name))
            data_path = os.path.join(data_dir, data_name)
            df = df_loader_fcn(data_path, sep=sep)
            format_dataframe(df, is_raise=False, is_set_index=True)
            df = df[['agent_id', 'x', 'y']]  # 'frame_id' is the index
            start_frame2valid_ids = build_start_frame2valid_ids_indexed(
                df, self.n_seq_frames)
            frame_keys = np.array(list(start_frame2valid_ids.keys()))
            df_info = DataframeInfo(df, start_frame2valid_ids, frame_keys,
                                    dataset_id, data_path)
            self.df_list.append(df_info)
            n_added += len(frame_keys)
            n_vic += df['agent_id'].unique().size
            # print('{} at {} added'.format(data_name, n_added))
            # print('n_vic at {}'.format(n_vic))
            # if n_added > 5000:
            #     break

        self.n_batch_per_df = np.array([df_info.frame_keys.size for df_info in self.df_list])
        self.n_csum = np.hstack((0, self.n_batch_per_df.cumsum()))

    def __len__(self):
        return self.n_csum[-1]

    def get_df(self, index):
        """
        :param index: 
        :return: 
        - agent_data: n_frames, n_agents, 2
        - dataset_id: 
        - datafile_id:
        """
        df_ind = np.searchsorted(self.n_csum[1:], index, side='right')
        df_info = self.df_list[df_ind]
        i = index - self.n_csum[df_ind]
        frame = df_info.frame_keys[i]
        seq_df = df_info.df.loc[frame:frame + self.n_seq_frames - 1, :]
        valid_agent_ids = df_info.start_frame2valid_ids[frame]
        seq_df = seq_df[seq_df['agent_id'].isin(valid_agent_ids)]
        agent_data = sequence_df2array(seq_df, self.n_seq_frames, len(valid_agent_ids))
        return agent_data, df_info.dataset_id, df_ind

    def get_frame_info(self, index):
        df_ind = np.searchsorted(self.n_csum[1:], index, side='right')
        df_info = self.df_list[df_ind]
        i = index - self.n_csum[df_ind]
        frame = df_info.frame_keys[i]
        return df_info.datafile_path, frame


def build_start_frame2valid_ids_indexed(df, n_seq_frames):
    """
    Assume:
    - frame_ids are continuous, next frame is always +1
    - each agent_id appears at most once in each frame
    :param df: [frame_id(index) agent_id] exist
    :param n_seq_frames: sequence length for which agent must be observed entirely
    :return:
        start_frame2valid_ids: dict[frame_id] = tuple(agent_ids)
    """
    start_frame2valid_ids = {}
    frames = df.index.unique()
    n_added = 0
    for frame in frames:
        seq_df = df.loc[frame:frame+n_seq_frames-1, :]
        agent_ids = seq_df['agent_id'].unique()
        valid_agent_ids = tuple(
            agent_id for agent_id in agent_ids if
            n_seq_frames == (seq_df['agent_id'] == agent_id).sum()
        )
        if valid_agent_ids:
            start_frame2valid_ids[frame] = valid_agent_ids
            n_added += 1
        if n_added % 1000 == 0:
            print('    Processed {} scenarios'.format(n_added))
    return start_frame2valid_ids


def sequence_df2array(df, n_frames, n_agents):
    """
    0) reset index on frame_id to be a column, then group by both columns
    1) make multi index with frame_id and then group by both indices
    - but these only group by, and we need to use the values as indices in ndarray
    - instead: reset index and sort by both columns to prepare for reshape
    :param df: [frame_id(index) agent_id x y] with n_frames * n_agents coordinates
    :param n_frames: 
    :param n_agents: 
    :return: n_frames, n_agents, 2
    """
    arr = df.reset_index().sort_values(by=['frame_id', 'agent_id']).values[:, 2:]
    return arr.reshape((n_frames, n_agents, 2))


def concat_datasets(dataset_list):
    """
    :param dataset_list: of TrajectoryTypeDataset
    :return: modified dataset_list[0]
    """
    ret_dataset = dataset_list[0]
    for dataset in dataset_list[1:]:
        ret_dataset.df_list.extend(dataset.df_list)
        ret_dataset.n_batch_per_df = np.hstack((
            ret_dataset.n_batch_per_df, dataset.n_batch_per_df))
    ret_dataset.n_csum = np.hstack((0, ret_dataset.n_batch_per_df.cumsum()))
    return ret_dataset


def format_dataframe(df, is_raise=False, is_set_index=False):
    """
    :param df: exist [frame_id]
     modifies df to
     - start at frame_id zero
     - have frame_id such that next frame_id is +1
    :param is_raise: 
    :param is_set_index: sets frame_id as index
    """
    frames = np.unique(df['frame_id'].values)
    if frames[0] != 0 and is_raise:
        raise ValueError('frame_id starts at {} instead of 0'.format(frames[0]))
    df['frame_id'] -= frames[0]
    frames -= frames[0]
    is_skip_bad = (frames[1:] - frames[:-1] != 1).any()
    if is_skip_bad and is_raise:
        dif = frames[1:] - frames[:-1]
        raise ValueError('frame_id has gap > 1 at: {}'.format(dif[dif != 1])[0])
    elif is_skip_bad:
        map_dict = {frame: i for i, frame in enumerate(frames)}
        df['frame_id'] = df['frame_id'].map(map_dict)
    if is_set_index:
        df.set_index('frame_id', inplace=True)
        df.sort_index(inplace=True)  # since frame is likely non-unique index


def resample_dataset(df, src_dt, dest_dt):
    """
    Interpolate data collected at src_dt sampling interval
    to dest_dt sampling interval
    :param df: TrajectoryTypeDataset
    - so frame_id starts at zero
    - following frame_id is +1
    :param src_dt: 
    :param dest_dt: 
    :return: resampled TrajectoryTypeDataset
    """
    df_list = []
    agent_ids = df['agent_id'].unique()
    for agent_id in agent_ids:
        df_agent = df.loc[df['agent_id'] == agent_id, :].sort_values(by=['frame_id'])
        old_frame_id = df_agent['frame_id'].values
        bounds = old_frame_id[[0, -1]] * src_dt/dest_dt
        bounds = np.array([np.ceil(bounds[0]), np.floor(bounds[1])]).astype(np.int)
        new_frame_id = np.arange(bounds[0], bounds[1]+1)
        x = np.interp(new_frame_id*dest_dt, old_frame_id*src_dt, df_agent['x'].values)
        y = np.interp(new_frame_id*dest_dt, old_frame_id*src_dt, df_agent['y'].values)
        df_i = pd.DataFrame(new_frame_id, columns=['frame_id'])
        df_i['agent_id'] = agent_id
        df_i['x'] = x
        df_i['y'] = y
        df_list.append(df_i)
    resampled_df = pd.concat(df_list, ignore_index=True)
    return resampled_df
