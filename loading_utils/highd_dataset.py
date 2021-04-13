"""
Frame of raw data:

  ---------> x (longitudinal) [meters]
 |
 |
 v
 y (lateral) [meters]

"""
import os
from glob import glob
import numpy as np
import pandas as pd
import loading_utils.tt_dataset as tt
from loading_utils.constants import DATASETS_ROOT


EXTRA_LANE_WIDTH_M = 3.8


class DrivingDirection(object):
    """
    Tags to select vehicles traveling on upper lanes (left)
    or lower lanes (right, so originally positive x).
    """
    upper = 1
    lower = 2


def get_lane_edges_from_tt(raw_data_folder_path, tt_datafile_path):
    direction_tag = int(tt_datafile_path[-5])
    name = os.path.basename(tt_datafile_path)[:2]
    record_path = os.path.join(
        raw_data_folder_path, '{}_recordingMeta.csv'.format(name))
    df = pd.read_csv(record_path, header=0, sep=',')
    if direction_tag == DrivingDirection.upper:
        direction_str = 'upperLaneMarkings'
    else:
        direction_str = 'lowerLaneMarkings'
    edges = np.array(df[direction_str][0].split(';')) \
        .astype(np.float)
    lane_edges = make_lane_edges_from_lane_boundaries(edges)
    return lane_edges


def make_lane_edges_from_lane_boundaries(lb):
    """
    :param lb: m edges corresponding to m-1 lanes
    :return:
        lane_edges: m+2*n_extra, 2 | assume drivers do not move move than
            n_extra lanes outside of provided 'real' lanes
    """
    n_extra = 3
    edges = np.hstack((
        lb[0] - np.arange(1, n_extra+1)[::-1] * EXTRA_LANE_WIDTH_M,
        lb,
        lb[-1] + np.arange(1, n_extra+1) * EXTRA_LANE_WIDTH_M
    ))
    lane_edges = np.zeros((edges.size, 2))
    lane_edges[:, 0] = edges
    lane_edges[:-1, 1] = (lane_edges[1:, 0] + lane_edges[:-1, 0]) / 2
    return lane_edges


class HighdDataset(object):
    """
    Measurement frequency = 25Hz
    For each recording numbered XX (eg 01), we have the files:
    - XX_tracks.csv
    - XX_tracksMeta.csv
    - XX_recordingMeta.csv
    - XX_highway.png
    """
    FOLDER = 'highd-dataset-v1.0/data'
    TT_FORMAT_FOLDER = ''
    RAW_DT = 1./25
    GLOB_STR = '*_tracks.csv'

    def __init__(self, direction_tag=DrivingDirection.upper):
        """
        :param direction_tag: load only vehicles moving in this direction
            - and format as NGSIM frame with positive motion = (+) longitude
        """
        self.direction_tag = direction_tag
        self.location_ids = np.arange(1, 6+1)

    @staticmethod
    def load_raw(p, direction_tag=DrivingDirection.upper):
        """
        Load track data, including only agents with matching direction tag
        :param p: path to track data
        :param direction_tag:
        :return:
        """
        p_meta = p.replace('tracks.', 'tracksMeta.')
        df_meta = pd.read_csv(p_meta, header=0, sep=',')
        tagged_agent_ids = np.unique(
            df_meta.loc[df_meta['drivingDirection'] == direction_tag, 'id'].values)
        df = pd.read_csv(p, header=0, sep=',')
        return df[df['id'].isin(tagged_agent_ids)]

    @staticmethod
    def raw2tt(df, offset_agent_id=0, direction_tag=DrivingDirection.upper):
        df['centered_y'] = df['y'] + 0.5 * df['height']
        df['ngsim_x'] = df['centered_y']
        if direction_tag == DrivingDirection.upper:
            x_sign = -1.
        else:
            x_sign = 1.
        df['ngsim_y'] = x_sign * df['x']
        df = df[['frame', 'id', 'ngsim_x', 'ngsim_y']]
        df = df.rename(columns={
            'frame': 'frame_id',
            'id': 'agent_id',
            'ngsim_x': 'x',
            'ngsim_y': 'y',
        }, inplace=False)
        df = df.astype({'frame_id': int, 'agent_id': int})
        df['agent_id'] += offset_agent_id
        tt.format_dataframe(df, is_raise=False)
        return df

    def get_recordings(self):
        """
        Load only tracks for locations in location_ids
        :return:
            name: name of file to later be saved in tt format as name.txt
            track_path: absolute path to raw trajectory file
        """
        glob_str = os.path.join(
            DATASETS_ROOT, self.FOLDER,
            self.GLOB_STR
        )
        for track_path in glob(glob_str):
            recording_path = track_path.replace('tracks.csv', 'recordingMeta.csv')
            df_recording = pd.read_csv(recording_path, header=0, sep=',')
            recording_location_id = df_recording['locationId'].values[0]
            if recording_location_id not in self.location_ids:
                continue
            name = os.path.basename(track_path)
            name = name.replace('.csv', '')
            yield name, track_path

    def load_as_trajectorytype_format(self, recording_path):
        df = self.load_raw(recording_path, direction_tag=self.direction_tag)
        df = self.raw2tt(df, direction_tag=self.direction_tag)
        return df

    @staticmethod
    def make_all_lane_edges(dataset):
        """
        :param dataset: n datafiles' worth of data
            - each its own path
        :return:
            all_lane_edges: n list | [i] = n_i, 2 lane edges
        """
        n = len(dataset.df_list)
        all_lane_edges = []
        raw_data_folder_path = os.path.join(DATASETS_ROOT, HighdDataset.FOLDER)
        for i in range(n):
            tt_datafile_path = dataset.df_list[i].datafile_path
            all_lane_edges.append(get_lane_edges_from_tt(
                raw_data_folder_path, tt_datafile_path))
        return all_lane_edges

    @staticmethod
    def get_lane_edges(all_lane_edges, dataset_id, datafile_id):
        """
        :param all_lane_edges: n, n_lanes+1, 2 | all lane bounds used for both i-80 and us-101
        :param dataset_id:
        :param datafile_id: index in {0, ..., n}
        :return:
        """
        return all_lane_edges[datafile_id]


class HighdLocations13Dataset(HighdDataset):
    """
    Locations 1-3
    """
    TT_FORMAT_FOLDER = 'tt_format/10hz/highd/locations1-3'

    def __init__(self, **kwargs):
        super(HighdLocations13Dataset, self).__init__(**kwargs)
        self.location_ids = np.array([1, 2, 3])


class HighdLocations46Dataset(HighdDataset):
    """
    Locations 4-6
    """
    TT_FORMAT_FOLDER = 'tt_format/10hz/highd/locations4-6'

    def __init__(self, **kwargs):
        super(HighdLocations46Dataset, self).__init__(**kwargs)
        self.location_ids = np.array([4, 5, 6])
