"""
Frame of raw data:

  ---------> y (longitudinal) [meters]
 |
 |
 v
 x (lateral) [meters]

"""
import os
from glob import glob
import numpy as np
import pandas as pd
import loading_utils.tt_dataset as tt
from loading_utils.constants import DATASETS_ROOT


LANE_WIDTH_M = 3.7


def load_df(path, **kwargs):
    df = pd.read_csv(path, header=0, sep=',')
    df = convert_ft2meters(df)
    df = df.rename(columns={
        'Frame_ID': 'frame_id',
        'Vehicle_ID': 'agent_id',
        'Local_X': 'x',
        'Local_Y': 'y',
    }, inplace=False)
    df = df[['frame_id', 'agent_id', 'x', 'y']]
    df = df.astype({'frame_id': int, 'agent_id': int})
    tt.format_dataframe(df, is_raise=False)
    return df


def convert_ft2meters(df):
    df[['Local_X', 'Local_Y', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc', 'Space_Headway']] *= 0.3048
    return df


def make_lane_edges():
    far_right_id = 6
    lane_guides = np.array([i * LANE_WIDTH_M + 1. for i in range(far_right_id + 1)])
    lane_edges = np.zeros((lane_guides.size + 6, 2))
    lane_edges[[0, 1, 2, -3, -2, -1], 0] = lane_guides[[0, 0, 0, -1, -1, -1]] + [-12, -8, -4, 4, 8, 12]
    lane_edges[3:-3, 0] = lane_guides
    lane_edges[:-1, 1] = (lane_edges[1:, 0] + lane_edges[:-1, 0]) / 2
    return lane_edges


class NgsimDataset(object):
    """
    Measurement frequency = 10Hz
    """
    FOLDER = ''
    TT_FORMAT_FOLDER = ''
    RAW_DT = 1./10
    GLOB_STR = '*/trajectories-*.csv'

    @staticmethod
    def load_raw(p):
        df = pd.read_csv(p, header=0, sep=',')
        df = convert_ft2meters(df)
        return df

    @staticmethod
    def raw2tt(df, offset_agent_id=0):
        df = df.rename(columns={
            'Frame_ID': 'frame_id',
            'Vehicle_ID': 'agent_id',
            'Local_X': 'x',
            'Local_Y': 'y',
        }, inplace=False)
        df = df[['frame_id', 'agent_id', 'x', 'y']]
        df = df.astype({'frame_id': int, 'agent_id': int})
        df['agent_id'] += offset_agent_id
        tt.format_dataframe(df, is_raise=False)
        return df

    def get_recordings(self):
        """
        :return:
            name: name of file to later be saved in tt format as name.txt
            track_path: absolute path to raw trajectory file
        """
        glob_str = os.path.join(
            DATASETS_ROOT, self.FOLDER,
            self.GLOB_STR
        )
        for track_path in glob(glob_str):
            name = os.path.basename(track_path)
            name = name.replace('.csv', '')
            yield name, track_path

    def load_as_trajectorytype_format(self, recording_path):
        df = self.load_raw(recording_path)
        df = self.raw2tt(df)
        return df

    @staticmethod
    def make_all_lane_edges(dataset):
        """
        :param dataset:
        :return:
            all_lane_edges: n_lanes+1, 2 | all lane bounds used for both i-80 and us-101
        """
        return make_lane_edges()

    @staticmethod
    def get_lane_edges(all_lane_edges, dataset_id, datafile_id):
        """
        :param all_lane_edges: n_lanes+1, 2 | all lane bounds used for both i-80 and us-101
        :param dataset_id:
        :param datafile_id:
        :return:
        """
        return all_lane_edges


class I80Dataset(NgsimDataset):
    FOLDER = 'ngsim/i-80/vehicle-trajectory-data'
    TT_FORMAT_FOLDER = 'tt_format/10hz/ngsim/i80'


class Us101Dataset(NgsimDataset):
    FOLDER = 'ngsim/us-101/vehicle-trajectory-data'
    TT_FORMAT_FOLDER = 'tt_format/10hz/ngsim/us101'
