import os
import loading_utils.tt_dataset as tt
import loading_utils.dataset_tags as dta
from loading_utils.constants import DATASETS_ROOT
import numpy as np


def convert_format2tt_and_freq(dataset_tag, dt, save_folder):
    """
    For each recording in dataset, save a single file in TrajectoryType format
    - including recordings that consist of multiple files
    :return: 
    """
    dataset = dta.DATASET_TAG2INFO[dataset_tag]()
    for name, recording in dataset.get_recordings():
        df = dataset.load_as_trajectorytype_format(recording)
        df = tt.resample_dataset(df, dataset.RAW_DT, dt)
        save_path = os.path.join(save_folder, name + '.txt')
        print('Saving {} to\n{}\n'.format(name, save_path))
        # df.to_csv(save_path, index=False, sep=',', header=True)


def main_convert_format2tt_and_freq():
    dt = 0.1
    dataset_tag = dta.DatasetTag.i80
    save_folder = os.path.join(DATASETS_ROOT, 'tt_format/10hz/ngsim/i80')
    # dataset_tag = dta.DatasetTag.us101
    # save_folder = os.path.join(DATASETS_ROOT, 'tt_format/10hz/ngsim/us101')
    convert_format2tt_and_freq(dataset_tag, dt, save_folder)


def main_downsampler_integer():
    data_folder = os.path.join(DATASETS_ROOT, 'tt_format/10hz/ngsim/i80')
    frame_skip = 5
    save_folder = os.path.join(DATASETS_ROOT, 'tt_format/2hz/ngsim/i80')
    for p in [os.path.join(data_folder, name) for
              name in os.listdir(data_folder) if
              any([ext in name for ext in ['.csv', '.txt']])]:
        df = tt.load_named_df(p)
        frames = df['frame_id'].unique()
        df = df[df['frame_id'].isin(np.arange(frames.min(), frames.max() + 1, frame_skip))]
        name = os.path.basename(p)
        save_path = os.path.join(save_folder, name)
        print('Saving {} to\n{}\n'.format(name, save_path))
        df.to_csv(save_path, index=False, sep=',')


if __name__ == '__main__':
    # main_convert_format2tt_and_freq()
    # main_downsampler_integer()
    pass







