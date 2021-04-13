import os
from loading_utils import tt_dataset as tt


def process_data2datasets(
        data_dir, n_obs, n_pred, sep=',',
        dataset_id2file_id_list=(), is_fill_nan=False, valid_ids_kwargs=None,
        df_loader_fcn=None, dataset_type=tt.TrajectoryTypeDataset):
    """
    Creates dataset based on layout of data_dir
    1) data_dir contains no txt/csv files
    - process each folder into TrajectoryTypeDataset
    - combine these into single one, retaining dataset_id
    2) data_dir contains txt/csv
    - process these into single TrajectoryTypeDataset
    :param data_dir: absolute path to directory of data dir, or directly to txt/csv
    - attempt to match data files with images in the data dir
    :param n_obs:
    :param n_pred:
    :param sep: separator for reading data
    :param dataset_id2file_id_list: for selecting which dataset/dataset's files to load
    :param is_fill_nan:
    :param valid_ids_kwargs:
    :param df_loader_fcn:
    :return:
        dataset: TrajectoryTypeDataset
    """
    is_contains_csv = any([any([ext in name for ext in ['.txt', 'csv']]) for name in os.listdir(data_dir)])
    if is_contains_csv:
        dataset_file_ids = dataset_id2file_id_list[0] if 0 in dataset_id2file_id_list else None
        dataset = dataset_type(
            data_dir, n_obs, n_pred, sep=sep, dataset_id=0,
            dataset_file_ids=dataset_file_ids, is_fill_nan=is_fill_nan,
            valid_ids_kwargs=valid_ids_kwargs, df_loader_fcn=df_loader_fcn,
        )
        return dataset

    dataset_list = []
    for i, sub_data_dir in enumerate(
            [os.path.join(data_dir, name) for
             name in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, name))]):
        dataset_file_ids = None
        if dataset_id2file_id_list and i in dataset_id2file_id_list:
            dataset_file_ids = dataset_id2file_id_list[i]
        dataset = dataset_type(
            sub_data_dir, n_obs, n_pred, sep=sep, dataset_id=i,
            dataset_file_ids=dataset_file_ids, is_fill_nan=is_fill_nan,
            valid_ids_kwargs=valid_ids_kwargs, df_loader_fcn=df_loader_fcn,
        )
        dataset_list.append(dataset)
    dataset = tt.concat_datasets(dataset_list)
    return dataset
