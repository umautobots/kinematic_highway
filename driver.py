import numpy as np
import os
import loading_utils.dataset_tags as dta
from loading_utils.constants import DATASETS_ROOT
import loading_utils.multi_dataset as mtt
import loading_utils.tt_dataset as tt
import loading_utils.occlusion_dataset as ott
import evaluation.results as re
import evaluation.metrics_2d as metrics


def main_driver():
    n_obs = 30
    n_pred = 50
    dataset_tag = dta.DatasetTag.us101
    dataset_info = dta.DATASET_TAG2INFO[dataset_tag]
    data_dir = os.path.join(DATASETS_ROOT, dataset_info.TT_FORMAT_FOLDER)
    dataset = mtt.process_data2datasets(
        data_dir,
        n_obs, n_pred,
        dataset_type=tt.TrajectoryTypeDataset,
        # dataset_type=ott.OccludedTrajectoryTypeDataset,
    )
    all_lane_edges = dataset_info.make_all_lane_edges(dataset)
    metric_fcn_list = [
        ['q[d_t]', metrics.get_point_quantile_expected_dist_by_time_fcns(
            quantile=0.2, select_inds=np.arange(9, n_pred, 10))],
        ['E[d_t]', metrics.get_expected_dist_by_time_fcns(
           select_inds=np.arange(9, n_pred, 10))],
        ['rmse[t]', metrics.get_rmse_by_time_fcns(
            select_inds=np.arange(9, n_pred, 10))],
        ['time', metrics.get_timing_fcns()],
    ]
    is_display = False
    if is_display:
        import display.predictions_2d as di
    np.random.seed(1)

    # setup for prediction methods
    from baselines import cv_kalman
    from kinematic_model import kin_model as test_model9
    method_info = [
        (
            'CV_KF',
            cv_kalman.predict,
            dict(n_steps=n_pred),
        ),
        (
            'Proposed',
            lambda p, dataset_id, datafile_id: test_model9.predict_all(
                p, dataset_info.get_lane_edges(all_lane_edges, dataset_id, datafile_id), 100, n_pred),
            dict(),
        ),
    ]
    prediction_methods = [re.TrajectoryResults(*info) for info in method_info]
    running_eval = re.RunningEvaluation(metric_fcn_list, prediction_methods)

    # predict
    print('{} sets in datasets'.format(len(dataset)))
    for i in range(len(dataset)):
        if (i % 10 == 0) and i > 0:
            print('    processed {}'.format(i))
        if (i % 100 == 0) and i > 0:
            print('\nCurrent metrics\n')
            running_eval.reduce()
            print('Predictions on {}'.format(i))
        vic_xy, dataset_id, datafile_id = dataset.get_df(i)
        vic_xy_obs = vic_xy[:n_obs, ...]

        for predict_fcn in prediction_methods:
            predict_fcn.predict(vic_xy_obs, dataset_id, datafile_id)

        xy_true = vic_xy[n_obs:, ...]
        running_eval.evaluate(prediction_methods, xy_true)

        if is_display:
            info = dataset.get_frame_info(i)
            datafile_path = info[0]
            frame = info[1]
            ego_ind = info[3] if len(info) == 4 else np.inf
            for j in range(xy_true.shape[1]):
                di.display_predictions(
                    vic_xy_obs, j, xy_true,
                    prediction_methods,
                    data_title=di.format_example_title(i, j),
                    datafile_path=datafile_path,
                    ego_ind=ego_ind,
                )
        # save mem
        for prediction_method in prediction_methods:
            prediction_method.clear()
    print('\n')
    running_eval.reduce(decimals=4)


if __name__ == '__main__':
    main_driver()
