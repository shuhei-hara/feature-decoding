'''Feature decoding evaluation.'''
import os
import re

from bdpy.dataform import Features, DecodedFeatures, SQLite3KeyValueStore
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from bdpy.pipeline.config import init_hydra_cfg
import hdf5storage
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# Main #######################################################################

class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass

def random_derangement(n):
    while True:
        perm = np.random.permutation(n)
        if not np.any(perm == np.arange(n)):  # どの位置にも元と同じ行がないか確認
            return perm


def featdec_eval(
        decoded_feature_path,
        true_feature_path,
        output_file='./evaluation.db',
        subjects=None,
        rois=None,
        layers=None,
        feature_index_file=None,
        feature_decoder_path=None,
        average_sample=True,
):
    '''Evaluation of feature decoding.

    Input:

    - decoded_feature_path
    - true_feature_path

    Output:

    - output_file

    Parameters:

    TBA
    '''

    np.random.seed(123)

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_path))
    print('')
    print('True features (Test): {}'.format(true_feature_path))
    print('')
    print('Layers: {}'.format(layers))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################
    subject = f'sub-{subjects}'
    # True features
    if feature_index_file is not None:
        features_test = Features(true_feature_path, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_path)

    features_test_original = Features('/flash/DoyaU/shuhei/bdata_decoding/features_test_original_norm', feature_index=feature_index_file)

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_path)

    # Metrics ################################################################
    metrics = ['unit_correlatio_o', 'unit_correlatio_s', 'same_image_correlation', 'different_image_correlation', 'true_feature_correlation', 'r_o', 'r_s',
                'pattern_same_blur', 'pattern_diff_blur', 'pattern_same_original', 'pattern_diff_original']
    pooled_operation = {
        "unit_correlation_o": "concat",
        "unit_correlation_s": "concat",
        "same_image_correlation": "concat",
        "different_image_correlation": "concat",
        "true_feature_correlation": "concat",
        "r_o": "concat",
        "r_s": "concat",
        'pattern_same_blur': "concat",
        'pattern_diff_blur': "concat",
        'pattern_same_original': "concat",
        'pattern_diff_original': "concat",
    }
    # Evaluating decoding performances #######################################

    # if os.path.exists(output_file):
    #     print('Loading {}'.format(output_file))
    #     results_db = ResultsStore(output_file)
    # else:
    print('Creating new evaluation result store')
    keys = ["layer", "subject", "roi", "metric"]
    results_db = ResultsStore(output_file, keys=keys)

    for layer in layers[::-1]:
        print('Layer: {}'.format(layer))

        true_y = features_test.get(layer=layer)
        true_labels = features_test.labels

        ### TEST Non-blurred
        true_original_y = features_test_original.get(layer=layer)


        for roi in list(rois):
            print('Subject: {} - ROI: {}'.format(subject, roi))

            # Check if the evaluation is already done
            exists = True
            for metric in metrics:
                exists = exists and results_db.exists(layer=layer, subject=subject, roi=roi, metric=metric)
            # if exists:
            #     print('Already done. Skipped.')
            #     continue

            # Load decoded features
            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi)
            pred_labels = np.array(decoded_features.selected_label)

            if not average_sample:
                pred_labels = [re.match('sample\d*-(.*)', x).group(1) for x in pred_labels]

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y


            # Load Y mean and SD
            # Proposed by Ken Shirakawa. See https://github.com/KamitaniLab/brain-decoding-cookbook/issues/13.
            norm_param_dir = os.path.join(
                feature_decoder_path,
                layer, subject, roi,
                'model'
            )

            train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
            train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

            deranged_indices = random_derangement(true_y_sorted.shape[0])
            true_y_deranged_blur = true_y_sorted[deranged_indices]
            true_y_deranged_original = true_original_y[deranged_indices]

            # Evaluation ---------------------------

            # Unit correlation (check whether original or stimuli features)
            results_db.set(layer=layer, subject=subject, roi=roi, metric='unit_correlation_o', value=np.array([]))
            unit_correlation_o = np.array([pearsonr(pred_y[:, i], true_original_y[:, i])[0] for i in range(pred_y.shape[1])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='unit_correlation_o', value=unit_correlation_o)
            print('Mean decoding accuracy (unit_correlation_o):' + str(np.mean(unit_correlation_o)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='unit_correlation_s', value=np.array([]))
            unit_correlation_s = np.array([pearsonr(pred_y[:, i], true_y_sorted[:, i])[0] for i in range(pred_y.shape[1])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='unit_correlation_s', value=unit_correlation_s)
            print('Mean decoding accuracy (unit_correlation_s):' + str(np.mean(unit_correlation_s)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='same_image_correlation', value=np.array([]))
            same_image_correlation = np.array([pearsonr(pred_y[i, :], true_original_y[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='same_image_correlation', value=same_image_correlation)
            print('Mean decoding accuracy (same_image_correlation):' + str(np.mean(same_image_correlation)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='different_image_correlation', value=np.array([]))
            # different_image_correlation = np.array([
            #     np.mean([
            #         pearsonr(pred_y[i], true_original_y[j])[0]
            #         for j in range(100) if j != i
            #     ])
            #     for i in range(100)
            # ])
            different_image_correlation = np.array([pearsonr(pred_y[i, :], true_y_deranged_original[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='different_image_correlation', value=different_image_correlation)
            print('Mean decoding accuracy (different_image_correlation):' + str(np.mean(different_image_correlation)))
            

            results_db.set(layer=layer, subject=subject, roi=roi, metric='true_feature_correlation', value=np.array([]))
            # true_feature_correlation = np.array([
            #     np.mean([
            #         pearsonr(true_y_sorted[i], true_original_y[j])[0]
            #         for j in range(100) if j != i
            #     ])
            #     for i in range(100)
            # ])
            true_feature_correlation = np.array([pearsonr(true_y_sorted[i, :], true_y_deranged_blur[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='true_feature_correlation', value=true_feature_correlation)
            print('Mean decoding accuracy (true_feature_correlation):' + str(np.mean(true_feature_correlation)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='r_o', value=np.array([]))
            r_o = np.array([pearsonr(pred_y[i, :], true_original_y[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='r_o', value=r_o)
            print('Mean decoding accuracy (r_o):' + str(np.mean(r_o)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='r_s', value=np.array([]))
            r_s = np.array([pearsonr(pred_y[i, :], true_y_sorted[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='r_s', value=r_s)
            print('Mean decoding accuracy (r_s):' + str(np.mean(r_s)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_same_blur', value=np.array([]))
            pattern_same_blur = np.array([pattern_correlation(true_y_sorted,pred_y,mean=train_y_mean, std=train_y_std)])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_same_blur', value=pattern_same_blur)
            print('Mean decoding accuracy (pattern_same_blur):' + str(np.mean(pattern_same_blur)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_diff_blur', value=np.array([]))
            pattern_diff_blur = np.array([pattern_correlation(true_y_deranged_blur,pred_y,mean=train_y_mean, std=train_y_std)])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_diff_blur', value=pattern_diff_blur)
            print('Mean decoding accuracy (pattern_diff_blur):' + str(np.mean(pattern_diff_blur)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_same_original', value=np.array([]))
            pattern_same_original = np.array([pattern_correlation(true_original_y,pred_y,mean=train_y_mean, std=train_y_std)])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_same_original', value=pattern_same_original)
            print('Mean decoding accuracy (pattern_same_original):' + str(np.mean(pattern_same_original)))

            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_diff_original', value=np.array([]))
            pattern_diff_original = np.array([pattern_correlation(true_y_deranged_original,pred_y,mean=train_y_mean, std=train_y_std)])
            results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_diff_original', value=pattern_diff_original)
            print('Mean decoding accuracy (pattern_diff_original):' + str(np.mean(pattern_diff_original)))


    print('All done')

    return output_file


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()
    subject = cfg['subject']

    decoded_feature_path = cfg["decoded_feature"]["path"]
    gt_feature_path      = cfg["decoded_feature"]["features"]["paths"][0]

    feature_decoder_path = cfg["decoded_feature"]["decoder"]["path"]
    # subjects = [s["name"] for s in cfg["decoded_feature"]["fmri"]["subjects"]]
    rois = [r["name"] for r in cfg["decoded_feature"]["fmri"]["rois"]]
    layers = cfg["decoded_feature"]["features"]["layers"]

    feature_index_file = cfg.decoder.features.get("index_file", None)
    average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]

    featdec_eval(
        decoded_feature_path,
        gt_feature_path,
        output_file=os.path.join(path_to_outut, f'{subject}_evaluation.db'),
        subjects=subject,
        rois=rois,
        layers=layers,
        feature_index_file=feature_index_file,
        feature_decoder_path=feature_decoder_path,
        average_sample=average_sample
    )