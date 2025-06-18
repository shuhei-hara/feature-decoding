'''Feature decoding (corss-validation) evaluation.'''


from typing import Dict, List, Optional

from itertools import product
import os
import re
import bdpy.stats

from bdpy.dataform import Features, DecodedFeatures, SQLite3KeyValueStore
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from bdpy.pipeline.config import init_hydra_cfg
import hdf5storage
import numpy as np
import yaml
import scipy.stats
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


# Main #######################################################################

class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass

def random_derangement(n):
    while True:
        perm = np.random.permutation(n)
        if not np.any(perm == np.arange(n)): 
            return perm


def featdec_cv_eval(
        decoded_feature_path: str,
        true_feature_path: str,
        subjects: Optional[List[str]] = None,
        output_file_pooled: str = './evaluation.db',
        output_file_fold: str = './evaluation_fold.db',
        rois: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        feature_index_file: Optional[str] = None,
        feature_decoder_path: Optional[str] = None,
        average_sample: bool = True,
):
    '''Evaluation of feature decoding.

    Input:

    - decoded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

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

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_path)

    cv_folds = decoded_features.folds

    # Metrics ################################################################
    metrics = ['unit_correlation', 'image_correlation', 'different_image_correlation', 'true_feature_correlation', 'r2_unit', "r2_image",'pattern_same','pattern_diff']
    pooled_operation = {
        "unit_correlation": "concat",
        "image_correlation": "concat",
        "different_image_correlation": "concat",
        "true_feature_correlation": "concat",
        "r2_unit": "concat",
        "r2_image": "concat",
        "pattern_same": "concat",
        "pattern_diff": "concat",
    }

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file_fold):
        print('Loading {}'.format(output_file_fold))
        results_db = ResultsStore(output_file_fold)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "fold", "metric"]
        results_db = ResultsStore(output_file_fold, keys=keys)

    true_labels = features_test.labels

    for layer in layers:
        print('Layer: {}'.format(layer))
        true_y = features_test.get_features(layer=layer)
        # true_y = scipy.stats.zscore(true_y,axis=0)

        for roi, fold in list(product(rois, cv_folds)):
            print('Subject: {} - ROI: {} - Fold: {}'.format(subject, roi, fold))

            # Check if the evaluation is already done
            exists = True
            for metric in metrics:
                exists = exists and results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric=metric)
            # if exists:
            #     print('Already done. Skipped.')
            #     continue

            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi, fold=fold)
            pred_labels = np.array(decoded_features.selected_label)

            if not average_sample:
                pred_labels = [re.match('trial_\d*-(.*)', x).group(1) for x in pred_labels]

            # Use predicted data that has a label included in true_labels
            selector = np.array([True if p in true_labels else False for p in pred_labels])
            pred_y = pred_y[selector, :]
            pred_labels = pred_labels[selector]
            
            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            # Load Y mean and SD
            # Proposed by Ken Shirakawa. See https://github.com/KamitaniLab/brain-decoding-cookbook/issues/13.
            norm_param_dir = os.path.join(
                feature_decoder_path,
                layer, subject, roi, fold,
                'model'
            )

            train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
            train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

            deranged_indices = random_derangement(true_y_sorted.shape[0])
            true_y_deranged = true_y_sorted[deranged_indices]

            # Evaluation ---------------------------
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='unit_correlation', value=np.array([]))
            unit_correlation = np.array([pearsonr(pred_y[:, i], true_y_sorted[:, i])[0] for i in range(pred_y.shape[1])])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='unit_correlation', value=unit_correlation)
            print('Mean decoding accuracy (unit_correlation):' + str(np.mean(unit_correlation)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='image_correlation', value=np.array([]))
            image_correlation = np.array([pearsonr(pred_y[i, :], true_y_sorted[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='image_correlation', value=image_correlation)
            print('Mean decoding accuracy (image_correlation):' + str(np.mean(image_correlation)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='different_image_correlation', value=np.array([]))
            different_image_correlation = np.array([pearsonr(pred_y[i, :], true_y_deranged[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='different_image_correlation', value=different_image_correlation)
            print('Mean decoding accuracy (different_image_correlation):' + str(np.mean(different_image_correlation)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='true_feature_correlation', value=np.array([]))
            true_feature_correlation = np.array([pearsonr(true_y_sorted[i, :], true_y_deranged[i, :])[0] for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='true_feature_correlation', value=true_feature_correlation)
            print('Mean decoding accuracy (true_feature_correlation):' + str(np.mean(true_feature_correlation)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='r2_unit', value=np.array([]))
            r2_unit = np.array([r2_score(true_y_sorted[:, i],pred_y[:, i]) for i in range(pred_y.shape[1])])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='r2_unit', value=r2_unit)
            print('Mean decoding accuracy (r2_unit):' + str(np.mean(r2_unit)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='r2_image', value=np.array([]))
            r2_image = np.array([r2_score(true_y_sorted[i, :],pred_y[i, :]) for i in range(pred_y.shape[0])])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='r2_image', value=r2_image)
            print('Mean decoding accuracy (r2_image):' + str(np.mean(r2_image)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_same', value=np.array([]))
            pattern_same = np.array([pattern_correlation(true_y_sorted,pred_y,mean=train_y_mean, std=train_y_std)])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_same', value=pattern_same)
            print('Mean decoding accuracy (pattern_same):' + str(np.mean(pattern_same)))

            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_diff', value=np.array([]))
            pattern_diff = np.array([pattern_correlation(true_y_deranged,pred_y,mean=train_y_mean, std=train_y_std)])
            results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_diff', value=pattern_diff)
            print('Mean decoding accuracy (pattern_diff):' + str(np.mean(pattern_diff)))



    print('All fold done')

    # Pooled accuracy
    if os.path.exists(output_file_pooled):
        print('Loading {}'.format(output_file_pooled))
        pooled_db = ResultsStore(output_file_pooled)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "metric"]
        pooled_db = ResultsStore(output_file_pooled, keys=keys)

    done_all = True  # Flag indicating that all conditions have been pooled
    for layer, roi, metric in product(layers, rois, metrics):
        # Check if pooling is done
        # if pooled_db.exists(layer=layer, subject=subject, roi=roi, metric=metric):
        #     continue
        pooled_db.set(layer=layer, subject=subject, roi=roi, metric=metric, value=np.array([]))

        # Check if all folds are complete
        done = True
        for fold in cv_folds:
            if not results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric=metric):
                done = False
                break

        # When all folds are complete, pool the results.
        if done:
            acc = []
            for fold in cv_folds:
                acc.append(results_db.get(layer=layer, subject=subject, roi=roi,
                                          fold=fold, metric=metric))
            if pooled_operation[metric] == "mean":
                acc = np.nanmean(acc, axis=0)
            elif pooled_operation[metric] == "concat":
                acc = np.hstack(acc)
            pooled_db.set(layer=layer, subject=subject, roi=roi,
                          metric=metric, value=acc)

        # If there are any unfinished conditions,
        # do not pool the results and set the done_all flag to False.
        else:
            pooled_db.delete(layer=layer, subject=subject, roi=roi, metric=metric)
            done_all = False
            continue

    if done_all:
        print('All pooling done.')
    else:
        print("Some pooling has not finished.")

    return output_file_pooled, output_file_fold


# Entry point ################################################################



cfg = init_hydra_cfg()
subject = cfg['subject']

decoded_feature_path = cfg["decoded_feature"]["path"]
gt_feature_path      = cfg["decoded_feature"]["features"]["paths"][0]  # FIXME

feature_decoder_path = cfg["decoded_feature"]["decoder"]["path"]
subjects = [s["name"] for s in cfg["decoded_feature"]["fmri"]["subjects"]]
rois = [r["name"] for r in cfg["decoded_feature"]["fmri"]["rois"]]
layers = cfg["decoded_feature"]["features"]["layers"]

feature_index_file = cfg.decoder.features.get("index_file", None)
average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]

featdec_cv_eval(
    decoded_feature_path,
    gt_feature_path,
    subjects=subject,
    output_file_pooled=os.path.join(decoded_feature_path, f'{subject}_evaluation.db'),
    output_file_fold=os.path.join(decoded_feature_path, f'{subject}_evaluation_fold.db'),
    rois=rois,
    layers=layers,
    feature_index_file=feature_index_file,
    feature_decoder_path=feature_decoder_path,
    average_sample=average_sample,
)