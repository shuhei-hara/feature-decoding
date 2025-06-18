'''DNN Feature decoding (corss-validation) - feature prediction script.'''


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time

import bdpy
from bdpy.dataform import load_array, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.ml.crossvalidation import make_cvindex_generator
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np


# Main #######################################################################

def featdec_cv_fastl2lir_predict(
        fmri_data,
        decoder_path,
        output_dir='./feature_decoding_cv',
        rois=None,
        label_key=None,
        cv_key='Run',
        cv_folds=None,
        cv_exclusive=None,
        layers=None,
        feature_index_file=None,
        excluded_labels=[],
        average_sample=True,
        chunk_axis=1,
        analysis_name="feature_prediction",
        subject=None
):
    '''Cross-validation feature decoding.

    Input:

    - fmri_data
    - decoder_path

    Output:

    - output_dir

    Parameters:

    TBA

    Note:

    If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
    Note that Y[0] should be sample dimension.
    '''
    # layers = layers[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data.keys()))
    print('ROIs:            %s' % list(rois.keys()))
    print('Decoders:        %s' % decoder_path)
    print('Layers:          %s' % layers)
    print('CV:              %s' % cv_key)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    # FIXME: support multiple datasets
    data_brain = {
        sbj: bdpy.BData(dat_file[0])
        for sbj, dat_file in fmri_data.items()
    }
    data_brain = {f'sub-{subject}': data_brain[f'sub-{subject}']}

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Distributed computation setup ------------------------------------------
    distcomp_db = os.path.join('./tmp', analysis_name + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    sbj = f'sub-{subject}'
    for layer, roi in product(layers, rois):
        print('--------------------')
        print('Layer:   %s' % layer)
        print('Subject: %s' % sbj)
        print('ROI:     %s' % roi)

        # Cross-validation setup
        if cv_exclusive is not None:
            cv_exclusive_array = data_brain[sbj].select(cv_exclusive)
        else:
            cv_exclusive_array = None

        cv_index = make_cvindex_generator(
            data_brain[sbj].select(cv_key),
            folds=cv_folds,
            exclusive=cv_exclusive_array
        )
        if 'name' in cv_folds[0]:
            cv_labels = ['cv-{}'.format(cv['name']) for cv in cv_folds]
        else:
            cv_labels = ['cv-fold{}'.format(icv + 1) for icv in range(len(cv_folds))]

        for cv_label, (train_index, test_index) in zip(cv_labels, cv_index):
            print('CV fold: {} ({} training; {} test)'.format(cv_label, len(train_index), len(test_index)))

            # Setup
            # -----
            analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + cv_label + '-' + layer
            decoded_feature_dir = os.path.join(output_dir, layer, sbj, roi, cv_label, 'decoded_features')

            # if os.path.exists(decoded_feature_dir):
            #     print('%s is already done. Skipped.' % analysis_id)
            #     continue

            makedir_ifnot(decoded_feature_dir)

            # if not distcomp.lock(analysis_id):
            #     print('%s is already running. Skipped.' % analysis_id)
            #     continue

            # Preparing data
            # --------------
            print('Preparing data')

            start_time = time()

            # Brain data
            brain = data_brain[sbj].select(rois[roi])
            brain_labels = data_brain[sbj].get_label(label_key)

            # Extract test data
            brain = brain[test_index, :]
            brain_labels = np.array(brain_labels)[test_index]

            # Averaging brain data
            if average_sample:
                brain_labels_unique = np.unique(brain_labels)
                if excluded_labels is not None:
                    brain_labels_unique = [lb for lb in brain_labels_unique if lb not in excluded_labels]
                brain = np.vstack([np.mean(brain[(np.array(brain_labels) == lb).flatten(), :], axis=0)
                                   for lb in brain_labels_unique])
            else:
                # Label + sample no.
                brain_labels_unique = ['trial_{:04}-{}'.format(i + 1, lb) for i, lb in enumerate(brain_labels)]

            print('Elapsed time (data preparation): %f' % (time() - start_time))

            # Model directory
            # ---------------
            model_dir = os.path.join(decoder_path, layer, sbj, roi, cv_label, 'model')

            # Preprocessing
            # -------------
            brain_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
            brain_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)
            feat_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
            feat_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

            brain = (brain - brain_mean) / brain_norm

            # Prediction
            # ----------
            print('Prediction')

            start_time = time()

            model = FastL2LiR()

            test = ModelTest(model, brain)
            test.model_format = 'bdmodel'
            test.model_path = model_dir
            test.dtype = np.float32
            test.chunk_axis = chunk_axis

            feat_pred = test.run()

            print('Total elapsed time (prediction): %f' % (time() - start_time))

            # Postprocessing
            # --------------
            feat_pred = feat_pred * feat_norm + feat_mean

            # Save results
            # ------------
            print('Saving results')

            start_time = time()

            # Predicted features
            for i, label in enumerate(brain_labels_unique):
                # Predicted features
                _feat = np.array([feat_pred[i,]])  # To make feat shape 1 x M x N x ...

                # Save file name
                save_file = os.path.join(decoded_feature_dir, '%s.mat' % label)

                # Save
                save_array(save_file, _feat, key='feat', dtype=np.float32, sparse=False)

            print('Saved %s' % decoded_feature_dir)

            print('Elapsed time (saving results): %f' % (time() - start_time))

            distcomp.unlock(analysis_id)

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################


cfg = init_hydra_cfg()
subject = cfg['subject']

analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

decoder_path = cfg["decoded_feature"]["decoder"]["path"]

test_fmri_data = {
    subject["name"]: subject["paths"]
    for subject in cfg["decoded_feature"]["fmri"]["subjects"]
}
rois = {
    roi["name"]: roi["select"]
    for roi in cfg["decoded_feature"]["fmri"]["rois"]
}
label_key = cfg["decoded_feature"]["fmri"]["label_key"]

layers = cfg["decoded_feature"]["features"]["layers"]
feature_index_file = cfg.decoder.features.get("index_file", None)

decoded_feature_dir = cfg["decoded_feature"]["path"]

average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]
excluded_labels = cfg.decoded_feature.fmri.get("exclude_labels", [])

cv_folds = cfg.cv.get("folds", None)
# cv_exclusive = cfg.cv.get("exclusive_key", None)
cv_exclusive = None

featdec_cv_fastl2lir_predict(
    test_fmri_data,
    decoder_path,
    output_dir=decoded_feature_dir,
    rois=rois,
    label_key=label_key,
    cv_key=cfg["cv"]["key"],
    cv_folds=cv_folds,
    cv_exclusive=cv_exclusive,
    layers=layers,
    feature_index_file=feature_index_file,
    excluded_labels=excluded_labels,
    average_sample=average_sample,
    chunk_axis=cfg["decoder"]["parameters"]["chunk_axis"],
    analysis_name=analysis_name,
    subject=subject
)