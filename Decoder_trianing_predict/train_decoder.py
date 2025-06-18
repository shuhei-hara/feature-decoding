'''DNN Feature decoding (corss-validation) - decoders training script.'''


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time
import warnings
import pickle

import bdpy
from bdpy.bdata.utils import select_data_multi_bdatas, get_labels_multi_bdatas
from bdpy.dataform import Features, save_array
from bdpy.dataform.utils import get_multi_features
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.ml.crossvalidation import make_cvindex_generator
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml
import scipy.stats

# do : module load python/3.11.4

# Main #######################################################################

def featdec_fastl2lir_train(
        fmri_data: Dict[str, List[str]],
        features_paths: List[str],
        output_dir: str = './feature_decoding_cv',
        rois: Optional[Dict[str, str]] = None,
        num_voxel: Optional[Dict[str, int]] = None,
        label_key: Optional[str] = None,
        layers: Optional[List[str]] = None,
        feature_index_file: Optional[str] = None,
        alpha: int = 100,
        chunk_axis: int = 1,
        analysis_name: str = "feature_decoder_training",
        subject=None
):
    '''Cross-validation feature decoder training.

    Input:

    - fmri_data
    - features_paths

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
    # print('Subjects:        %s' % list(fmri_data.keys()))
    print('Subjects:        %s' % subject)
    print('ROIs:            %s' % list(rois.keys()))
    print('Target features: %s' % features_paths)
    print('Layers:          %s' % layers)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: [bdpy.BData(f) for f in data_files] for sbj, data_files in fmri_data.items()}
    data_brain = {f'sub-{subject}': data_brain[f'sub-{subject}']}

    if feature_index_file is not None:
        data_features = [Features(f, feature_index=os.path.join(f, feature_index_file)) for f in features_paths]
    else:
        data_features = [Features(f) for f in features_paths]

    print(data_features[0])

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    sbj = f'sub-{subject}'
    # for layer, sbj, roi in np.random.permutation(list(product(layers, fmri_data.keys(), rois.keys()))):
    for layer, roi in product(layers, rois.keys()):

        print('--------------------')
        print('Layer:      %s' % layer)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])

        # Setup
        # -----
        analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + layer
        model_dir = os.path.join(output_dir, layer, sbj, roi, 'model')
        makedir_ifnot(model_dir)

        # Check whether the analysis has been done or not.
        # info_file = os.path.join(model_dir, 'info.yaml')
        # if os.path.exists(info_file):
        #     with open(info_file, 'r') as f:
        #         info = yaml.safe_load(f)
        #     while info is None:
        #         warnings.warn('Failed to load info from %s. Retrying...'
        #                         % info_file)
        #         with open(info_file, 'r') as f:
        #             info = yaml.safe_load(f)
        #     if '_status' in info and 'computation_status' in info['_status']:
        #         if info['_status']['computation_status'] == 'done':
        #             print('%s is already done and skipped' % analysis_id)
        #             continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        brain = select_data_multi_bdatas(data_brain[sbj], rois[roi])
        brain_labels = get_labels_multi_bdatas(data_brain[sbj], label_key)

        feat_labels = brain_labels
        feat = get_multi_features(data_features, layer, labels=feat_labels)        

        feat_train = feat

        # print('feat_train: ', feat_train.shape)
        # return

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Calculate normalization parameters
        # ----------------------------------

        # Normalize X (fMRI data)
        brain_mean = np.mean(brain, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
        brain_norm = np.std(brain, axis=0, ddof=1)[np.newaxis, :]

        # Normalize Y (DNN features)
        feat_mean = np.mean(feat_train, axis=0)[np.newaxis, :]
        feat_norm = np.std(feat_train, axis=0, ddof=1)[np.newaxis, :]

        # Save normalization parameters
        # -----------------------------
        print('Saving normalization parameters.')
        norm_param = {
            'x_mean': brain_mean, 'y_mean': feat_mean,
            'x_norm': brain_norm, 'y_norm': feat_norm
        }
        save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
        for sv in save_targets:
            save_file = os.path.join(model_dir, sv + '.mat')
            if not os.path.exists(save_file):
                try:
                    save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                    print('Saved %s' % save_file)
                except Exception:
                    warnings.warn('Failed to save %s. Possibly double running.' % save_file)

        # Preparing learning
        # ------------------
        model = FastL2LiR()
        model_param = {
            'alpha':  alpha,
            'n_feat': num_voxel[roi],
            'dtype':  np.float32
        }

        # Distributed computation setup
        # -----------------------------
        makedir_ifnot('./tmp')
        distcomp_db = os.path.join('./tmp', analysis_name + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

        # Model training
        # --------------
        print('Model training')
        start_time = time()

        train = ModelTraining(model, brain, feat_train)
        train.id = analysis_id
        train.model_parameters = model_param

        train.X_normalize = {'mean': brain_mean,
                                'std': brain_norm}
        train.Y_normalize = {'mean': feat_mean,
                                'std': feat_norm}
        # train.Y_sort = {'index': feat_index}

        train.dtype = np.float32
        train.chunk_axis = chunk_axis
        train.save_format = 'bdmodel'
        train.save_path = model_dir
        train.distcomp = distcomp

        train.run()

        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_name)

    return output_dir




cfg = init_hydra_cfg()
subject = cfg['subject']

analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]


training_fmri = {
    subject["name"]: subject["paths"]
    for subject in cfg["decoder"]["fmri"]["subjects"]
}
rois = {
    roi["name"]: roi["select"]
    for roi in cfg["decoder"]["fmri"]["rois"]
}
num_voxel = {
    roi["name"]: roi["num"]
    for roi in cfg["decoder"]["fmri"]["rois"]
}
label_key = cfg["decoder"]["fmri"]["label_key"]

training_features = cfg["decoder"]["features"]["paths"]
layers = cfg["decoder"]["features"]["layers"]
feature_index_file = cfg.decoder.features.get("index_file", None)


decoder_dir = cfg["decoder"]["path"]

featdec_fastl2lir_train(
    training_fmri,
    training_features,
    output_dir=decoder_dir,
    rois=rois,
    num_voxel=num_voxel,
    label_key=label_key,
    layers=layers,
    feature_index_file=feature_index_file,
    alpha=cfg["decoder"]["parameters"]["alpha"],
    chunk_axis=cfg["decoder"]["parameters"]["chunk_axis"],
    analysis_name=analysis_name,
    subject=subject
)