import os
import re
import glob
from collections import OrderedDict
import gc

import numpy as np

import bdpy
from bdpy.mri import create_bdata_fmriprep, add_rois, merge_rois, add_hcp_rois, add_hcp_visual_cortex
from bdpy.preproc import average_sample, reduce_outlier, regressout, shift_sample


# do : module load python/3.11.4

import argparse

# Parse the subject argument from the command line
parser = argparse.ArgumentParser(description="Preprocessing for each subject")
parser.add_argument('--subject', type=str, help='Subject ID')
args = parser.parse_args()

subject = args.subject


# Settings ###################################################################

# Here you can specify BIDS directory which contains fmriprep output
# in `derivative` directory.
# If it's left as 'None', the script automatically set it to '../bids'.

bids_dir = ''

# The output bdata will be saved here.

output_dir = ''

# Output data type can be:
# - 'volume_native' (volume fMRI data in the individual brain)
# - 'volume_native_resampled' (volume fMRI data in the individual brain resampled into template space)
# - 'volume_standard' (volume fMRI data in the MNI standard brain)
# - 'surface_native' (surface fMRI data in the individual brain)
# - 'surface_standard' (surface fMRI data in the fsaverage standard brain)
# - 'surface_standard_41k' (surface fMRI data in the fsaverage6 standard brain)
# - 'surface_standard_10k' (surface fMRI data in the fsaverage5 standard brain)

# CAUTION: 'volume_standard' is not supported yet.

output_data_type_list = [
    # 'volume_native',
    'volume_standard',
    # 'surface_native',
    # 'surface_standard',
    # 'surface_standard_41k',
    # 'surface_standard_10k'
]

# Exclude sessions and runs --------------------------------------------------

# Here you can specify session(s) or run(s) that are excluded from the
# resulting BData. Excluded session(s) and run(s) are specified as a
# dictionary as below. Leave the dictionary empty to include all sessions and
# runs.
#
# Examples:
#
#  exclude = {'session': [2, 3]}                          # Exclude session 2 and 3
#  exclude = {'run': [5, 6, 7, 8]}                        # Exclude run 5-8 from all sessions
#  exclude = {'session/run': [None, None, [1, 2, 3, 4]]}  # Exclude 1-4 runs in the 3rd session
#

# exclude = {'run': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
# exclude = {'run': [16, 17, 18, 19, 20]}
exclude = {} 


# ROIs -----------------------------------------------------------------------

# Here you need to specify ROI mask or label files assigned to the resulting
# BData. You should select appropriate ROIs for the output data type.
#
# - 'volume_native' -> Volume ROI masks resliced (resampled) to the fmriprep
#                      native (individual) volume space.
# - 'volume_native_resampled' -> Volume ROI masks resliced (resampled) to the
#                                user-defined template space.
# - 'volume_standard' -> Volume ROI masks resliced (resampled) to the fmriprep
#                        template volume space (MNI152 NL 2009c Asymmetric).
# - 'surface_native' -> Freesurfer labels or annotations on the native
#                       (individual) surface.
# - 'surface_standard' -> Freesurfer labels or annotations on the standard
#                         surface (e.g., fsaverage).
#

# roi_mask_native_dir = '<Path to the ROI mask directory (the mask should be resampled by the functional space)>'
roi_mask_native_dir = ''
roi_mask_standard_dir = ''
roi_mask_native_resampled_dir = ''
roi_label_surface_dir = ''

rois_files = {
    # Native (individual) spaces
    'volume_native':           [os.path.join(roi_mask_native_dir, 'freesurfer_*', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_dir, 'pickatlas', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_dir, 'hcp180', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_dir, 'retinotopy_*', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_dir, 'localizer_*', 'r_*.nii.gz')],

    'volume_standard':           [os.path.join(roi_mask_standard_dir, '*.nii.gz')],

    'volume_native_resampled': [os.path.join(roi_mask_native_resampled_dir, 'freesurfer_*', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_resampled_dir, 'pickatlas', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_resampled_dir, 'hcp180', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_resampled_dir, 'retinotopy_*', 'r_*.nii.gz'),
                                os.path.join(roi_mask_native_resampled_dir, 'localizer_*', 'r_*.nii.gz')],

    'surface_native':          [(os.path.join(roi_label_surface_dir, 'freesurfer_*', 'lh.*.label'),
                                 os.path.join(roi_label_surface_dir, 'freesurfer_*', 'rh.*.label')),
                                (os.path.join(roi_label_surface_dir, 'hcp180', 'lh.*.label'),
                                 os.path.join(roi_label_surface_dir, 'hcp180', 'rh.*.label')),
                                (os.path.join(roi_label_surface_dir, 'retinotopy_*', 'lh.*.label'),
                                 os.path.join(roi_label_surface_dir, 'retinotopy_*', 'rh.*.label')),
                                (os.path.join(roi_label_surface_dir, 'localizer_*', 'lh.*.label'),
                                 os.path.join(roi_label_surface_dir, 'localizer_*', 'rh.*.label'))],

    # Standard spaces
    # You usually do not need to modify the following lines.
    # 'surface_standard':        [(os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', 'lh.aparc.a2009s.annot'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', 'rh.aparc.a2009s.annot')),
    #                             (os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', 'lh.aparc.annot'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', 'rh.aparc.annot')),
    #                             (os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', 'hcp180/lh.*.label'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage', 'label', 'hcp180/rh.*.label'))],
    # 'surface_standard_41k':    [(os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'label', 'lh.aparc.a2009s.annot'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'label', 'rh.aparc.a2009s.annot')),
    #                             (os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'label', 'lh.aparc.annot'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'label', 'rh.aparc.annot')),
    #                             (os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'label', 'hcp180/lh.*.label'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'label', 'hcp180/rh.*.label'))],
    # 'surface_standard_10k':    [(os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage5', 'label', 'lh.aparc.a2009s.annot'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage5', 'label', 'rh.aparc.a2009s.annot')),
    #                             (os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage5', 'label', 'lh.aparc.annot'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage5', 'label', 'rh.aparc.annot')),
    #                             (os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage5', 'label', 'hcp180/lh.*.label'),
    #                              os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage5', 'label', 'hcp180/rh.*.label'))],
}

# Merging ROIs ---------------------------------------------------------------
# roi_merge = {'volume_standard': OrderedDict()}
# # Defining `roi_merge` as Ordereddict to resolve dependency in ROI definition.

# roi_merge['volume_standard']['ROI_V1']  = 'V1.*'
# roi_merge['volume_standard']['ROI_V2']  = 'V2.*'
# roi_merge['volume_standard']['ROI_V3']  = 'V3.*'
# roi_merge['volume_standard']['ROI_V4'] = 'V4.*'
# roi_merge['volume_standard']['ROI_LOC'] = 'LOC.*'
# roi_merge['volume_standard']['ROI_FFA'] = 'FFA.*'
# roi_merge['volume_standard']['ROI_PPA'] = 'PPA.*'
# roi_merge['volume_standard']['ROI_LVC'] = 'ROI_V1 + ROI_V2 + ROI_V3'
# roi_merge['volume_standard']['ROI_HVC'] = 'ROI_LOC + ROI_FFA + ROI_PPA'
# roi_merge['volume_standard']['ROI_VC']  = 'ROI_LVC + ROI_V4 + ROI_HVC'

# Don't modify `roi_prefix_mapper` unless you're sure on what you are doing.
roi_prefix_mapper = {'aparc.annot': 'freesurfer_dk',
                     'aparc.a2009s': 'freesurfer_destrieux'}

# Label mapping table --------------------------------------------------------

# Since BData can only contain float numbers in the dataset, you need to
# define label-to-number mapping table to convert string labels (e.g.,
# 'stimulus_name') into numbers if the task event files in the BIDS dataset
# include string data.

# Currently, the label-to-number table can be defined as:

#   - Python dict
#   - File (csv or tsv)

# `label_mapper` defines the table for each column in the task event file.
# The key of `label_mapper` represents a column in the task event file.
# The each value in `label_mapper` should be either (1) a Python dictionary
# of label-to-number mapping table or (2) path to a file containing a list
# of stimulus labels and numbers.

label_mapper = {'stimulus_name' : ''}



# Other data settings --------------------------------------------------------

# Whether split data by task labels or not (i.e., create a single BData file
# for each subject/task). This should be True unless you want to merge data
# of different tasks into one file.

split_task_label = True

# Preprocessing parameters ---------------------------------------------------

shift_size = 2  # Num of volumes (not seconds) to be shifted

# Main #######################################################################

if bids_dir is None:
    bids_dir = '../bids'

bids_dir = os.path.abspath(bids_dir)

data_id = os.path.basename(os.path.dirname(bids_dir))

for output_data_type in output_data_type_list:

    if output_data_type in ['volume_native', 'volume_native_resampled', 'volume_standard']:
        brain_data_key = 'VoxelData'
    elif output_data_type in ['surface_native', 'surface_standard', 'surface_standard_41k', 'surface_standard_10k']:
        brain_data_key = 'VertexData'
    else:
        raise ValueError('Unknown output data type: %s' % output_data_type)

    if output_data_type in ['volume_native_resampled']:
        output_data_type_fix = 'volume_native'
    else:
        output_data_type_fix = output_data_type

    print('----------------------------------------')
    print('Data ID:        %s' % data_id)
    print('BIDS directory: %s' % bids_dir)
    print('')

    # Load data ------------------------------------------------------------------
    print('----------------------------------------')
    print('Loading %s' % bids_dir)
    bdata_list, data_labels = create_bdata_fmriprep(bids_dir,
                                                    fmriprep_dir='',
                                                    data_mode=output_data_type_fix,
                                                    label_mapper=label_mapper,
                                                    exclude=exclude,
                                                    split_task_label=split_task_label,
                                                    with_confounds=True,
                                                    return_data_labels=True,
                                                    return_list=True,
                                                    subject=subject)
    print('')
    print('All data loaded')


    for bdata, data_label in zip(bdata_list, data_labels):

        # Misc fix on data_label
        data_label = re.sub('^sub-', '', data_label)
        data_label = re.sub('_task-', '_', data_label)

        # Disp info
        print('----------------------------------------')
        print('Data %s' % data_label)
        print('Num columns: %d' % bdata.dataset.shape[1])
        print('Num sample:  %d' % bdata.dataset.shape[0])

        print('')

        # Add ROIs -------------------------------------------------------------------
        print('----------------------------------------')
        print('Adding ROIs')

        if output_data_type in ['volume_native', 'volume_native_resampled', 'volume_standard']:
            # Volume data
            data_type = 'volume'
        elif output_data_type in ['surface_native', 'surface_standard', 'surface_standard_41k', 'surface_standard_10k']:
            # Surface data
            data_type = 'surface'
        else:
            raise ValueError('Unknown output data type: %s' % output_data_type)

        bdata = add_rois(bdata, rois_files[output_data_type], data_type=data_type, prefix_map=roi_prefix_mapper)
        
        print([k for k in bdata.metadata.key if 'V1' in k or 'ROI' in k])


        roi_flag = bdata.get_metadata('ROI_standard_V1') 

        # Merge ROIs -----------------------------------------------------------------
        # if output_data_type in roi_merge:
        #     roi_merge_t = roi_merge[output_data_type]
        #     for roi_name, roi_expr in roi_merge_t.items():
        #         bdata = merge_rois(bdata, roi_name, roi_expr)

        # Add HCP ROIs ---------------------------------------------------------------
        # bdata = add_hcp_rois(bdata)
        # bdata = add_hcp_visual_cortex(bdata)

        # Save raw (unpreprocessed) BData --------------------------------------------
        print('----------------------------------------')
        print('Saving raw (unpreprocessed) BData')
        save_file = os.path.join(output_dir, data_label + '_fmap_' + output_data_type + '_raw' + '.h5')
        bdata.save(save_file)
        print('Saved %s' % save_file)
        print('')

        print('Raw data size: %d x %d' % (bdata.dataset.shape[0], bdata.dataset.shape[1]))
        print('')

        # Preprocessing --------------------------------------------------------------
        print('----------------------------------------')
        print('Preprocessing')
        print('')

        # Shift data
        runs = bdata.get('Run').flatten()
        bdata.applyfunc(shift_sample, where=[brain_data_key, 'MotionParameter', 'Confounds'], group=runs,
                        shift_size=shift_size)

        # Motion regressout, mean substraction, and linear detrending
        runs = bdata.get('Run').flatten()
        motionparam = bdata.get('MotionParameter')
        bdata.applyfunc(regressout, where=brain_data_key, group=runs,
                        regressor=motionparam,
                        remove_dc=True, linear_detrend=True)

        # Outlier reduction
        runs = bdata.get('Run').flatten()
        bdata.applyfunc(reduce_outlier, where=brain_data_key, group=runs,
                        std=True, std_threshold=3, n_iter=10,
                        maxmin=False)

        # Within-block averaging
        blocks = bdata.get('Block').flatten()
        bdata.applyfunc(average_sample, where=brain_data_key, group=blocks)

        # Remove one-back repetition blocks
        trialtype = bdata.get('trial_type').flatten()
        # trialtype = bdata.get('stim_file').flatten()
        print('trialtype', trialtype)
        bdata.dataset = np.delete(bdata.dataset, np.where(trialtype == 2), axis=0)

        # Remove rest blocks
        trialtype = bdata.get('trial_type').flatten()
        # trialtype = bdata.get('stim_file').flatten()
        bdata.dataset = np.delete(bdata.dataset, np.where(trialtype < 0), axis=0)

        # Save preprocessed BData ----------------------------------------------------
        print('----------------------------------------')
        print('Saving preprocessed BData')
        save_file = os.path.join(output_dir, data_label + '_fmap_' + output_data_type + '_prep' + '.h5')
        bdata.save(save_file)
        print('Saved %s' % save_file)
        print('')

        print('Preprocessed data size: %d x %d' % (bdata.dataset.shape[0], bdata.dataset.shape[1]))
        print('')

        del bdata
        gc.collect()

print('Completed!')