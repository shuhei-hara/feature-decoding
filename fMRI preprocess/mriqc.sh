#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --array= # number of subject 
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=mriqc
#SBATCH --partition=compute
##### END OF JOB DEFINITION  #####

export STUDY=#Home directory

module load singularity

BIDS_DIR=""

TEMPLATEFLOW_HOST_HOME=$STUDY/.cache/templateflow
FMRIPREP_HOST_CACHE=$STUDY/.cache/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}

OUTPUT_DIR=''

export SINGULARITYENV_FS_LICENSE=$STUDY/license.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=$STUDY/.templateflow

SINGULARITY_CMD='singularity run --cleanenv $STUDY/mriqc-0.15.1.simg'

subject= #subject name

cmd="${SINGULARITY_CMD} ${BIDS_DIR} ${OUTPUT_DIR} participant --participant-label $subject -w /flash/DoyaU/shuhei/work"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
