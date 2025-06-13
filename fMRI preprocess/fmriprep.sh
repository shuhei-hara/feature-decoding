#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --array= # number of subject
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=fmriprep
#SBATCH --partition=compute
##### END OF JOB DEFINITION  #####

export STUDY=# Home directory

module load singularity

BIDS_DIR=""
SING_DIR=''

TEMPLATEFLOW_HOST_HOME=$STUDY/.cache/templateflow
FMRIPREP_HOST_CACHE=$STUDY/.cache/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}

OUTPUT_DIR=''

export SINGULARITYENV_FS_LICENSE=$STUDY/license.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=$STUDY/.templateflow

SINGULARITY_CMD='singularity run --cleanenv -B $BIDS_DIR:/data -B $OUTPUT_DIR:/deriv -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} -B '':/work ${SING_DIR}/fmriprep_23.2.1.sif'

subject= #subject
echo Subject: $subject

cmd="${SINGULARITY_CMD} /data /deriv participant --participant-label $subject -w /work/ -vv --omp-nthreads 8 --nthreads 12 --mem_mb 60000 --output-spaces func"


# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
# echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
#       >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
