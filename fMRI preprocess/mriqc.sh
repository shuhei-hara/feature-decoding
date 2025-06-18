# ===== SLURM Job Options =====
#SBATCH --job-name=your_job_name         
#SBATCH --partition=your_partition_name  
#SBATCH --array=1-3                      # the number of subjects                     
#SBATCH -n 1                             
#SBATCH --cpus-per-task=4                
#SBATCH --mem-per-cpu=32G                
#SBATCH --time=estimated_time                   
#SBATCH --mail-type=ALL                  
#SBATCH -o log/output-%A-%a.txt  



module load singularity

export STUDY=/path/to/your/home_or_project_directory
BIDS_DIR=/path/to/bids_dataset
OUTPUT_DIR=/path/to/output_directory

##### Set up template and fMRIPrep cache #####
TEMPLATEFLOW_HOST_HOME=$STUDY/.cache/templateflow
FMRIPREP_HOST_CACHE=$STUDY/.cache/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}


export SINGULARITYENV_FS_LICENSE=$STUDY/license.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=$STUDY/.templateflow

SINGULARITY_CMD='singularity run --cleanenv $STUDY/mriqc-0.15.1.simg'

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" BIDS/participants.tsv)

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
