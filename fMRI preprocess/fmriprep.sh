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
SING_DIR=/path/to/singularity_images

##### Set up template and fMRIPrep cache #####
TEMPLATEFLOW_HOST_HOME=$STUDY/.cache/templateflow
FMRIPREP_HOST_CACHE=$STUDY/.cache/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}

export SINGULARITYENV_FS_LICENSE=$STUDY/license.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=$STUDY/.templateflow

SINGULARITY_CMD="singularity run --cleanenv \
  -B ${BIDS_DIR}:/data \
  -B ${OUTPUT_DIR}:/deriv \
  -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
  -B ${FMRIPREP_HOST_CACHE}:/work \
  ${SING_DIR}/fmriprep_23.2.1.sif"

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" BIDS/participants.tsv)

##### fMRIPrep command #####
CMD="${SINGULARITY_CMD} /data /deriv participant \
  --participant-label ${subject} \
  -w /work/ \
  -vv \
  --omp-nthreads 8 \
  --nthreads 12 \
  --mem_mb 60000 \
  --output-spaces func"

##### Run #####
echo "Running task ${SLURM_ARRAY_TASK_ID} for subject: ${subject}"
echo "Commandline: ${CMD}"
eval $CMD
exitcode=$?

echo "Finished subject ${subject} (task ${SLURM_ARRAY_TASK_ID}) with exit code ${exitcode}"
exit $exitcode