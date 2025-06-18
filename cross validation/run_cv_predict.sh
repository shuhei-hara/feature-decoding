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

module load python/3.11.4

subject= subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" BIDS/participants.tsv)

cmd1='python cv_predict_feature_fastl2lir.py config/cv.yaml -o subject=${subject}'
eval $cmd1

cmd2='python cv_evaluation.py config/cv.yaml -o subject=${subject}'
eval $cmd2

exitcode=$?
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
