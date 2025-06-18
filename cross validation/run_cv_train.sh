# ===== SLURM Job Options =====
#SBATCH --job-name=your_job_name         
#SBATCH --partition=your_partition_name  
#SBATCH --array=1-3                      # the number of subjects                     
#SBATCH -n 1                             
#SBATCH --cpus-per-task=4                
#SBATCH --mem-per-cpu=32G                
#SBATCH --time=4:00:00                   
#SBATCH --mail-type=ALL                  
#SBATCH -o log/output-%A-%a.txt          

module load python/3.11.4

subject= subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" BIDS/participants.tsv)

cmd='python cv_train_decoder_fastl2lir.py config/cv.yaml -o subject=${subject}'

# Setup done, run the command
echo Commandline: $cmd
eval $cmd
exitcode=$?

echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode