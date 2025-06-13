#!/bin/bash
#SBATCH --mail-type=ALL		
#SBATCH --mail-user=shuhei.hara1@oist.jp
#SBATCH --array=7
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=4:00:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=cv
#SBATCH --partition=compute
##### END OF JOB DEFINITION  #####

module load python/3.11.4

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS/participants.tsv)

cmd='python cv_train_decoder_fastl2lir.py config/cv.yaml -o subject=${subject}'

# Setup done, run the command
echo Commandline: $cmd
eval $cmd
exitcode=$?

echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode