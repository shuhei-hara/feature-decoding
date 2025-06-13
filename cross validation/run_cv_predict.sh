#!/bin/bash
#SBATCH --mail-type=ALL		
#SBATCH --mail-user=shuhei.hara1@oist.jp
#SBATCH --array=1-21
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=cv
#SBATCH --partition=short
##### END OF JOB DEFINITION  #####

module load python/3.11.4

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS/participants.tsv)

# cmd1='python cv_predict_feature_fastl2lir.py config/cv.yaml -o subject=${subject}'
# eval $cmd1

cmd2='python cv_evaluation.py config/cv.yaml -o subject=${subject}'
eval $cmd2

exitcode=$?
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode