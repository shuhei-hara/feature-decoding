#!/bin/bash
#SBATCH --mail-type=ALL		
#SBATCH --mail-user=shuhei.hara1@oist.jp
#SBATCH --array=1-21
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:30:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=predict
#SBATCH --partition=short
##### END OF JOB DEFINITION  #####

module load python/3.11.4

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS/participants.tsv)

# cmd1='python predict_decoder.py config/deco_alexnet.yaml -o subject=${subject}'
# eval $cmd1

cmd2='python evaluation.py config/deco_alexnet.yaml -o subject=${subject}'
eval $cmd2

exitcode=$?
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode