#!/bin/bash
#SBATCH --mail-type=ALL		
#SBATCH --mail-user=shuhei.hara1@oist.jp
#SBATCH --array=2-21
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH --time=30:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=make_bdata_fmap
#SBATCH --partition=short
##### END OF JOB DEFINITION  #####

module load python/3.11.4

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS/participants.tsv)

cmd='python make_bdata_fmap.py  --subject ${subject}'

# Setup done, run the command
echo Commandline: $cmd
eval $cmd
exitcode=$?

echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode