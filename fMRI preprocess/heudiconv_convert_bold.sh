DIR_OUTPUT=path_to_outout
DIR_CODE=path_to_current_directory

ID_USER=$(whoami | id -u)

SUB_ID=subject_ID
DIR_INPUT=path_to_NiFTI_data




for sess in `seq 1 5`; do
	for trial in `seq 5 2 9`; do
		# Convert data to BIDS format using heuristic file
		docker run --rm -it \
			--user ${ID_USER} \
			-v $DIR_INPUT:/data:ro \
			-v $DIR_OUTPUT:/output \
			-v $DIR_CODE:/code \
			nipy/heudiconv:latest \
				-d /data/{subject}/ses-{session}/${trial}/*.dcm \
				-s $SUB_ID \
				-ss ${sess} \
				-f /code/heuristic.py \
				-c dcm2niix \
				-b \
				-o /output \
				--overwrite
		# Clean up .heudiconv (avoids reuse issues with heuristic)
		rm -r ${DIR_OUTPUT}/.heudiconv
	done
done	