DIR_INPUT=path_to_T1_data
DIR_OUTPUT=path_to_output
DIR_CODE=path_to_current_directory
ID_USER=$(whoami | id -u)

# To get list of subjects, don't put anything except subject directories
SUB_LIST=($(ls $DIR_INPUT))
for SUB_ID in ${SUB_LIST[@]}; do
	# Convert data to BIDS format using heuristic file
	docker run --rm -it \
		--user ${ID_USER} \
		-v $DIR_INPUT:/data:ro \
		-v $DIR_OUTPUT:/output \
		-v $DIR_CODE:/code \
		nipy/heudiconv:latest \
			-d /data/{subject}/1/*.dcm \
			-s $SUB_ID \
			-f /code/heuristic.py \
			-c dcm2niix \
			-b \
			-o /output \
			--overwrite
done

# Remove .heudiconv directory because of conflict
# rm -r ${DIR_OUTPUT}/.heudiconv