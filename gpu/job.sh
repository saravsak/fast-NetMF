declare -a DATASETS=( "blogcatalog" "microsoft" "flickr" "youtube" )
declare -a WINDOW=( "1" "3" "10" )
declare -a DIMS=("64" "128" "256" )

for DATASET in "${DATASETS[@]}"
do
	for WINDOW in "${WINDOW[@]}"
	do
		for DIM in "${DIMS[@]}"
		do
			sbatch experiments.sh $DATASET $DIM $WINDOW
		done
	done
done
