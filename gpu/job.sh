#!/bin/sh
#SBATCH -p bdw-v100
#SBATCH --job-name=fnmf
#SBATCH --ntasks=28
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=saravanakumar693@gmail.com

declare -a DATASETS=( "pubmed" )
declare -a WINDOW=( "1" "3" "10")
declare -a DIMS=("64" "128" "256" )

for DATASET in "${DATASETS[@]}"
do
	for WINDOW in "${WINDOW[@]}"
	do
		for DIM in "${DIMS[@]}"
		do
			#bash experiments_gpu.sh $DATASET $DIM $WINDOW
		#	bash experiments_pubmed_gpu.sh $DATASET $DIM $WINDOW
			#bash experiments_cpu.sh $DATASET $DIM $WINDOW
			bash experiments_cpu_pubmed.sh $DATASET $DIM $WINDOW
		done
	done
done
