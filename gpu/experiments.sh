#!/bin/bash
#SBATCH -p all
#SBATCH --job-name=f-nmf
#SBATCH --ntasks=28
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=saravanakumar693@gmail.com

source ~/.profile
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh  intel64
module load cuda
set -x

# Debug flickr and microsoft

declare -a DATASETS=( "ppi" "blogcatalog" "pubmed" )
declare -a DIMENSION=( "64" "128" "256" )
declare -a WINDOW=( "3" "10" )

for DATASET in "${DATASETS[@]}"
do
	for DIM in "${DIMENSION[@]}"
	do
		for W in "${WINDOW[@]}"
		do
			for i in $(seq 1 5); 
			do 
				time ./netmf_small_dense_hybrid $DATASET $W $DIM 1 ../../nrl-data/${DATASET}.edgelist ../embeddings/${DATASET}_${W}_${DIM}.emb ../mapping/${DATASET}_${W}_${DIM}.map
			done
		done
	done
done

for DATASET in "${DATASETS[@]}"
do
	for DIM in "${DIMENSION[@]}"
	do
		for W in "${WINDOW[@]}"
		do
			for i in $(seq 1 5); 
			do 
				time ./netmf_small_dense_mkl $DATASET $W $DIM 1 ../../nrl-data/${DATASET}.edgelist ../embeddings/${DATASET}_${W}_${DIM}.emb ../mapping/${DATASET}_${W}_${DIM}.map
			done
		done
	done
done

for DATASET in "${DATASETS[@]}"
do
	for DIM in "${DIMENSION[@]}"
	do
		for W in "${WINDOW[@]}"
		do
			for i in $(seq 1 5); 
			do 
				time ./netmf_small_sparse_hybrid $DATASET $W $DIM 1 ../../nrl-data/${DATASET}.edgelist ../embeddings/${DATASET}_${W}_${DIM}.emb ../mapping/${DATASET}_${W}_${DIM}.map
			done
		done
	done
done

