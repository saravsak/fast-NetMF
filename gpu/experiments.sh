#!/bin/bash
#SBATCH -p fpga
#SBATCH --job-name=fnmf
#SBATCH --ntasks=28
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=saravanakumar693@gmail.com

source ~/.profile
source nrlscriptenv
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh  intel64
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2019.0.117/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries/linux/lib/intel64_lin/

set -x

# Debug flickr and microsoft

DATASET=$1
DIM=$2
W=$3


time ./small_sparse_cpu $DATASET $W $DIM 1 ../../fnmf-data/${DATASET}/csr/ /scratch/SK/fnmf_embeddings/mkl_sparse_${DATASET}_${DIM}_${W}_1_SVD.emb SVD
time ./small_dense_cpu $DATASET $W $DIM 1 ../../fnmf-data/${DATASET}/dense/ /scratch/SK/fnmf_embeddings/mkl_dense_${DATASET}_${DIM}_${W}_1_SVD.emb SVD

time python ~/NetMF/predict.py --label ~/fnmf-data/${DATASET}.mat --embedding /scratch/SK/fnmf_embeddings/mkl_dense_${DATASET}_${DIM}_${W}_1_SVD.emb --seed 10 --out-path ./mkl_result.csv --algorithm netmf-small-mkl-dense --dataset ${DATASET} --window $W --dimension $DIM --b 1 --fact SVD
time python ~/NetMF/predict.py --label ~/fnmf-data/${DATASET}.mat --embedding /scratch/SK/fnmf_embeddings/mkl_sparse_${DATASET}_${DIM}_${W}_1_SVD.emb --seed 10 --out-path ./mkl_result.csv --algorithm netmf-small-mkl-sparse --dataset ${DATASET} --window $W --dimension $DIM --b 1 --fact SVD


