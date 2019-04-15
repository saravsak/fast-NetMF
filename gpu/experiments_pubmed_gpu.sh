#!/bin/sh
#SBATCH -p bdw-v100
#SBATCH --job-name=fnmf
#SBATCH --ntasks=28
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=saravanakumar693@gmail.com

source activate nrlscriptenv
module load cuda/9.2
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh  intel64
#export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2019.0.117/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries/linux/lib/intel64_lin/


# Debug flickr and microsoft
set -x

DATASET=$1
DIM=$2
W=$3


time python ~/NetMF/predict.py --label ~/fnmf-data/${DATASET}.mat --embedding /scratch/SK/fnmf_embeddings/gpu_dense_${DATASET}_${DIM}_${W}_1_SVD.emb --seed 10 --out-path ./gpu_result.csv --algorithm netmf-small-gpu-dense --dataset ${DATASET} --window $W --dimension $DIM --b 1 --fact SVD
time python ~/NetMF/predict.py --label ~/fnmf-data/${DATASET}.mat --embedding /scratch/SK/fnmf_embeddings/gpu_sparse_${DATASET}_${DIM}_${W}_1_SVD.emb --seed 10 --out-path ./gpu_result.csv --algorithm netmf-small-gpu-sparse --dataset ${DATASET} --window $W --dimension $DIM --b 1 --fact SVD

time python ~/NetMF/predict.py --label ~/fnmf-data/${DATASET}.mat --embedding /scratch/SK/fnmf_embeddings/gpu_dense_${DATASET}_${DIM}_${W}_1_NMF.emb --seed 10 --out-path ./gpu_result.csv --algorithm netmf-small-gpu-dense --dataset ${DATASET} --window $W --dimension $DIM --b 1 --fact NMF
time python ~/NetMF/predict.py --label ~/fnmf-data/${DATASET}.mat --embedding /scratch/SK/fnmf_embeddings/gpu_sparse_${DATASET}_${DIM}_${W}_1_NMF.emb --seed 10 --out-path ./gpu_result.csv --algorithm netmf-small-gpu-sparse --dataset ${DATASET} --window $W --dimension $DIM --b 1 --fact NMF



