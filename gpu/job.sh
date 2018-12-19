#PBS -N FNMF
#PBS -A PAS1425
#PBS -l walltime=00:20:00
#PBS -l nodes=1:ppn=28:gpus=1
#PBS -j oe

cd /users/PAS1197/osu9208/fast-NetMF/gpu
set -x

module load cuda

./netmf_${ALGO}

