#!/bin/bash

export MKL_ROOT=~/intel/compilers_and_libraries_2019.1.144/linux/mkl
export MKLINCLUDE=$MKL_ROOT/include
export MKLPATH=$MKL_ROOT/lib/intel64_lin
export MATLAB=/usr/local/MATLAB/R2018b/extern/include

export FLAGS="../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver -lcusparse -m64 -I$MKLINCLUDE --linker-options $MKLPATH/libmkl_intel_lp64.a,$MKLPATH/libmkl_sequential.a,$MKLPATH/libmkl_core.a,-lpthread" 

nvcc -o ../utils/graph.o -c ../utils/graph.cpp
nvcc -o ../utils/io.o -c ../utils/io.cpp
nvcc -o ../utils/utils.o -c ../utils/utils.cpp

nvcc -g -O3 -o netmf_small_sparse_hybrid netmf_small_sparse_hybrid.cu $FLAGS
#nvcc -g -O3 -o netmf_small_dense_hybrid netmf_small_dense_hybrid.cu $FLAGS
#nvcc -g -O3 -o netmf_large_dense_hybrid netmf_large_dense_hybrid.cu $FLAGS

#nvcc -g -O3 -o netmf_small_dense_unified_hybrid netmf_small_dense_unified_hybrid.cu $FLAGS

#nvcc -g -O3 -o netmf netmf.cu $FLAGS

