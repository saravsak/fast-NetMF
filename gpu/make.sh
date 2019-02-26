#!/bin/bash

export MKL_ROOT=~/intel/compilers_and_libraries_2019.1.144/linux/mkl
export MKLINCLUDE=$MKL_ROOT/include
export MKLPATH=$MKL_ROOT/lib/intel64_lin
export MATLAB=/usr/local/MATLAB/R2018b/extern/include

nvcc -o ../utils/graph.o -c ../utils/graph.cpp
nvcc -o ../utils/io.o -c ../utils/io.cpp
nvcc -o ../utils/utils.o -c ../utils/utils.cpp
#nvcc -o netmf_small netmf_small.cu ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver -lcusparse
#nvcc -o netmf_large netmf_large.cu ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver
#nvcc -I ../lib -O2 -o netmf_small_redsvd netmf_small_redsvd.cu ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver  
nvcc -g -O2 -I$MATLAB -o netmf_small netmf_small.cu ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver -lcusparse -m64 -I$MKLINCLUDE \
    --linker-options $MKLPATH/libmkl_intel_lp64.a,$MKLPATH/libmkl_sequential.a,$MKLPATH/libmkl_core.a,-lpthread \


