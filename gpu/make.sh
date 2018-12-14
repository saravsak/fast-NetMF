#!/bin/bash
nvcc -o ../utils/graph.o -c ../utils/graph.cpp
nvcc -o ../utils/spgraph.o -c ../utils/spgraph.cpp
nvcc -o ../utils/io.o -c ../utils/io.cpp
#nvcc -o ../utils/graphio.o -c ../utils/graphio.cpp
nvcc -o ../utils/utils.o -c ../utils/utils.cpp
nvcc -o netmf_small netmf_small.cu ../utils/spgraph.o ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver
nvcc -o netmf_large netmf_large.cu ../utils/spgraph.o ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver
nvcc -o main main.cu ../utils/spgraph.o ../utils/graph.o ../utils/io.o ../utils/utils.o -lcublas -lcusolver

