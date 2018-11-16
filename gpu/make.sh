#!/bin/bash
nvcc -o ../utils/graph.o -c ../utils/graph.cpp
nvcc -o ../utils/graphio.o -c ../utils/graphio.cpp
nvcc -o netmf.o -c netmf.cpp
nvcc main.cu ../utils/graph.o ../utils/graphio.o netmf.o -lcublas
