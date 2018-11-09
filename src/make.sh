nvcc -o graph.o -c graph.cpp
nvcc -o utils.o -c utils.cpp
nvcc main.cpp graph.o utils.o

