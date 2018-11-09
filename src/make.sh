nvcc -o graph.o -c graph.cpp
nvcc -o utils.o -c utils.cpp
nvcc -o netmf.o -c netmf.cpp
nvcc main.cpp graph.o utils.o netmf.o
