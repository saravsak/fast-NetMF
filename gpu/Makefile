CXX=g++
CXXFLAGS=-O3 -std=c++11 

NVCC=nvcc
NVCCFLAGS += -O3 -w -arch=sm_61 --std=c++11 -m64 -g #-G #-rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp #-lineinfo 
NVCCLINKFLAGS= -lcublas -lcusolver -lcusparse -lcudart

MKLROOT = ~/intel/compilers_and_libraries_2019.1.144/linux/mkl
MKLINCLUDE := $(MKLROOT)/include
MKLPATH := $(MKLROOT)/lib/intel64_lin

MKLFLAGS:= -I $(MKLINCLUDE)
MKLLINKFLAGS:= --linker-options $(MKLPATH)/libmkl_intel_lp64.a,$(MKLPATH)/libmkl_sequential.a,$(MKLPATH)/libmkl_core.a,-lpthread


all: netmf_small_sparse netmf_small_dense netmf_large_dense

netmf_small_sparse: netmf_small_sparse_hybrid.cu utils.o graph.o io.o
	${NVCC} ${NVCCFLAGS} -o netmf_small_sparse_hybrid $< utils.o graph.o io.o $(NVCCLINKFLAGS) $(MKLFLAGS) $(MKLLINKFLAGS)

netmf_small_dense: netmf_small_dense_hybrid.cu utils.o graph.o io.o
	${NVCC} ${NVCCFLAGS} -o netmf_small_dense_hybrid $< utils.o graph.o io.o $(NVCCLINKFLAGS) $(MKLFLAGS) $(MKLLINKFLAGS)

netmf_large_dense: netmf_large_dense_hybrid.cu utils.o graph.o io.o
	${NVCC} ${NVCCFLAGS} -o netmf_large_dense_hybrid $< utils.o graph.o io.o $(NVCCLINKFLAGS) $(MKLFLAGS) $(MKLLINKFLAGS)

utils.o: ../utils/utils.cpp ../utils/utils.h
	${CXX} ${CXXFLAGS} -c $< -o utils.o

graph.o: ../utils/graph.cpp ../utils/graph.h
	${CXX} ${CXXFLAGS} -c $< -o graph.o

io.o: ../utils/io.cpp ../utils/io.h
	${CXX} ${CXXFLAGS} -c $< -o io.o
clean:
	rm -rf netmf_small_sparse_hybrid netmf_small_dense_hybrid netmf_large_dense_hybrid *.o 
