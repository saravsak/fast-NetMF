GPU_ARCH 	:= -arch=sm_61
ROOT		:= /home/sk

CXX		:= icc
CXXFLAGS	:= -O3 -march=native -std=c++11 -fPIC -qopenmp -mkl -g

ICCROOT		:= /opt/intel/system_studio_2019/compilers_and_libraries/linux
ICCINCLUDE	:= $(ICCROOT)/include
ICCLIB		:= $(ICCROOT)/lib/intel64_lin
ICCINC		:= $(ICCLIB)/libirc.a,$(ICCLIB)/libiomp5.so

MKLROOT 	:= $(ROOT)/intel/compilers_and_libraries/linux/mkl
MKLINCLUDE 	:= $(MKLROOT)/include
MKLLIB 		:= $(MKLROOT)/lib/intel64_lin
MKLICC		:= $(MKLLIB)/libmkl_intel_lp64.a,$(MKLLIB)/libmkl_sequential.a,$(MKLLIB)/libmkl_core.a,$(ICCINC),-lpthread

NVCC		:= nvcc
NVCCFLAGS 	:= -O3 -w $(GPUARCH) --std=c++11 -m64 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp -g
NVCCLINK	:= -lcublas -lcusolver -lcusparse -lcudart

OBJS		:= utils.o graph.o io.o

#MKLFLAGS:= -I $(MKLINCLUDE)
#ICCLINK:= /opt/intel/compilers_and_libraries/linux/lib/intel64_lin
#ICCLINK:= /opt/intel/system_studio_2019/lib/intel64_lin
#MKLLINKFLAGS:= --linker-options $(MKLICC)
#ICCLINK		:= -l$(MKLLIB)/libmkl_intel_lp64.a -l$(MKLLIB)/libmkl_sequential.a -l$(MKLLIB)/libmkl_core.a -l$(ICCINC) -lpthread

NMFROOT:=../lib/nmf
NMF:=$(NMFROOT)/nmf-utils.o $(NMFROOT)/update_kernel.o $(NMFROOT)/model.o


all: gpu cpu

gpu: small_dense_gpu small_sparse_gpu large_gpu

cpu: small_dense_cpu small_sparse_cpu large_cpu

large: large_gpu large_cpu

small: small_dense_gpu small_sparse_gpu small_dense_cpu small_sparse_cpu 

small_dense_gpu: netmf_small_dense_hybrid.cu $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o small_dense_gpu $< $(OBJS) $(NVCCLINK) -I $(MKLINCLUDE) --linker-options $(MKLICC)

small_sparse_gpu: netmf_small_sparse_hybrid.cu $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o small_sparse_gpu $< $(OBJS) $(NVCCLINK) -I $(MKLINCLUDE) --linker-options $(MKLICC)

large_gpu: netmf_large_dense_hybrid.cu $(OBJS)
	${NVCC} ${NVCCFLAGS} -o large_dense_gpu $< $(OBJS) $(NVCCLINK) -I $(MKLINCLUDE) --linker-options $(MKLICC)

small_dense_cpu: netmf_small_dense_mkl.cpp $(OBJS)
	${CXX} ${CXXFLAGS} -o small_dense_cpu $< $(OBJS) -I $(MKLINCLUDE) 

small_sparse_cpu: netmf_small_sparse_mkl.cpp $(OBJS)
	${CXX} ${CXXFLAGS} -o small_sparse_cpu $< $(OBJS) -I $(MKLINCLUDE)

large_cpu: netmf_large_dense_mkl.cpp $(OBJS)
	${CXX} ${CXXFLAGS} -o large_dense_cpu $< $(OBJS) -I $(MKLINCLUDE)

utils.o: ../utils/utils.cpp ../utils/utils.h
	${CXX} ${CXXFLAGS} -c $< -o utils.o

graph.o: ../utils/graph.cpp ../utils/graph.h
	${CXX} ${CXXFLAGS} -c $< -o graph.o

io.o: ../utils/io.cpp ../utils/io.h
	${CXX} ${CXXFLAGS} -c $< -o io.o
clean:
	rm -rf $(OBJS) small_dense_gpu small_sparse_gpu small_dense_cpu small_sparse_cpu large_gpu large_cpu 
