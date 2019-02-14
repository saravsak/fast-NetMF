#include<stdlib.h>
#include<iostream>
#include<time.h>
#include<chrono>
#include<algorithm>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include <cusparse_v2.h>

#include "../utils/graph.h"
#include "../utils/io.h"

//#include<mkl.h>
//#include<mkl_solvers_ee.h>
//#include<mkl_spblas.h>
//#include<mkl_feast_evcount.h>


__global__ void compute_d(float* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	id = id * size + id;
	
	/* Make assumption here that graph is  	*/
	/* connected and every node has degree 	*/
        /* atleast 1. 		       		*/

	deg[id] = sqrt(1/deg[id]); 
}

__global__ void compute_s(float* S, float* X, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] += X[id]; 
}

__global__ void transform_s(float* S, int volume, int window_size, int b, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] = (S[id] * float(volume))/ ((float) window_size * (float) b); 
}

__global__ void transform_m(float* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	M[id] =logf(M[id] > 1?M[id]:1);
}

__global__ void sqrt_si(float* S, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = sqrt((float) S[id]);
}

void print_matrix(float* S, int size){
	std::cout<<std::endl<<std::endl;
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			std::cout<<S[i*size + j]<<" ";
		}
		std::cout<<std::endl;
	}

}


int main ( void ){
	/**************
 	* NetMF small *
	**************/

	/* CuSparse housekeeping */
	cusparseHandle_t cusparse_handle;    
	cusparseCreate(&cusparse_handle);	

	/* Load graph */
        log("Reading data from file");

	Graph g =  read_graph("../data/test/small_test.csv","edgelist");

	log("Printing adj matrix");
	print_matrix(g.adj, g.size);	
	
	log("Printing degree matrix");
	print_matrix(g.degree, g.size);	
	/* Convert graph to sparse */	
	// Create dense device array

	log("Creating dense device array");
	float *adj_device_dense;	
	float *degree_device_dense; 

	log("Allocating space for dense mat on device");
	cudaMalloc(&adj_device_dense, g.size * g.size * sizeof(float)); 	
	cudaMalloc(&degree_device_dense, g.size * g.size * sizeof(float)); 

	log("Copying host to device");	
	cudaMemcpy(adj_device_dense, g.adj, g.size * g.size * sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(degree_device_dense, g.degree, g.size * g.size * sizeof(float), cudaMemcpyHostToDevice);

	log("Creating matrix descriptors");	
	cusparseMatDescr_t adj_descr;
	cusparseCreateMatDescr(&adj_descr);
	cusparseSetMatType(adj_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(adj_descr, CUSPARSE_INDEX_BASE_ZERO);
	
	log("Creating matrix descriptors");	
	cusparseMatDescr_t degree_descr;
	cusparseCreateMatDescr(&degree_descr);
	cusparseSetMatType(degree_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(degree_descr, CUSPARSE_INDEX_BASE_ZERO);

	csr adj_csr, degree_csr;

	adj_csr.nnz = 0;
	degree_csr.nnz = 0;

	adj_csr.lda = g.size;
	degree_csr.lda = g.size;
	
	log("Computing nnzPerVector");	
	cudaMalloc(&adj_csr.d_nnzPerVector, g.size * sizeof(float));
	cusparseSnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, g.size, g.size, adj_descr, adj_device_dense, adj_csr.lda, adj_csr.d_nnzPerVector, &adj_csr.nnz);

	cudaMalloc(&degree_csr.d_nnzPerVector, g.size * sizeof(float));
	cusparseSnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, g.size, g.size, degree_descr, degree_device_dense, degree_csr.lda, degree_csr.d_nnzPerVector, &degree_csr.nnz);


	log("Computing nnzPerVector host");	
	adj_csr.h_nnzPerVector = (int *)malloc(g.size * sizeof(int));
	cudaMemcpy(adj_csr.h_nnzPerVector, adj_csr.d_nnzPerVector, g.size * sizeof(int), cudaMemcpyDeviceToHost);

	degree_csr.h_nnzPerVector = (int *)malloc(g.size * sizeof(int));
	cudaMemcpy(degree_csr.h_nnzPerVector, degree_csr.d_nnzPerVector, g.size * sizeof(int), cudaMemcpyDeviceToHost);

    	printf("Number of nonzero elements in dense adjacency matrix = %i\n\n", adj_csr.nnz);
    	for (int i = 0; i < g.size; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, adj_csr.h_nnzPerVector[i]);
    	printf("\n");

    	printf("Number of nonzero elements in dense degree matrix = %i\n\n", degree_csr.nnz);
    	for (int i = 0; i < g.size; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, degree_csr.h_nnzPerVector[i]);
    	printf("\n");

}
