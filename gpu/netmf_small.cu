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

void printCsr(
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const char* name)
{
    const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE)? 0:1 ;

    printf("matrix %s is %d-by-%d, nnz=%d, base=%d, output base-1\n", name, m, n, nnz, base);
    for(int row = 0 ; row < m ; row++){
        const int start = csrRowPtrA[row  ] - base;
        const int end   = csrRowPtrA[row+1] - base;
        for(int colidx = start ; colidx < end ; colidx++){
            const int col = csrColIndA[colidx] - base;
            const float Areg = csrValA[colidx];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}
__global__ void compute_d(double* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	//id = id * size + id;
	
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

__global__ void transform_s(double* S, int volume, int window_size, int b, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = (S[id] * float(volume))/ ((float) window_size * (float) b); 
}

__global__ void prune_m(double* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;

	if(M[id] > 1+1e-10)
		M[id] = M[id];
	else
		M[id] = 0;	
}


__global__ void transform_m(double* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	M[id] =log(M[id]);
}

__global__ void sqrt_si(float* S, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = sqrt((float) S[id]);
}

void print_matrix(double* S, int size){
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

	/* Settings */
	int window_size = 3;
	int dimension = 2;
	int b = 1;

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
	double *adj_device_dense;	
	double *degree_device_dense; 

	log("Allocating space for dense mat on device");
	cudaMalloc(&adj_device_dense, g.size * g.size * sizeof(double)); 	
	cudaMalloc(&degree_device_dense, g.size * g.size * sizeof(double)); 

	log("Copying host to device");	
	cudaMemcpy(adj_device_dense, g.adj, g.size * g.size * sizeof(double), cudaMemcpyHostToDevice);	
	cudaMemcpy(degree_device_dense, g.degree, g.size * g.size * sizeof(double), cudaMemcpyHostToDevice);

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
	cudaMalloc(&adj_csr.d_nnzPerVector, g.size * sizeof(double));
	cusparseDnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, g.size, g.size, adj_descr, adj_device_dense, adj_csr.lda, adj_csr.d_nnzPerVector, &adj_csr.nnz);

	cudaMalloc(&degree_csr.d_nnzPerVector, g.size * sizeof(double));
	cusparseDnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, g.size, g.size, degree_descr, degree_device_dense, degree_csr.lda, degree_csr.d_nnzPerVector, &degree_csr.nnz);


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

	log("Allocating host side sparse mat");
	adj_csr.h_values = (double *)malloc(adj_csr.nnz * sizeof(double));
	adj_csr.h_rowIndices = (int *)malloc((g.size+1) * sizeof(int));
	adj_csr.h_colIndices = (int *)malloc(adj_csr.nnz * sizeof(int));	
	
	log("Allocating host side sparse mat");
	degree_csr.h_values = (double *)malloc(degree_csr.nnz * sizeof(double));
	degree_csr.h_rowIndices = (int *)malloc((g.size+1) * sizeof(int));
	degree_csr.h_colIndices = (int *)malloc(degree_csr.nnz * sizeof(int));	

	log("Allocating device side sparse mat");
	cudaMalloc(&adj_csr.d_values, adj_csr.nnz * sizeof(double));
	cudaMalloc(&adj_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	cudaMalloc(&adj_csr.d_colIndices, adj_csr.nnz * sizeof(int));
	
	log("Allocating device side sparse mat");
	cudaMalloc(&degree_csr.d_values, degree_csr.nnz * sizeof(double));
	cudaMalloc(&degree_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	cudaMalloc(&degree_csr.d_colIndices, degree_csr.nnz * sizeof(int));

	cusparseDdense2csr(cusparse_handle, 
			g.size, g.size, 
			adj_descr,
		        adj_device_dense,	
			adj_csr.lda, 
			adj_csr.d_nnzPerVector, 
			adj_csr.d_values, adj_csr.d_rowIndices, adj_csr.d_colIndices); 
	cusparseDdense2csr(cusparse_handle, 
			g.size, g.size, 
			degree_descr, 
			degree_device_dense,
			degree_csr.lda, 
			degree_csr.d_nnzPerVector, 
			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices); 


	cudaMemcpy(adj_csr.h_values, 
			adj_csr.d_values, 
			adj_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(adj_csr.h_rowIndices, 
			adj_csr.d_rowIndices, 
			(g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(adj_csr.h_colIndices, 
			adj_csr.d_colIndices, 
			adj_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(degree_csr.h_values, 
			degree_csr.d_values, 
			degree_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaMemcpy(degree_csr.h_rowIndices, 
			degree_csr.d_rowIndices, 
			(g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(degree_csr.h_colIndices,
		       	degree_csr.d_colIndices, 
			degree_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

    	printf("\nOriginal adj matrix in CSR format\n\n");
    	for (int i = 0; i < adj_csr.nnz; ++i) printf("A[%i] = %f ", i, adj_csr.h_values[i]); printf("\n");

    	for (int i = 0; i < (g.size + 1); ++i) printf("RowIndices[%i] = %i \n", i, adj_csr.h_rowIndices[i]); printf("\n");

    	for (int i = 0; i < adj_csr.nnz; ++i) printf("ColIndices[%i] = %i \n", i, adj_csr.h_colIndices[i]);  

    	printf("\nOriginal degree matrix in CSR format\n\n");
    	for (int i = 0; i < degree_csr.nnz; ++i) printf("A[%i] = %f ", i, degree_csr.h_values[i]); printf("\n");

    	for (int i = 0; i < (g.size + 1); ++i) printf("RowIndices[%i] = %i \n", i, degree_csr.h_rowIndices[i]); printf("\n");

    	for (int i = 0; i < degree_csr.nnz; ++i) printf("ColIndices[%i] = %i \n", i, degree_csr.h_colIndices[i]);  

	/* CUDA housekeeping */
	float num_threads = 128;
	dim3 threads(num_threads);
	dim3 grids((int)ceil((float)g.size/num_threads));
	
	/* Compute D = D^{-1/2} */
	log("Computing normalized D");
	compute_d<<<grids, threads>>>(degree_csr.d_values, degree_csr.nnz);
	
	log("Computed normalized D");
	cudaMemcpy(degree_csr.h_values, degree_csr.d_values, degree_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i=0;i<degree_csr.nnz;i++) std::cout<<degree_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<degree_csr.nnz;i++) std::cout<<degree_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<degree_csr.h_rowIndices[i]<< " "; std::cout<<"\n";

	/* Compute X = D^{-1/2}AD^{-1/2} */

	// Compute X_temp = D^{-1/2}A first	
	cusparseMatDescr_t X_temp_descr;
	cusparseCreateMatDescr(&X_temp_descr);
	cusparseSetMatType(X_temp_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(X_temp_descr, CUSPARSE_INDEX_BASE_ZERO);

	csr X_temp_csr;
	int baseX;
       	X_temp_csr.nnz = 0;
	int *nnzTotalDevHostPtr = &X_temp_csr.nnz;

	cusparseStatus_t status;	
	cudaError_t cuda_status;

	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc(&X_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	status = cusparseXcsrgemmNnz(cusparse_handle, 
				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				g.size, g.size, g.size,
				degree_descr, degree_csr.nnz,
				degree_csr.d_rowIndices, degree_csr.d_colIndices, 
				adj_descr, adj_csr.nnz,
				adj_csr.d_rowIndices, adj_csr.d_colIndices,
				X_temp_descr, X_temp_csr.d_rowIndices,
				nnzTotalDevHostPtr
				);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "nnz calculation failed" << std::endl;
		std::cout << "status = " << status << std::endl;
		exit(0);
	}		
	std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;

	if(NULL != nnzTotalDevHostPtr){
		X_temp_csr.nnz = *nnzTotalDevHostPtr;
	}else{
		cudaMemcpy(&X_temp_csr.nnz, X_temp_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseX, X_temp_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
		X_temp_csr.nnz -= baseX;
	}

	cudaMalloc(&X_temp_csr.d_colIndices, X_temp_csr.nnz * sizeof(int));
	cudaMalloc(&X_temp_csr.d_values, X_temp_csr.nnz * sizeof(double));

	X_temp_csr.h_colIndices = (int *) malloc(X_temp_csr.nnz * sizeof(double));
	X_temp_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(double));
	X_temp_csr.h_values = (double *) malloc(X_temp_csr.nnz * sizeof(double));

	cusparseDcsrgemm(cusparse_handle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			g.size, g.size, g.size,
			degree_descr, degree_csr.nnz,
			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
			adj_descr, adj_csr.nnz,
			adj_csr.d_values, adj_csr.d_rowIndices, adj_csr.d_colIndices,
			X_temp_descr, 
			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices);
	cudaDeviceSynchronize();

	cuda_status = cudaMemcpy(X_temp_csr.h_values, X_temp_csr.d_values, X_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	std::cout<<cuda_status;
	std::cout<<cuda_status;
	cuda_status = cudaMemcpy(X_temp_csr.h_rowIndices, X_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<cuda_status;
	std::cout<<cuda_status;
	cuda_status = cudaMemcpy(X_temp_csr.h_colIndices, X_temp_csr.d_colIndices, X_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout<<cuda_status;

	log("Print X_temp");
	for(int i=0;i<X_temp_csr.nnz;i++) std::cout<<X_temp_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<X_temp_csr.nnz;i++) std::cout<<X_temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<X_temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
	
	// Compute X = X_tempD^{-1/2} second
	cusparseMatDescr_t X_descr;
	cusparseCreateMatDescr(&X_descr);
	cusparseSetMatType(X_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(X_descr, CUSPARSE_INDEX_BASE_ZERO);

	csr X_csr;
       	X_csr.nnz = 0;
	nnzTotalDevHostPtr = &X_csr.nnz;

	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc(&X_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	status = cusparseXcsrgemmNnz(cusparse_handle, 
				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				g.size, g.size, g.size,
				X_temp_descr, X_temp_csr.nnz,
				X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices, 
				degree_descr, degree_csr.nnz,
				degree_csr.d_rowIndices, degree_csr.d_colIndices,
				X_descr, X_csr.d_rowIndices,
				nnzTotalDevHostPtr
				);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "nnz calculation failed" << std::endl;
		std::cout << "status = " << status << std::endl;
		exit(0);
	}		
	std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;

	if(NULL != nnzTotalDevHostPtr){
		X_csr.nnz = *nnzTotalDevHostPtr;
	}else{
		cudaMemcpy(&X_csr.nnz, X_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseX, X_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
		X_csr.nnz -= baseX;
	}

	cudaMalloc(&X_csr.d_colIndices, X_csr.nnz * sizeof(int));
	cudaMalloc(&X_csr.h_rowIndices, (g.size + 1) * sizeof(int));
	cudaMalloc(&X_csr.d_values, X_csr.nnz * sizeof(double));

	X_csr.h_colIndices = (int *) malloc(X_csr.nnz * sizeof(double));
	X_csr.h_rowIndices = (int *) malloc(X_csr.nnz * sizeof(double));
	X_csr.h_values = (double *) malloc(X_csr.nnz * sizeof(double));

	cusparseDcsrgemm(cusparse_handle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			g.size, g.size, g.size,
			X_temp_descr, X_temp_csr.nnz,
			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices,
			degree_descr, degree_csr.nnz,
			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
			X_descr, 
			X_csr.d_values, X_csr.d_rowIndices, X_csr.d_colIndices);
	cudaDeviceSynchronize();

	cuda_status = cudaMemcpy(X_csr.h_values, X_csr.d_values, X_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cuda_status = cudaMemcpy(X_csr.h_rowIndices, X_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cuda_status = cudaMemcpy(X_csr.h_colIndices, X_csr.d_colIndices, X_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout<<cuda_status;

	log("Print X");
	for(int i=0;i<X_csr.nnz;i++) std::cout<<X_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<X_csr.nnz;i++) std::cout<<X_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<X_csr.h_rowIndices[i]<< " "; std::cout<<"\n";


	 /* Compute S = sum(X^{0}....X^{window_size}) */

	csr S_csr, temp_csr, temp1_csr, S_temp_csr;
	S_csr.nnz = X_csr.nnz;
	temp_csr.nnz = X_csr.nnz;

	cudaMalloc(&S_csr.d_values, S_csr.nnz * sizeof(double));
	cudaMalloc(&S_csr.d_colIndices, S_csr.nnz * sizeof(int));
	cudaMalloc(&S_csr.d_rowIndices, (g.size + 1) * sizeof(int));

	cudaMalloc(&temp_csr.d_values, temp_csr.nnz * sizeof(double));
	cudaMalloc(&temp_csr.d_colIndices, temp_csr.nnz * sizeof(int));
	cudaMalloc(&temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));

	log("Copying X to S");
	cudaMemcpy(S_csr.d_values, X_csr.d_values, S_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(S_csr.d_colIndices, X_csr.d_colIndices, S_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(S_csr.d_rowIndices, X_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

	log("Copying S to host");
	S_csr.h_values = (double *) malloc(S_csr.nnz * sizeof(double));
	S_csr.h_colIndices = (int *) malloc(S_csr.nnz * sizeof(int));
	S_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(int));

	cudaMemcpy(S_csr.h_values, S_csr.d_values, S_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(S_csr.h_colIndices, S_csr.d_colIndices, S_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(S_csr.h_rowIndices, S_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0;i<S_csr.nnz;i++) std::cout<<S_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<S_csr.nnz;i++) std::cout<<S_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<S_csr.h_rowIndices[i]<< " "; std::cout<<"\n";

	log("Copying X to temp");
	cudaMemcpy(temp_csr.d_values, X_csr.d_values, temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(temp_csr.d_colIndices, X_csr.d_colIndices, temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(temp_csr.d_rowIndices, X_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

	log("Copying S to host");
	temp_csr.h_values = (double *) malloc(temp_csr.nnz * sizeof(double));
	temp_csr.h_colIndices = (int *) malloc(temp_csr.nnz * sizeof(int));
	temp_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(int));

	cudaMemcpy(temp_csr.h_values, temp_csr.d_values, temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_csr.h_colIndices, temp_csr.d_colIndices, temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_csr.h_rowIndices, temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0;i<temp_csr.nnz;i++) std::cout<<temp_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<temp_csr.nnz;i++) std::cout<<temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";


	cusparseMatDescr_t S_descr;
	cusparseCreateMatDescr(&S_descr);
	cusparseSetMatType(S_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(S_descr, CUSPARSE_INDEX_BASE_ZERO);

	cusparseMatDescr_t S_temp_descr;
	cusparseCreateMatDescr(&S_temp_descr);
	cusparseSetMatType(S_temp_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(S_temp_descr, CUSPARSE_INDEX_BASE_ZERO);
	
	cusparseMatDescr_t temp_descr;
	cusparseCreateMatDescr(&temp_descr);
	cusparseSetMatType(temp_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(temp_descr, CUSPARSE_INDEX_BASE_ZERO);

	cusparseMatDescr_t temp1_descr;
	cusparseCreateMatDescr(&temp1_descr);
	cusparseSetMatType(temp1_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(temp1_descr, CUSPARSE_INDEX_BASE_ZERO);

	double alf = 1.00;
	double beta = 1.00;

	for(int i=2;i<=window_size;i++){
		std::cout<<"Computing X^"<<i<<std::endl;
		temp1_csr.nnz = 0;
		temp1_csr.d_rowIndices = NULL;
		temp1_csr.d_colIndices = NULL;
		temp1_csr.d_values = NULL;
		
		temp1_csr.h_rowIndices = NULL;
		temp1_csr.h_colIndices = NULL;
		temp1_csr.h_values = NULL;
		
		S_temp_csr.nnz = 0;
		S_temp_csr.d_rowIndices = NULL;
		S_temp_csr.d_colIndices = NULL;
		S_temp_csr.d_values = NULL;
		
		S_temp_csr.h_rowIndices = NULL;
		S_temp_csr.h_colIndices = NULL;
		S_temp_csr.h_values = NULL;

		nnzTotalDevHostPtr = &temp1_csr.nnz;

		cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
		cudaMalloc(&temp1_csr.d_rowIndices, (g.size + 1) * sizeof(double));
		status = cusparseXcsrgemmNnz(cusparse_handle, 
					CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
					g.size, g.size, g.size,
					temp_descr, temp_csr.nnz,
					temp_csr.d_rowIndices, temp_csr.d_colIndices,
					X_descr, X_csr.nnz,
					X_csr.d_rowIndices, X_csr.d_colIndices,
					temp1_descr, temp1_csr.d_rowIndices,
					nnzTotalDevHostPtr);

		if(status != 0){
			std::cout<<"Error encountered"<<std::endl;
			std::cout<<"Status: "<<status<<std::endl;
			exit(0);	

		}


		if(NULL!=nnzTotalDevHostPtr) temp1_csr.nnz = *nnzTotalDevHostPtr;
		else{
			cudaMemcpy(&temp1_csr.nnz, temp1_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&baseX, temp1_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
			temp1_csr.nnz -= baseX;
		
		}

		std::cout<<"NNZ ="<<temp1_csr.nnz<<std::endl;

		cudaMalloc(&temp1_csr.d_values, temp1_csr.nnz * sizeof(double));
		cudaMalloc(&temp1_csr.d_colIndices, temp1_csr.nnz * sizeof(int));

		cusparseDcsrgemm(cusparse_handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				g.size, g.size, g.size,
				temp_descr, temp_csr.nnz,
				temp_csr.d_values, temp_csr.d_rowIndices, temp_csr.d_colIndices,
				X_descr, X_csr.nnz,
				X_csr.d_values, X_csr.d_rowIndices, X_csr.d_colIndices,
				temp1_descr, 
				temp1_csr.d_values, temp1_csr.d_rowIndices, temp1_csr.d_colIndices);
	
		log("Printing intermediate result");					
		temp1_csr.h_values = (double *) malloc(temp1_csr.nnz * sizeof(double));
		temp1_csr.h_colIndices = (int *) malloc(temp1_csr.nnz * sizeof(int));
		temp1_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(int));

		cudaMemcpy(temp1_csr.h_values, temp1_csr.d_values, temp1_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(temp1_csr.h_colIndices, temp1_csr.d_colIndices, temp1_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(temp1_csr.h_rowIndices, temp1_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

		for(int i=0;i<temp1_csr.nnz;i++) std::cout<<temp1_csr.h_values[i]<< " "; std::cout<<"\n";
		for(int i=0;i<temp1_csr.nnz;i++) std::cout<<temp1_csr.h_colIndices[i]<< " "; std::cout<<"\n";
		for(int i=0;i<g.size + 1;i++) std::cout<<temp1_csr.h_rowIndices[i]<< " "; std::cout<<"\n";

		cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

		S_temp_csr.nnz = 0;
		nnzTotalDevHostPtr = &S_temp_csr.nnz;
		cudaMalloc(&S_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));
		
		status = cusparseXcsrgeamNnz(cusparse_handle,
					g.size, g.size,
					temp1_descr, temp1_csr.nnz, temp1_csr.d_rowIndices, temp1_csr.d_colIndices,
					S_descr,S_csr.nnz, S_csr.d_rowIndices, S_csr.d_colIndices,
					S_temp_descr, S_temp_csr.d_rowIndices, nnzTotalDevHostPtr);	

		if(status != 0){
			std::cout<<"Error encountered"<<std::endl;
			std::cout<<"Status: "<<status<<std::endl;
			exit(0);	

		}
		if(NULL!=nnzTotalDevHostPtr) S_temp_csr.nnz = *nnzTotalDevHostPtr;
		else{
			cudaMemcpy(&S_temp_csr.nnz, S_temp_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&baseX, S_temp_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
			S_temp_csr.nnz -= baseX;
		}

		std::cout<<"SUM-NNZ"<<S_temp_csr.nnz<<std::endl;

		cudaMalloc(&S_temp_csr.d_colIndices, sizeof(int) * S_temp_csr.nnz);
		cudaMalloc(&S_temp_csr.d_values, sizeof(double) * S_temp_csr.nnz);

		cusparseDcsrgeam(cusparse_handle, g.size, g.size,
				&alf,
				temp1_descr, temp1_csr.nnz,
				temp1_csr.d_values, temp1_csr.d_rowIndices, temp1_csr.d_colIndices,
				&beta,
				S_descr, S_csr.nnz,
				S_csr.d_values, S_csr.d_rowIndices, S_csr.d_colIndices,
				S_temp_descr,
				S_temp_csr.d_values, S_temp_csr.d_rowIndices, S_temp_csr.d_colIndices);

		log("Printing intermediate sum");
		S_temp_csr.h_values = (double *) malloc(sizeof(double) * S_temp_csr.nnz);
		S_temp_csr.h_rowIndices = (int *) malloc(sizeof(int) * (g.size + 1));
		S_temp_csr.h_colIndices = (int *) malloc(sizeof(int) * S_temp_csr.nnz);

		cudaMemcpy(S_temp_csr.h_values, S_temp_csr.d_values, S_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(S_temp_csr.h_rowIndices, S_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(S_temp_csr.h_colIndices, S_temp_csr.d_colIndices, S_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

		for(int i=0;i<S_temp_csr.nnz;i++) std::cout<<S_temp_csr.h_values[i]<< " "; std::cout<<"\n";
		for(int i=0;i<S_temp_csr.nnz;i++) std::cout<<S_temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
		for(int i=0;i<g.size + 1;i++) std::cout<<S_temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";

		S_csr.nnz = S_temp_csr.nnz;
		S_csr.d_values = NULL;
		S_csr.d_rowIndices = NULL;
		S_csr.d_colIndices = NULL;
		
		cudaMalloc(&S_csr.d_values, S_csr.nnz * sizeof(double)); 
		cudaMalloc(&S_csr.d_rowIndices, (g.size + 1) * sizeof(int)); 
		cudaMalloc(&S_csr.d_colIndices, S_csr.nnz * sizeof(int)); 

		cudaMemcpy(S_csr.d_values, S_temp_csr.d_values, S_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(S_csr.d_rowIndices, S_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(S_csr.d_colIndices, S_temp_csr.d_colIndices, S_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);


		temp_csr.nnz = temp1_csr.nnz;
		temp_csr.d_values = NULL;
		temp_csr.d_rowIndices = NULL;
		temp_csr.d_colIndices = NULL;

		cudaMalloc(&temp_csr.d_values, temp_csr.nnz * sizeof(double)); 
		cudaMalloc(&temp_csr.d_rowIndices, (g.size + 1) * sizeof(int)); 
		cudaMalloc(&temp_csr.d_colIndices, temp_csr.nnz * sizeof(int)); 

		cudaMemcpy(temp_csr.d_values, temp1_csr.d_values, temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(temp_csr.d_rowIndices, temp1_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(temp_csr.d_colIndices, temp1_csr.d_colIndices, temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
			
	}

        /* Compute S = S * (vol / (window_size * b)) */

	transform_s<<<grids, threads>>>(S_csr.d_values, g.volume, window_size, b, S_csr.nnz);

	S_csr.h_values = (double *)malloc(sizeof(double) * S_csr.nnz);

	cudaMemcpy(S_csr.h_values, S_csr.d_values, S_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);

	log("Transformed S values");
	for(int i=0;i<S_csr.nnz;i++) std::cout<<S_csr.h_values[i]<< " "; std::cout<<"\n";

	log("Computing M");


        /* Compute M = D^{-1/2} * S * D^{-1/2} */

	// Compute X_temp = D^{-1/2} * S first	
       	X_temp_csr.nnz = 0;
	nnzTotalDevHostPtr = &X_temp_csr.nnz;

	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc(&X_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	status = cusparseXcsrgemmNnz(cusparse_handle, 
				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				g.size, g.size, g.size,
				degree_descr, degree_csr.nnz,
				degree_csr.d_rowIndices, degree_csr.d_colIndices, 
				S_descr, S_csr.nnz,
				S_csr.d_rowIndices, S_csr.d_colIndices,
				X_temp_descr, X_temp_csr.d_rowIndices,
				nnzTotalDevHostPtr
				);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "nnz calculation failed" << std::endl;
		std::cout << "status = " << status << std::endl;
		exit(0);
	}		
	std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;

	if(NULL != nnzTotalDevHostPtr){
		X_temp_csr.nnz = *nnzTotalDevHostPtr;
	}else{
		cudaMemcpy(&X_temp_csr.nnz, X_temp_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseX, X_temp_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
		X_temp_csr.nnz -= baseX;
	}

	cudaMalloc(&X_temp_csr.d_colIndices, X_temp_csr.nnz * sizeof(int));
	cudaMalloc(&X_temp_csr.d_values, X_temp_csr.nnz * sizeof(double));

	X_temp_csr.h_colIndices = (int *) malloc(X_temp_csr.nnz * sizeof(double));
	X_temp_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(double));
	X_temp_csr.h_values = (double *) malloc(X_temp_csr.nnz * sizeof(double));

	cusparseDcsrgemm(cusparse_handle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			g.size, g.size, g.size,
			degree_descr, degree_csr.nnz,
			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
			S_descr, S_csr.nnz,
			S_csr.d_values, S_csr.d_rowIndices, S_csr.d_colIndices,
			X_temp_descr, 
			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices);
	cudaDeviceSynchronize();

	cuda_status = cudaMemcpy(X_temp_csr.h_values, X_temp_csr.d_values, X_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cuda_status = cudaMemcpy(X_temp_csr.h_rowIndices, X_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cuda_status = cudaMemcpy(X_temp_csr.h_colIndices, X_temp_csr.d_colIndices, X_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

	log("Print X_temp");
	for(int i=0;i<X_temp_csr.nnz;i++) std::cout<<X_temp_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<X_temp_csr.nnz;i++) std::cout<<X_temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<X_temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
	
	// Compute M = X_temp * D^{-1/2} second
	cusparseMatDescr_t M_descr;
	cusparseCreateMatDescr(&M_descr);
	cusparseSetMatType(M_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(M_descr, CUSPARSE_INDEX_BASE_ZERO);

	csr M_csr;
       	M_csr.nnz = 0;
	nnzTotalDevHostPtr = &M_csr.nnz;

	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc(&M_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	status = cusparseXcsrgemmNnz(cusparse_handle, 
				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				g.size, g.size, g.size,
				X_temp_descr, X_temp_csr.nnz,
				X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices, 
				degree_descr, degree_csr.nnz,
				degree_csr.d_rowIndices, degree_csr.d_colIndices,
				M_descr, M_csr.d_rowIndices,
				nnzTotalDevHostPtr
				);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "nnz calculation failed" << std::endl;
		std::cout << "status = " << status << std::endl;
		exit(0);
	}		
	std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;

	if(NULL != nnzTotalDevHostPtr){
		M_csr.nnz = *nnzTotalDevHostPtr;
	}else{
		cudaMemcpy(&M_csr.nnz, M_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseX, M_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
		M_csr.nnz -= baseX;
	}

	cudaMalloc(&M_csr.d_colIndices, M_csr.nnz * sizeof(int));
	cudaMalloc(&M_csr.h_rowIndices, (g.size + 1) * sizeof(int));
	cudaMalloc(&M_csr.d_values, M_csr.nnz * sizeof(double));

	M_csr.h_colIndices = (int *) malloc(M_csr.nnz * sizeof(double));
	M_csr.h_rowIndices = (int *) malloc(M_csr.nnz * sizeof(double));
	M_csr.h_values = (double *) malloc(M_csr.nnz * sizeof(double));

	cusparseDcsrgemm(cusparse_handle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			g.size, g.size, g.size,
			X_temp_descr, X_temp_csr.nnz,
			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices,
			degree_descr, degree_csr.nnz,
			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
			M_descr, 
			M_csr.d_values, M_csr.d_rowIndices, M_csr.d_colIndices);
	cudaDeviceSynchronize();

	cuda_status = cudaMemcpy(M_csr.h_values, M_csr.d_values, M_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cuda_status = cudaMemcpy(M_csr.h_rowIndices, M_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cuda_status = cudaMemcpy(M_csr.h_colIndices, M_csr.d_colIndices, M_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout<<cuda_status;

	log("Print M");
	for(int i=0;i<M_csr.nnz;i++) std::cout<<M_csr.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<M_csr.nnz;i++) std::cout<<M_csr.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<M_csr.h_rowIndices[i]<< " "; std::cout<<"\n";

	log("Transforming M");

	prune_m<<<grids,threads>>>(M_csr.d_values, M_csr.nnz);
	cudaMemcpy(M_csr.h_values, M_csr.d_values, M_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i=0;i<M_csr.nnz;i++) std::cout<<M_csr.h_values[i]<< " "; std::cout<<"\n";
	cudaDeviceSynchronize();



	double threshold = 1.00;
	log("Setting threshold");
	std::cout<<"Threshold:"<<threshold;
	csr filtered_M;

	cudaMalloc(&filtered_M.d_rowIndices, sizeof(int) * (g.size + 1));
	size_t lworkInBytes = 0;
	char *d_work=NULL;
	cusparseDpruneCsr2csr_bufferSizeExt(cusparse_handle,
        					g.size,g.size,
        					M_csr.nnz,M_descr,
        					M_csr.d_values,M_csr.d_rowIndices,M_csr.d_colIndices,
        					&threshold,
        					M_descr,
        					filtered_M.d_values, filtered_M.d_rowIndices, filtered_M.d_colIndices,
        					&lworkInBytes);	

    	printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
	cudaMalloc((void**)&d_work, lworkInBytes);

	cusparseDpruneCsr2csrNnz(cusparse_handle,
					g.size,g.size,
        				M_csr.nnz,M_descr,
  					M_csr.d_values,M_csr.d_rowIndices,M_csr.d_colIndices,
        				&threshold,
        				M_descr,
        				filtered_M.d_rowIndices, &filtered_M.nnz, /* host */
        				d_work);

	printf("nnzC = %d\n", filtered_M.nnz);
    	if (0 == filtered_M.nnz ){
        	printf("C is empty \n");
        	return 0;
    	}


	cudaMalloc(&filtered_M.d_colIndices, sizeof(int) * filtered_M.nnz);
	cudaMalloc(&filtered_M.d_values, sizeof(double) * filtered_M.nnz);

	cusparseDpruneCsr2csr(cusparse_handle,
   				g.size,g.size,
        			M_csr.nnz, M_descr,
        			M_csr.d_values, M_csr.d_rowIndices, M_csr.d_colIndices,
        			&threshold,
        			M_descr,
        			filtered_M.d_values,filtered_M.d_rowIndices, filtered_M.d_colIndices,
        			d_work);


	log("Printing pruned M");
	filtered_M.h_values = (double *) malloc(sizeof(double) * filtered_M.nnz);
	filtered_M.h_colIndices = (int *) malloc(sizeof(int) * filtered_M.nnz);
	filtered_M.h_rowIndices = (int *) malloc(sizeof(int) * (g.size + 1));

	cudaMemcpy(filtered_M.h_values, filtered_M.d_values, sizeof(double) * filtered_M.nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(filtered_M.h_colIndices, filtered_M.d_colIndices, sizeof(int) * filtered_M.nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(filtered_M.h_rowIndices, filtered_M.d_rowIndices, sizeof(int) * (g.size + 1), cudaMemcpyDeviceToHost);
	
	for(int i=0;i<filtered_M.nnz;i++) std::cout<<filtered_M.h_values[i]<< " "; std::cout<<"\n";
	for(int i=0;i<filtered_M.nnz;i++) std::cout<<filtered_M.h_colIndices[i]<< " "; std::cout<<"\n";
	for(int i=0;i<g.size + 1;i++) std::cout<<filtered_M.h_rowIndices[i]<< " "; std::cout<<"\n";
	
	log("Printing log of M");
	transform_m<<<grids,threads>>>(filtered_M.d_values, filtered_M.nnz);

	cudaMemcpy(filtered_M.h_values, filtered_M.d_values, sizeof(double) * filtered_M.nnz, cudaMemcpyDeviceToHost);
	for(int i=0;i<filtered_M.nnz;i++) std::cout<<filtered_M.h_values[i]<< " "; std::cout<<"\n";

		

}
