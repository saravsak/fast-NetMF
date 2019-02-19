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

#include<mkl.h>
#include<mkl_solvers_ee.h>
#include<mkl_spblas.h>
//#include<mkl_feast_evcount.h>

#define DEBUG true


void print_csr(
    int m,
    int nnz,
    csr mat_csr,
    const char* name)
{
    printf("matrix %s is %d-by-%d, nnz=%d\n", name, m, m, nnz);
    std::cout<<"Values: "; for(int i=0;i<nnz;i++) std::cout<<mat_csr.h_values[i]<<" "; std::cout<<'\n';
    std::cout<<"Cols: "; for(int i=0;i<nnz;i++) std::cout<<mat_csr.h_colIndices[i]<<" "; std::cout<<'\n';
    std::cout<<"Rows: "; for(int i=0;i<m+1;i++) std::cout<<mat_csr.h_rowIndices[i]<<" "; std::cout<<'\n';
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


void allocate_csr_row(csr *csr_mat, int num_rows){
	csr_mat->h_rowIndices = (int *) malloc((num_rows+1) * sizeof(int));
	cudaMalloc(&csr_mat->d_rowIndices, (num_rows+1) * sizeof(int));

}
void allocate_csr_col_val(csr *csr_mat, int nnz){
	csr_mat->h_values = (double *) malloc(nnz * sizeof(double));
	csr_mat->h_colIndices = (int *) malloc(nnz * sizeof(int));

	cudaMalloc(&csr_mat->d_values, nnz*sizeof(double));
	cudaMalloc(&csr_mat->d_colIndices, nnz * sizeof(int));

}

void allocate_csr(csr *csr_mat, int nnz, int num_rows){
	csr_mat->h_values = (double *) malloc(nnz * sizeof(double));
	csr_mat->h_colIndices = (int *) malloc(nnz * sizeof(int));
	csr_mat->h_rowIndices = (int *) malloc((num_rows+1) * sizeof(int));

	cudaMalloc(&csr_mat->d_values, nnz*sizeof(double));
	cudaMalloc(&csr_mat->d_colIndices, nnz * sizeof(int));
	cudaMalloc(&csr_mat->d_rowIndices, (num_rows+1) * sizeof(int));

}


void device2host(csr *csr_mat, int nnz, int num_rows){
	cudaMemcpy(csr_mat->h_values, csr_mat->d_values, 
			nnz * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_mat->h_colIndices, csr_mat->d_colIndices, 
			nnz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_mat->h_rowIndices, csr_mat->d_rowIndices, 
			(num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
}

void host2device(csr *csr_mat, int nnz, int num_rows){
	cudaMemcpy(csr_mat->d_values, csr_mat->h_values, 
			nnz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_mat->d_colIndices, csr_mat->h_colIndices, 
			nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_mat->d_rowIndices, csr_mat->h_rowIndices, 
			(num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

void multiply_csr(csr *A, csr *B, csr *C, int m, int n, int k, cusparseHandle_t context,cusparseMatDescr_t descr ){

	cusparseStatus_t status;

	int base;
	int *nnzTotalDevHostPtr;
	C->nnz = 0;
	nnzTotalDevHostPtr = &C->nnz;

	cusparseSetPointerMode(context, CUSPARSE_POINTER_MODE_HOST);

	allocate_csr_row(C, m);
	cudaMalloc(&C->d_rowIndices, (m+1) * sizeof(int));
	
	status = cusparseXcsrgemmNnz(context, 
                      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      m, n, k,
                      descr, A->nnz,
                      A->d_rowIndices, A->d_colIndices, 
                      descr, B->nnz,
                      B->d_rowIndices, B->d_colIndices,
                      descr, C->d_rowIndices,
                      nnzTotalDevHostPtr
                      );

	if(status!=CUSPARSE_STATUS_SUCCESS){
		std::cout<<"Error occured in finding NNZ";
		std::cout<<"\n Status "<<status;
		exit(0);
	}

	if(NULL != nnzTotalDevHostPtr){
		C->nnz = *nnzTotalDevHostPtr;
	}else{
		cudaMemcpy(&C->nnz, C->d_rowIndices + m, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&base, C->d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
		C->nnz -= base;
	}

	allocate_csr_col_val(C, C->nnz);	

	cusparseDcsrgemm(context, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			m,n,k,
			descr, A->nnz,
			A->d_values, A->d_rowIndices, A->d_colIndices,
			descr, B->nnz,
			B->d_values, B->d_rowIndices, B->d_colIndices,
			descr, 
			C->d_values, C->d_rowIndices, C->d_colIndices);

	cudaDeviceSynchronize();
}


int main ( void ){
	/**************
 	* NetMF small *
	**************/

	/* Section 0: Preliminaries */

	/* Settings */
	int window_size = 3;
	int dimension = 128;
	int b = 1;

	/* Load graph */
        log("Reading data from file");
	
	Graph g =  read_graph("../data/test/small_test.csv","edgelist");

	log("Printing adj matrix");
	if(DEBUG)
		print_matrix(g.adj, g.size);	
	
	log("Printing degree matrix");
	if(DEBUG)
		print_matrix(g.degree, g.size);

	/* CUDA housekeeping */
	float num_threads = 128;
	dim3 threads(num_threads);
	dim3 grids((int)ceil((float)g.size/num_threads));

	/* CuSparse housekeeping */
	cusparseHandle_t cusparse_handle;    
	cusparseCreate(&cusparse_handle);	

	cusparseMatDescr_t mat_descr;
	cusparseCreateMatDescr(&mat_descr);
	cusparseSetMatType(mat_descr, 
			CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(mat_descr, 
			CUSPARSE_INDEX_BASE_ZERO);
	int LDA = g.size;


	/* Section 1. Convert graph to sparse */	

	/* Procedure 
	   1. Create dense adjacency and degree matrix on device
	   2. Allocate space for adjacency and degree matrix on device
	   3. Copy dense matrix from host to device
	   4. Create CSR data structure for both matrices
	   5. Compute nnz/row of dense matrix
	   6. Apply Dense2CSR
	 */

	/* Step 1: Create dense adjacency matrix and degree matrixx on device */
	log("Creating dense device array");
	double *adj_device_dense;	
	double *degree_device_dense; 

	/* Step 2: Allocate space for adjacency and degree matrix on device */
	log("Allocating space for dense mat on device");
	cudaMalloc(&adj_device_dense, 
			g.size * g.size * sizeof(double)); 	
	cudaMalloc(&degree_device_dense, 
			g.size * g.size * sizeof(double)); 

	/* Step 3: Copy dense matrix from host to device */
	log("Copying dense matrix from host to device");	
	cudaMemcpy(adj_device_dense, 
			g.adj, 
			g.size * g.size * sizeof(double), 
			cudaMemcpyHostToDevice);	
	cudaMemcpy(degree_device_dense, 
			g.degree, 
			g.size * g.size * sizeof(double), 
			cudaMemcpyHostToDevice);

	/* Step 4: Create CSR struct for both matrices */
	log("Converting dense matrix to CSR format");	
	csr adj_csr,    /* Variable to hold adjacency matrix in CSR format */
	    degree_csr; /* Variable to hold degree matrix in CSR format */

	adj_csr.nnz = 0; /* Initialize number of non zeros in adjacency matrix */
	degree_csr.nnz = 0; /* Initialize number of non zeros in degree matrix */

	/* Step 5: Compute nnz/row of dense matrix */	
	log("Computing nnzPerVector for A");

	cudaMalloc(&adj_csr.d_nnzPerVector, 
			g.size * sizeof(int));
	cusparseDnnz(cusparse_handle, 
			CUSPARSE_DIRECTION_ROW, 
			g.size, g.size, 
			mat_descr, 
			adj_device_dense, LDA, 
			adj_csr.d_nnzPerVector, &adj_csr.nnz);
	adj_csr.h_nnzPerVector = (int *)malloc(g.size * sizeof(int));
	cudaMemcpy(adj_csr.h_nnzPerVector, 
			adj_csr.d_nnzPerVector, 
			g.size * sizeof(int), 
			cudaMemcpyDeviceToHost);
	
	if(DEBUG){
    		printf("Number of nonzero elements in dense adjacency matrix = %i\n", adj_csr.nnz);
    		
		for (int i = 0; i < g.size; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, adj_csr.h_nnzPerVector[i]);
	}

	log("Computing nnzPerVector for D");
	cudaMalloc(&degree_csr.d_nnzPerVector, 
			g.size * sizeof(int));
	cusparseDnnz(cusparse_handle, 
			CUSPARSE_DIRECTION_ROW, 
			g.size, g.size, 
			mat_descr, 
			degree_device_dense, LDA, 
			degree_csr.d_nnzPerVector, &degree_csr.nnz);
	degree_csr.h_nnzPerVector = (int *)malloc(g.size * sizeof(int));
	cudaMemcpy(degree_csr.h_nnzPerVector, 
			degree_csr.d_nnzPerVector, 
			g.size * sizeof(int), 
			cudaMemcpyDeviceToHost);


	if(DEBUG){
    		printf("Number of nonzero elements in dense degree matrix = %i\n", degree_csr.nnz);
    		for (int i = 0; i < g.size; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, degree_csr.h_nnzPerVector[i]);
	}


	/* Step 6: Convert dense matrix to sparse matrices */
	allocate_csr(&adj_csr, adj_csr.nnz, g.size);
	cusparseDdense2csr(cusparse_handle, 
			g.size, g.size, 
			mat_descr,
		        adj_device_dense,	
			LDA, 
			adj_csr.d_nnzPerVector, 
			adj_csr.d_values, adj_csr.d_rowIndices, adj_csr.d_colIndices); 
	device2host(&adj_csr, adj_csr.nnz, g.size);	
	if(DEBUG){
		print_csr(
    			g.size,
    			adj_csr.nnz,
    			adj_csr,
    			"Adjacency matrix");
	}

	allocate_csr(&degree_csr, degree_csr.nnz, g.size);
	cusparseDdense2csr(cusparse_handle, 
			g.size, g.size, 
			mat_descr, 
			degree_device_dense,
			LDA, 
			degree_csr.d_nnzPerVector, 
			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices); 
	device2host(&degree_csr, degree_csr.nnz, g.size);	

	if(DEBUG){
		print_csr(
    			g.size,
    			degree_csr.nnz,
    			degree_csr,
    			"Degree matrix");
	}

	log("Completed conversion of data from dense to sparse");

	/* Section 2: Compute X = D^{-1/2} * A * D^{-1/2} */
	/* Procedure
	   1. Compute D' = D^{-1/2}
	   2. Compute X' = D' * A
	   3. Compute X = X' * D'
	*/
	
	/* Step 1: Compute D' = D^{-1/2} */
	log("Computing normalized D");
	compute_d<<<grids, threads>>>(degree_csr.d_values, degree_csr.nnz);
	
	log("Computed normalized D");
	if(DEBUG){
		device2host(&degree_csr, degree_csr.nnz, g.size);
		print_csr(
			g.size,
			degree_csr.nnz,
			degree_csr,
			"Normalized Degree Matrix");
	}
			

	/* Step 2: Compute X' = D' * A */
	csr X_temp;
	multiply_csr(&adj_csr, &degree_csr, &X_temp, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&X_temp, X_temp.nnz, g.size);
		print_csr(g.size, X_temp.nnz, X_temp, "X' = D' * A");
	}
	


//	
//	// Compute X = X_tempD^{-1/2} second
//	cusparseMatDescr_t X_descr;
//	cusparseCreateMatDescr(&X_descr);
//	cusparseSetMatType(X_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(X_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//	csr X_csr;
//       	X_csr.nnz = 0;
//	nnzTotalDevHostPtr = &X_csr.nnz;
//
//	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
//	cudaMalloc(&X_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//	status = cusparseXcsrgemmNnz(cusparse_handle, 
//				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//				g.size, g.size, g.size,
//				X_temp_descr, X_temp_csr.nnz,
//				X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices, 
//				degree_descr, degree_csr.nnz,
//				degree_csr.d_rowIndices, degree_csr.d_colIndices,
//				X_descr, X_csr.d_rowIndices,
//				nnzTotalDevHostPtr
//				);
//	if (status != CUSPARSE_STATUS_SUCCESS) {
//		std::cout << "nnz calculation failed" << std::endl;
//		std::cout << "status = " << status << std::endl;
//		exit(0);
//	}		
//	//std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;
//
//	if(NULL != nnzTotalDevHostPtr){
//		X_csr.nnz = *nnzTotalDevHostPtr;
//	}else{
//		cudaMemcpy(&X_csr.nnz, X_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(&baseX, X_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
//		X_csr.nnz -= baseX;
//	}
//
//	cudaMalloc(&X_csr.d_colIndices, X_csr.nnz * sizeof(int));
//	cudaMalloc(&X_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//	cudaMalloc(&X_csr.d_values, X_csr.nnz * sizeof(double));
//
//	X_csr.h_colIndices = (int *) malloc(X_csr.nnz * sizeof(int));
//	X_csr.h_rowIndices = (int *) malloc((g.size +1) * sizeof(int));
//	X_csr.h_values = (double *) malloc(X_csr.nnz * sizeof(double));
//
//	cusparseDcsrgemm(cusparse_handle, 
//			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//			g.size, g.size, g.size,
//			X_temp_descr, X_temp_csr.nnz,
//			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices,
//			degree_descr, degree_csr.nnz,
//			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
//			X_descr, 
//			X_csr.d_values, X_csr.d_rowIndices, X_csr.d_colIndices);
//	cudaDeviceSynchronize();
//
//	cuda_status = cudaMemcpy(X_csr.h_values, X_csr.d_values, X_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//	cuda_status = cudaMemcpy(X_csr.h_rowIndices, X_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//	cuda_status = cudaMemcpy(X_csr.h_colIndices, X_csr.d_colIndices, X_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//
//	//std::cout<<cuda_status;
//
//	log("Print X");
////	for(int i=0;i<X_csr.nnz;i++) std::cout<<X_csr.h_values[i]<< " "; std::cout<<"\n";
////	for(int i=0;i<X_csr.nnz;i++) std::cout<<X_csr.h_colIndices[i]<< " "; std::cout<<"\n";
////	for(int i=0;i<g.size + 1;i++) std::cout<<X_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//
//
//	 /* Compute S = sum(X^{0}....X^{window_size}) */
//
//	csr S_csr, temp_csr, temp1_csr, S_temp_csr;
//	S_csr.nnz = X_csr.nnz;
//	temp_csr.nnz = X_csr.nnz;
//
//	cudaMalloc(&S_csr.d_values, S_csr.nnz * sizeof(double));
//	cudaMalloc(&S_csr.d_colIndices, S_csr.nnz * sizeof(int));
//	cudaMalloc(&S_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//
//	cudaMalloc(&temp_csr.d_values, temp_csr.nnz * sizeof(double));
//	cudaMalloc(&temp_csr.d_colIndices, temp_csr.nnz * sizeof(int));
//	cudaMalloc(&temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//
//	log("Copying X to S");
//	cudaMemcpy(S_csr.d_values, X_csr.d_values, S_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
//	cudaMemcpy(S_csr.d_colIndices, X_csr.d_colIndices, S_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
//	cudaMemcpy(S_csr.d_rowIndices, X_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
//
//	log("Copying S to host");
//	S_csr.h_values = (double *) malloc(S_csr.nnz * sizeof(double));
//	S_csr.h_colIndices = (int *) malloc(S_csr.nnz * sizeof(int));
//	S_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(int));
//
//	cudaMemcpy(S_csr.h_values, S_csr.d_values, S_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(S_csr.h_colIndices, S_csr.d_colIndices, S_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(S_csr.h_rowIndices, S_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//
//	//for(int i=0;i<S_csr.nnz;i++) std::cout<<S_csr.h_values[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<S_csr.nnz;i++) std::cout<<S_csr.h_colIndices[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<g.size + 1;i++) std::cout<<S_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//
//	log("Copying X to temp");
//	cudaMemcpy(temp_csr.d_values, X_csr.d_values, temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
//	cudaMemcpy(temp_csr.d_colIndices, X_csr.d_colIndices, temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
//	cudaMemcpy(temp_csr.d_rowIndices, X_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
//
//	log("Copying S to host");
//	temp_csr.h_values = (double *) malloc(temp_csr.nnz * sizeof(double));
//	temp_csr.h_colIndices = (int *) malloc(temp_csr.nnz * sizeof(int));
//	temp_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(int));
//
//	cudaMemcpy(temp_csr.h_values, temp_csr.d_values, temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(temp_csr.h_colIndices, temp_csr.d_colIndices, temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(temp_csr.h_rowIndices, temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//
//	//for(int i=0;i<temp_csr.nnz;i++) std::cout<<temp_csr.h_values[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<temp_csr.nnz;i++) std::cout<<temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<g.size + 1;i++) std::cout<<temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//
//
//	cusparseMatDescr_t S_descr;
//	cusparseCreateMatDescr(&S_descr);
//	cusparseSetMatType(S_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(S_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//	cusparseMatDescr_t S_temp_descr;
//	cusparseCreateMatDescr(&S_temp_descr);
//	cusparseSetMatType(S_temp_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(S_temp_descr, CUSPARSE_INDEX_BASE_ZERO);
//	
//	cusparseMatDescr_t temp_descr;
//	cusparseCreateMatDescr(&temp_descr);
//	cusparseSetMatType(temp_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(temp_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//	cusparseMatDescr_t temp1_descr;
//	cusparseCreateMatDescr(&temp1_descr);
//	cusparseSetMatType(temp1_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(temp1_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//	double alf = 1.00;
//	double beta = 1.00;
//
//	for(int i=2;i<=window_size;i++){
//		std::cout<<"Computing X^"<<i<<std::endl;
//		temp1_csr.nnz = 0;
//		temp1_csr.d_rowIndices = NULL;
//		temp1_csr.d_colIndices = NULL;
//		temp1_csr.d_values = NULL;
//		
//		temp1_csr.h_rowIndices = NULL;
//		temp1_csr.h_colIndices = NULL;
//		temp1_csr.h_values = NULL;
//		
//		S_temp_csr.nnz = 0;
//		S_temp_csr.d_rowIndices = NULL;
//		S_temp_csr.d_colIndices = NULL;
//		S_temp_csr.d_values = NULL;
//		
//		S_temp_csr.h_rowIndices = NULL;
//		S_temp_csr.h_colIndices = NULL;
//		S_temp_csr.h_values = NULL;
//
//		nnzTotalDevHostPtr = &temp1_csr.nnz;
//
//		cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
//		cudaMalloc(&temp1_csr.d_rowIndices, (g.size + 1) * sizeof(double));
//		status = cusparseXcsrgemmNnz(cusparse_handle, 
//					CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//					g.size, g.size, g.size,
//					temp_descr, temp_csr.nnz,
//					temp_csr.d_rowIndices, temp_csr.d_colIndices,
//					X_descr, X_csr.nnz,
//					X_csr.d_rowIndices, X_csr.d_colIndices,
//					temp1_descr, temp1_csr.d_rowIndices,
//					nnzTotalDevHostPtr);
//
//		if(status != 0){
//			std::cout<<"Error encountered"<<std::endl;
//			std::cout<<"Status: "<<status<<std::endl;
//			exit(0);	
//
//		}
//
//
//		if(NULL!=nnzTotalDevHostPtr) temp1_csr.nnz = *nnzTotalDevHostPtr;
//		else{
//			cudaMemcpy(&temp1_csr.nnz, temp1_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
//			cudaMemcpy(&baseX, temp1_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
//			temp1_csr.nnz -= baseX;
//		
//		}
//
//		//std::cout<<"NNZ ="<<temp1_csr.nnz<<std::endl;
//
//
//		cudaMalloc(&temp1_csr.d_values, temp1_csr.nnz * sizeof(double));
//		cudaMalloc(&temp1_csr.d_colIndices, temp1_csr.nnz * sizeof(int));
//
//		cusparseDcsrgemm(cusparse_handle,
//				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//				g.size, g.size, g.size,
//				temp_descr, temp_csr.nnz,
//				temp_csr.d_values, temp_csr.d_rowIndices, temp_csr.d_colIndices,
//				X_descr, X_csr.nnz,
//				X_csr.d_values, X_csr.d_rowIndices, X_csr.d_colIndices,
//				temp1_descr, 
//				temp1_csr.d_values, temp1_csr.d_rowIndices, temp1_csr.d_colIndices);
//	
//		log("Printing intermediate result");					
//		temp1_csr.h_values = (double *) malloc(temp1_csr.nnz * sizeof(double));
//		temp1_csr.h_colIndices = (int *) malloc(temp1_csr.nnz * sizeof(int));
//		temp1_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(int));
//
//		cudaMemcpy(temp1_csr.h_values, temp1_csr.d_values, temp1_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//		cudaMemcpy(temp1_csr.h_colIndices, temp1_csr.d_colIndices, temp1_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(temp1_csr.h_rowIndices, temp1_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//
//	//	for(int i=0;i<temp1_csr.nnz;i++) std::cout<<temp1_csr.h_values[i]<< " "; std::cout<<"\n";
//	//	for(int i=0;i<temp1_csr.nnz;i++) std::cout<<temp1_csr.h_colIndices[i]<< " "; std::cout<<"\n";
//	//	for(int i=0;i<g.size + 1;i++) std::cout<<temp1_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//
//		cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
//
//		S_temp_csr.nnz = 0;
//		nnzTotalDevHostPtr = &S_temp_csr.nnz;
//		cudaMalloc(&S_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//		
//		status = cusparseXcsrgeamNnz(cusparse_handle,
//					g.size, g.size,
//					temp1_descr, temp1_csr.nnz, temp1_csr.d_rowIndices, temp1_csr.d_colIndices,
//					S_descr,S_csr.nnz, S_csr.d_rowIndices, S_csr.d_colIndices,
//					S_temp_descr, S_temp_csr.d_rowIndices, nnzTotalDevHostPtr);	
//
//		if(status != 0){
//			std::cout<<"Error encountered"<<std::endl;
//			std::cout<<"Status: "<<status<<std::endl;
//			exit(0);	
//
//		}
//		if(NULL!=nnzTotalDevHostPtr) S_temp_csr.nnz = *nnzTotalDevHostPtr;
//		else{
//			cudaMemcpy(&S_temp_csr.nnz, S_temp_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
//			cudaMemcpy(&baseX, S_temp_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
//			S_temp_csr.nnz -= baseX;
//		}
//
//		//std::cout<<"SUM-NNZ"<<S_temp_csr.nnz<<std::endl;
//
//		cudaMalloc(&S_temp_csr.d_values, sizeof(double) * S_temp_csr.nnz);
//		cudaMalloc(&S_temp_csr.d_colIndices, sizeof(int) * S_temp_csr.nnz);
//
//		cusparseDcsrgeam(cusparse_handle, g.size, g.size,
//				&alf,
//				temp1_descr, temp1_csr.nnz,
//				temp1_csr.d_values, temp1_csr.d_rowIndices, temp1_csr.d_colIndices,
//				&beta,
//				S_descr, S_csr.nnz,
//				S_csr.d_values, S_csr.d_rowIndices, S_csr.d_colIndices,
//				S_temp_descr,
//				S_temp_csr.d_values, S_temp_csr.d_rowIndices, S_temp_csr.d_colIndices);
//
//		log("Printing intermediate sum");
//		S_temp_csr.h_values = (double *) malloc(sizeof(double) * S_temp_csr.nnz);
//		S_temp_csr.h_rowIndices = (int *) malloc(sizeof(int) * (g.size + 1));
//		S_temp_csr.h_colIndices = (int *) malloc(sizeof(int) * S_temp_csr.nnz);
//
//		cudaMemcpy(S_temp_csr.h_values, S_temp_csr.d_values, S_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//		cudaMemcpy(S_temp_csr.h_rowIndices, S_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(S_temp_csr.h_colIndices, S_temp_csr.d_colIndices, S_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//
//		//for(int i=0;i<S_temp_csr.nnz;i++) std::cout<<S_temp_csr.h_values[i]<< " "; std::cout<<"\n";
//		//for(int i=0;i<S_temp_csr.nnz;i++) std::cout<<S_temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
//		//for(int i=0;i<g.size + 1;i++) std::cout<<S_temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//
//		S_csr.nnz = S_temp_csr.nnz;
//		cudaFree(S_csr.d_values);
//		cudaFree(S_csr.d_rowIndices);
//		//cudaFree(S_csr.d_colIndices);
//		
//		cudaMalloc(&S_csr.d_values, S_csr.nnz * sizeof(double)); 
//		cudaMalloc(&S_csr.d_rowIndices, (g.size + 1) * sizeof(int)); 
//		cudaMalloc(&S_csr.d_colIndices, S_csr.nnz * sizeof(int)); 
//
//		cudaMemcpy(S_csr.d_values, S_temp_csr.d_values, S_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(S_csr.d_rowIndices, S_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(S_csr.d_colIndices, S_temp_csr.d_colIndices, S_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
//
//
//		temp_csr.nnz = temp1_csr.nnz;
//		cudaFree(temp_csr.d_values);
//		cudaFree(temp_csr.d_rowIndices);
//		//cudaFree(temp_csr.d_colIndices);
//
//		cudaMalloc(&temp_csr.d_values, temp_csr.nnz * sizeof(double)); 
//		cudaMalloc(&temp_csr.d_rowIndices, (g.size + 1) * sizeof(int)); 
//		cudaMalloc(&temp_csr.d_colIndices, temp_csr.nnz * sizeof(int)); 
//
//		cudaMemcpy(temp_csr.d_values, temp1_csr.d_values, temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(temp_csr.d_rowIndices, temp1_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(temp_csr.d_colIndices, temp1_csr.d_colIndices, temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
//			
//	}
//
//        /* Compute S = S * (vol / (window_size * b)) */
//
//	transform_s<<<grids, threads>>>(S_csr.d_values, g.volume, window_size, b, S_csr.nnz);
//
//	S_csr.h_values = (double *)malloc(sizeof(double) * S_csr.nnz);
//
//	cudaMemcpy(S_csr.h_values, S_csr.d_values, S_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//
//	log("Transformed S values");
//	//for(int i=0;i<S_csr.nnz;i++) std::cout<<S_csr.h_values[i]<< " "; std::cout<<"\n";
//
//	log("Computing M");
//
//
//        /* Compute M = D^{-1/2} * S * D^{-1/2} */
//
//	// Compute X_temp = D^{-1/2} * S first	
//       	X_temp_csr.nnz = 0;
//	nnzTotalDevHostPtr = &X_temp_csr.nnz;
//
//	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
//	cudaMalloc(&X_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//	status = cusparseXcsrgemmNnz(cusparse_handle, 
//				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//				g.size, g.size, g.size,
//				degree_descr, degree_csr.nnz,
//				degree_csr.d_rowIndices, degree_csr.d_colIndices, 
//				S_descr, S_csr.nnz,
//				S_csr.d_rowIndices, S_csr.d_colIndices,
//				X_temp_descr, X_temp_csr.d_rowIndices,
//				nnzTotalDevHostPtr
//				);
//	if (status != CUSPARSE_STATUS_SUCCESS) {
//		std::cout << "nnz calculation failed" << std::endl;
//		std::cout << "status = " << status << std::endl;
//		exit(0);
//	}		
//	//std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;
//
//	if(NULL != nnzTotalDevHostPtr){
//		X_temp_csr.nnz = *nnzTotalDevHostPtr;
//	}else{
//		cudaMemcpy(&X_temp_csr.nnz, X_temp_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(&baseX, X_temp_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
//		X_temp_csr.nnz -= baseX;
//	}
//
//	cudaMalloc(&X_temp_csr.d_colIndices, X_temp_csr.nnz * sizeof(int));
//	cudaMalloc(&X_temp_csr.d_values, X_temp_csr.nnz * sizeof(double));
//
//	X_temp_csr.h_colIndices = (int *) malloc(X_temp_csr.nnz * sizeof(double));
//	X_temp_csr.h_rowIndices = (int *) malloc((g.size + 1) * sizeof(double));
//	X_temp_csr.h_values = (double *) malloc(X_temp_csr.nnz * sizeof(double));
//
//	cusparseDcsrgemm(cusparse_handle, 
//			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//			g.size, g.size, g.size,
//			degree_descr, degree_csr.nnz,
//			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
//			S_descr, S_csr.nnz,
//			S_csr.d_values, S_csr.d_rowIndices, S_csr.d_colIndices,
//			X_temp_descr, 
//			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices);
//	cudaDeviceSynchronize();
//
//	cuda_status = cudaMemcpy(X_temp_csr.h_values, X_temp_csr.d_values, X_temp_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//	cuda_status = cudaMemcpy(X_temp_csr.h_rowIndices, X_temp_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//	cuda_status = cudaMemcpy(X_temp_csr.h_colIndices, X_temp_csr.d_colIndices, X_temp_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//
//	log("Print X_temp");
//	//for(int i=0;i<X_temp_csr.nnz;i++) std::cout<<X_temp_csr.h_values[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<X_temp_csr.nnz;i++) std::cout<<X_temp_csr.h_colIndices[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<g.size + 1;i++) std::cout<<X_temp_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//	
//	// Compute M = X_temp * D^{-1/2} second
//	cusparseMatDescr_t M_descr;
//	cusparseCreateMatDescr(&M_descr);
//	cusparseSetMatType(M_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(M_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//	csr M_csr;
//       	M_csr.nnz = 0;
//	nnzTotalDevHostPtr = &M_csr.nnz;
//
//	cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
//	cudaMalloc(&M_csr.d_rowIndices, (g.size + 1) * sizeof(int));
//	status = cusparseXcsrgemmNnz(cusparse_handle, 
//				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//				g.size, g.size, g.size,
//				X_temp_descr, X_temp_csr.nnz,
//				X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices, 
//				degree_descr, degree_csr.nnz,
//				degree_csr.d_rowIndices, degree_csr.d_colIndices,
//				M_descr, M_csr.d_rowIndices,
//				nnzTotalDevHostPtr
//				);
//	if (status != CUSPARSE_STATUS_SUCCESS) {
//		std::cout << "nnz calculation failed" << std::endl;
//		std::cout << "status = " << status << std::endl;
//		exit(0);
//	}		
//	//std::cout<<"Value of NNZ: "<<*nnzTotalDevHostPtr<<std::endl;
//
//	if(NULL != nnzTotalDevHostPtr){
//		M_csr.nnz = *nnzTotalDevHostPtr;
//	}else{
//		cudaMemcpy(&M_csr.nnz, M_csr.d_rowIndices + g.size, sizeof(int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(&baseX, M_csr.d_rowIndices, sizeof(int), cudaMemcpyDeviceToHost);
//		M_csr.nnz -= baseX;
//	}
//
//	cudaMalloc(&M_csr.d_colIndices, M_csr.nnz * sizeof(int));
//	cudaMalloc(&M_csr.h_rowIndices, (g.size + 1) * sizeof(int));
//	cudaMalloc(&M_csr.d_values, M_csr.nnz * sizeof(double));
//
//	M_csr.h_colIndices = (int *) malloc(M_csr.nnz * sizeof(double));
//	M_csr.h_rowIndices = (int *) malloc(M_csr.nnz * sizeof(double));
//	M_csr.h_values = (double *) malloc(M_csr.nnz * sizeof(double));
//
//	cusparseDcsrgemm(cusparse_handle, 
//			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
//			g.size, g.size, g.size,
//			X_temp_descr, X_temp_csr.nnz,
//			X_temp_csr.d_values, X_temp_csr.d_rowIndices, X_temp_csr.d_colIndices,
//			degree_descr, degree_csr.nnz,
//			degree_csr.d_values, degree_csr.d_rowIndices, degree_csr.d_colIndices,
//			M_descr, 
//			M_csr.d_values, M_csr.d_rowIndices, M_csr.d_colIndices);
//	cudaDeviceSynchronize();
//
//	cuda_status = cudaMemcpy(M_csr.h_values, M_csr.d_values, M_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//	cuda_status = cudaMemcpy(M_csr.h_rowIndices, M_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//	cuda_status = cudaMemcpy(M_csr.h_colIndices, M_csr.d_colIndices, M_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//
//	//std::cout<<cuda_status;
//
//	log("Print M");
//	//for(int i=0;i<M_csr.nnz;i++) std::cout<<M_csr.h_values[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<M_csr.nnz;i++) std::cout<<M_csr.h_colIndices[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<g.size + 1;i++) std::cout<<M_csr.h_rowIndices[i]<< " "; std::cout<<"\n";
//
//	log("Transforming M");
//
//	prune_m<<<grids,threads>>>(M_csr.d_values, M_csr.nnz);
//	cudaMemcpy(M_csr.h_values, M_csr.d_values, M_csr.nnz * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(M_csr.h_rowIndices, M_csr.d_rowIndices, (g.size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(M_csr.h_colIndices, M_csr.d_colIndices, M_csr.nnz * sizeof(int), cudaMemcpyDeviceToHost);
//	//for(int i=0;i<M_csr.nnz;i++) std::cout<<M_csr.h_values[i]<< " "; std::cout<<"\n";
//	cudaDeviceSynchronize();
//
//
//
//	double threshold = 1.00;
//	log("Setting threshold");
//	//std::cout<<"Threshold:"<<threshold;
//	csr filtered_M;
//
//	cudaMalloc(&filtered_M.d_rowIndices, sizeof(int) * (g.size + 1));
//	size_t lworkInBytes = 0;
//	char *d_work=NULL;
//	cusparseDpruneCsr2csr_bufferSizeExt(cusparse_handle,
//        					g.size,g.size,
//        					M_csr.nnz,M_descr,
//        					M_csr.d_values,M_csr.d_rowIndices,M_csr.d_colIndices,
//        					&threshold,
//        					M_descr,
//        					filtered_M.d_values, filtered_M.d_rowIndices, filtered_M.d_colIndices,
//        					&lworkInBytes);	
//
//    	//printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
//	cudaMalloc((void**)&d_work, lworkInBytes);
//
//	cusparseDpruneCsr2csrNnz(cusparse_handle,
//					g.size,g.size,
//        				M_csr.nnz,M_descr,
//  					M_csr.d_values,M_csr.d_rowIndices,M_csr.d_colIndices,
//        				&threshold,
//        				M_descr,
//        				filtered_M.d_rowIndices, &filtered_M.nnz, /* host */
//        				d_work);
//
//	//printf("nnzC = %d\n", filtered_M.nnz);
//    	if (0 == filtered_M.nnz ){
//        	printf("C is empty \n");
//        	return 0;
//    	}
//
//
//	cudaMalloc(&filtered_M.d_colIndices, sizeof(int) * filtered_M.nnz);
//	cudaMalloc(&filtered_M.d_values, sizeof(double) * filtered_M.nnz);
//
//	cusparseDpruneCsr2csr(cusparse_handle,
//   				g.size,g.size,
//        			M_csr.nnz, M_descr,
//        			M_csr.d_values, M_csr.d_rowIndices, M_csr.d_colIndices,
//        			&threshold,
//        			M_descr,
//        			filtered_M.d_values,filtered_M.d_rowIndices, filtered_M.d_colIndices,
//        			d_work);
//
//
//	log("Printing pruned M");
//	filtered_M.h_values = (double *) malloc(sizeof(double) * filtered_M.nnz);
//	filtered_M.h_colIndices = (int *) malloc(sizeof(int) * filtered_M.nnz);
//	filtered_M.h_rowIndices = (int *) malloc(sizeof(int) * (g.size + 1));
//
//	cudaMemcpy(filtered_M.h_values, filtered_M.d_values, sizeof(double) * filtered_M.nnz, cudaMemcpyDeviceToHost);
//	cudaMemcpy(filtered_M.h_colIndices, filtered_M.d_colIndices, sizeof(int) * filtered_M.nnz, cudaMemcpyDeviceToHost);
//	cudaMemcpy(filtered_M.h_rowIndices, filtered_M.d_rowIndices, sizeof(int) * (g.size + 1), cudaMemcpyDeviceToHost);
//	
//	//for(int i=0;i<filtered_M.nnz;i++) std::cout<<filtered_M.h_values[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<filtered_M.nnz;i++) std::cout<<filtered_M.h_colIndices[i]<< " "; std::cout<<"\n";
//	//for(int i=0;i<g.size + 1;i++) std::cout<<filtered_M.h_rowIndices[i]<< " "; std::cout<<"\n";
//	
//	log("Printing log of M");
//	transform_m<<<grids,threads>>>(filtered_M.d_values, filtered_M.nnz);
//
//	cudaMemcpy(filtered_M.h_values, filtered_M.d_values, sizeof(double) * filtered_M.nnz, cudaMemcpyDeviceToHost);
//	cudaMemcpy(filtered_M.h_colIndices, filtered_M.d_colIndices, sizeof(int) * filtered_M.nnz, cudaMemcpyDeviceToHost);
//	cudaMemcpy(filtered_M.h_rowIndices, filtered_M.d_rowIndices, sizeof(int) * (g.size + 1), cudaMemcpyDeviceToHost);
//	//for(int i=0;i<filtered_M.nnz;i++) std::cout<<filtered_M.h_values[i]<< " "; std::cout<<"\n";
//
//	char whichS = 'L';
//	char whichV = 'L';
//
//	MKL_INT pm[128];
//	mkl_sparse_ee_init(pm);
//	//pm[1] = 100;
//	//pm[2] = 2;
//	//pm[4] = 60;
//
//	MKL_INT mkl_rows = g.size;
//	MKL_INT mkl_cols = g.size;
//
//
//	MKL_INT rows_start[mkl_rows];
//	MKL_INT rows_end[mkl_rows];
//
//	for(int i=0;i<mkl_rows;i++){
//		rows_start[i] = filtered_M.h_rowIndices[i];
//		rows_end[i] = filtered_M.h_colIndices[i];
//	}
//
//	
//	MKL_INT mkl_col_idx[filtered_M.nnz];
//	for(int i=0;i<filtered_M.nnz;i++)
//		mkl_col_idx[i] = filtered_M.h_colIndices[i];
//
//
//	sparse_matrix_t M_mkl;
//	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
//
//	mkl_sparse_d_create_csr(&M_mkl, indexing,
//					mkl_rows, mkl_cols,
//					rows_start, rows_end,
//					mkl_col_idx, filtered_M.h_values);
//
//	log("Created MKL sparse");
//
//	matrix_descr mkl_descrM;
//	mkl_descrM.type = SPARSE_MATRIX_TYPE_GENERAL;	
//	mkl_descrM.mode = SPARSE_FILL_MODE_UPPER;
//	mkl_descrM.diag = SPARSE_DIAG_NON_UNIT;
//
//	MKL_INT k0 = dimension;
//	MKL_INT k;
//
//	double *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;
//
//	E_mkl = (double *)mkl_malloc(k0 * sizeof(double), 128);
//	K_L_mkl = (double *)mkl_malloc( k0*mkl_rows*sizeof( double), 128 );
//        K_R_mkl = (double *)mkl_malloc( k0*mkl_cols*sizeof( double), 128 );
//        res_mkl = (double *)mkl_malloc( k0*sizeof( double), 128 );
//
//	memset(E_mkl, 0 , k0);
//	memset(K_L_mkl, 0 , k0);
//	memset(K_R_mkl, 0 , k0);
//	memset(res_mkl, 0 , k0);
//
//	int mkl_status = 0;
//
//	log("Computing SVD via MKL");
//	mkl_status = mkl_sparse_d_svd(&whichS, &whichV, pm,
//			M_mkl, mkl_descrM,
//			k0, &k,
//			E_mkl,
//			K_L_mkl,
//			K_R_mkl,
//			res_mkl);
//	log("Computed SVD via MKL");
//
//	std::cout<<"Number of eigenvalues found: "<<k<<std::endl;
//	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";
//
//
}
