#include<stdlib.h>
#include<iostream>
#include<time.h>
#include<chrono>
#include<algorithm>
#include<numeric>
#include<math.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include <cusparse_v2.h> 

#include "../utils/graph.h"
#include "../utils/io.h"

#include<mkl.h>
#include<mkl_solvers_ee.h>
#include<mkl_spblas.h>

#define DEBUG true
#define VERBOSE false


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
__global__ void preprocess_laplacian(double* adj, double *degree, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	// Remove self loops
	// If deg(v) = 0 -> deg(0) = 1
		
	if(degree[id*size + id] == 0){
			degree[id*size + id] = 1.00;
			adj[id*size + id] = 1.00;	
	}else{
			adj[id*size + id] = 0.0;
	}	
}
__global__ void compute_d(double* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	//id = id * size + id;
	
	/* Make assumption here that graph is  	*/
	/* connected and every node has degree 	*/
        /* atleast 1. 		       		*/
	
	if(deg[id] == -1)
		deg[id] = 0;
	else	
		deg[id] = 1 / sqrt(deg[id]); 
}

__global__ void compute_s(float* S, float* X, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] += X[id]; 
}

__global__ void transform_si(double* S, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = sqrt(S[id]); 
}

__global__ void transform_s(double* S, float val, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	double mem = S[id];
	
	S[id] = mem * val; 
}

__global__ void prune_m(double* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;

	if(M[id] < 1)
		M[id] = 1;	
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

void free_csr(csr *A){
	free(A->h_values);
	free(A->h_rowIndices);
	free(A->h_colIndices);

	cudaFree(A->d_values);
	cudaFree(A->d_rowIndices);
	cudaFree(A->d_colIndices);
}

void copy_csr(csr *from_mat, csr *to_mat, int num_rows){
	to_mat->nnz = from_mat->nnz;

	/* Copy host variables */
	memcpy(to_mat->h_values, from_mat->h_values, to_mat->nnz * sizeof(double));
	memcpy(to_mat->h_colIndices, from_mat->h_colIndices, to_mat->nnz * sizeof(int));
	memcpy(to_mat->h_rowIndices, from_mat->h_rowIndices, (num_rows + 1) * sizeof(int));


	/* Copy device variables */
	cudaMemcpy(to_mat->d_values, from_mat->d_values, to_mat->nnz * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(to_mat->d_colIndices, from_mat->d_colIndices, to_mat->nnz * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(to_mat->d_rowIndices, from_mat->d_rowIndices, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

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

void add_csr(csr *A, csr *B, csr *C, int m, int n, cusparseHandle_t context,cusparseMatDescr_t descr){

	cusparseStatus_t status;

	int base;
	int *nnzTotalDevHostPtr;
	C->nnz = 0;
	nnzTotalDevHostPtr = &C->nnz;

	double alf = 1.0;
	double bet = 1.0;

	cusparseSetPointerMode(context, CUSPARSE_POINTER_MODE_HOST);
	allocate_csr_row(C, m);
		
	status = cusparseXcsrgeamNnz(context,
				m, n,
				descr, A->nnz, A->d_rowIndices, A->d_colIndices,
				descr, B->nnz, B->d_rowIndices, B->d_colIndices,
				descr, C->d_rowIndices, nnzTotalDevHostPtr);	

	if(status != CUSPARSE_STATUS_SUCCESS){
		std::cout<<"Error encountered"<<std::endl;
		std::cout<<"Status: "<<status<<std::endl;
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

	cusparseDcsrgeam(context, m, n,
			&alf,
			descr, A->nnz,
			A->d_values, A->d_rowIndices, A->d_colIndices,
			&bet,
			descr, B->nnz,
			B->d_values, B->d_rowIndices, B->d_colIndices,
			descr,
			C->d_values, C->d_rowIndices, C->d_colIndices);

}

void multiply_csr(csr *A, csr *B, csr *C, int m, int n, int k, cusparseHandle_t context,cusparseMatDescr_t descr ){

	cusparseStatus_t status;

	int base;
	int *nnzTotalDevHostPtr;
	C->nnz = 0;
	nnzTotalDevHostPtr = &C->nnz;

	cusparseSetPointerMode(context, CUSPARSE_POINTER_MODE_HOST);

	allocate_csr_row(C, m);
	//cudaMalloc(&C->d_rowIndices, (m+1) * sizeof(int));
	
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
	
	//Graph g =  read_graph("../data/test/small_test.csv","edgelist");
	Graph g =  read_graph("../data/ppi/ppi.edgelist","edgelist");
	
	if(DEBUG){
		if(VERBOSE){
			log("Printing adj matrix");
			print_matrix(g.adj, g.size);
		}
	}	
	
	log("Printing degree matrix");
	if(DEBUG){
		if(VERBOSE){
			print_matrix(g.degree, g.size);
		}
	}


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

	/* CuBlas Housekeeping */
	log("Creating cuBlas variables");
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	
	/* Section 1. Convert graph to sparse */	

	/* Procedure 
	   1. Create dense adjacency and degree matrix on device
	   2. Allocate space for adjacency and degree matrix on device
	   3. Copy dense matrix from host to device
	   4. Preprocess degree and adjacency matrix for laplacian computation
	   5. Create CSR data structure for both matrices
	   6. Compute nnz/row of dense matrix
	   7. Apply Dense2CSR
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

	/*Step 4: Compute volume and preprocess degree */
	preprocess_laplacian<<<grids,threads>>>(adj_device_dense, degree_device_dense, g.size);

	/* Step 5: Create CSR struct for both matrices */
	log("Converting dense matrix to CSR format");	
	csr adj_csr,    /* Variable to hold adjacency matrix in CSR format */
	    degree_csr; /* Variable to hold degree matrix in CSR format */

	adj_csr.nnz = 0; /* Initialize number of non zeros in adjacency matrix */
	degree_csr.nnz = 0; /* Initialize number of non zeros in degree matrix */

	/* Step 6: Compute nnz/row of dense matrix */	
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
    		
		if(VERBOSE)
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
    		if(VERBOSE)
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
	if(VERBOSE){
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

	if(DEBUG){
		device2host(&degree_csr, degree_csr.nnz, g.size);	
		std::sort(degree_csr.h_values, degree_csr.h_values + degree_csr.nnz);
		double sum = 0;
		for(int i=0;i<degree_csr.nnz;i++)
			sum+=degree_csr.h_values[i];
		//int sum = std::accumulate(degree_csr.h_values, degree_csr.h_values + degree_csr.nnz);
		std::cout<<"Sum: "<<sum<<std::endl;
		if(VERBOSE){
			print_csr(
    				g.size,
    				degree_csr.nnz,
    				degree_csr,
    				"Degree matrix");
		}
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
		std::sort(degree_csr.h_values, degree_csr.h_values + degree_csr.nnz);
		if(VERBOSE){
			print_csr(
				g.size,
				degree_csr.nnz,
				degree_csr,
				"Normalized Degree Matrix");
		}
	}
			

	/* Step 2: Compute X' = D' * A */
	log("Computing X' = D' * A");
	csr X_temp;
	multiply_csr(&degree_csr, &adj_csr, &X_temp, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&X_temp, X_temp.nnz, g.size);
		if(VERBOSE)
		{	print_csr(g.size, X_temp.nnz, X_temp, "X' = D' * A");}
	}
	

	/* Step 3: Compute X = X' * D */
	log("Computing X = X' * D");
	csr X;
	multiply_csr(&X_temp, &degree_csr, &X, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&X, X.nnz, g.size);
		std::cout<<"Min of X: "<<*std::min_element(X.h_values, X.h_values + X.nnz)<<std::endl;
		std::cout<<"Max of X: "<<*std::max_element(X.h_values, X.h_values + X.nnz)<<std::endl;
		int x_nz = 0;
		int x_nnz = 0;
		for(int i=0;i<X.nnz;i++){
			if(X.h_values[i] == 0)
				x_nz +=1;
			else
				x_nnz +=1;
		}	
		std::cout<<"# zeros: "<<x_nz<<std::endl;
		std::cout<<"# nonzeros: "<<x_nnz<<std::endl;
		if(VERBOSE)
			{print_csr(g.size, X.nnz, X, "X = X' * A");}
	}

	/* Section 3: Compute S = sum(X^{0}....X^{window_size}) */
	/* Procedure
	  1. Copy X to S
	  2. Copy X to temp
	  3. temp' = temp * X
	  4. S' = S + temp'  
	  5. temp = temp'
	  6. S = S'
	*/
	
	/* Step 0: Declare all variables */
	csr S;
	csr temp;
	csr temp_;
	csr S_;
	
	/* Step 1: Copy X to S */
	log("Copying X to S");
	allocate_csr(&S, X.nnz, g.size);
	copy_csr(&X, &S, g.size);

	/* Step 2: Copy X to temp */
	log("Copying X to temp");
	allocate_csr(&temp, X.nnz, g.size);
	copy_csr(&X, &temp, g.size);

	for(int i=2;i<=window_size;i++){
		/* Step 3: temp' = temp * X */
		log("Computing temp' = temp * X");
		multiply_csr(&temp, &X, &temp_, g.size, g.size, g.size, cusparse_handle, mat_descr);


		/* Step 4: S = S + temp */
		log("Computing S = S + temp");
		add_csr(&S, &temp_, &S_, g.size, g.size, cusparse_handle, mat_descr);

		/* Step 5: temp = temp' */
		log("Copying temp' to temp");
		free_csr(&temp);
		allocate_csr(&temp, temp_.nnz, g.size);
		copy_csr(&temp_, &temp, g.size);
		free_csr(&temp_);

		/* Step 6: S = S' */
		log("Copying S' to S");
		free_csr(&S);
		allocate_csr(&S, S_.nnz, g.size);
		copy_csr(&S_, &S, g.size);
		free_csr(&S_);
		
		if(DEBUG){
			device2host(&temp, temp.nnz, g.size);
			device2host(&temp, temp.nnz, g.size);
			//std::sort(temp_.h_values, temp_.h_values + temp_.nnz);
			std::cout<<"Min: "<<*std::min_element(temp.h_values, temp.h_values + temp.nnz)<<std::endl;
			std::cout<<"Max: "<<*std::max_element(temp.h_values, temp.h_values + temp.nnz)<<std::endl;
			device2host(&S, S.nnz, g.size);
			device2host(&S, S.nnz, g.size);
			//std::sort(temp_.h_values, temp_.h_values + temp_.nnz);
			std::cout<<"Min: "<<*std::min_element(S.h_values, S.h_values + S.nnz)<<std::endl;
			std::cout<<"Max: "<<*std::max_element(S.h_values, S.h_values + S.nnz)<<std::endl;
		}
	}

	if(DEBUG){
		device2host(&S, S.nnz, g.size);
		if(VERBOSE){
			print_csr(g.size,
					S.nnz,
					S,
					"Objective matrix");
		}
	}

        /* Section 4: Compute S = S * (vol / (window_size * b)) */

	/* Procedure 
	   1. Compute val = vol / (window_size * b)
	   2. Compute S[i] = S[i] / val
	*/

	log("Applying Transformation on S");
	/* Step 1: Compute val = vol / (window_size * b) */
	const double val = ((double) g.volume)/(((double) window_size) * ((double) b));

	if(DEBUG){
		std::cout<<"Mult value"<<val<<std::endl;
	}

	/* Step 2: Compute S[i] = S[i] * val */

		
	cublasDscal(cublas_handle, S.nnz,
                    &val,
                    S.d_values, 1);
	cudaDeviceSynchronize();
	
	if(DEBUG){
		device2host(&S, S.nnz, g.size);
	
		std::cout<<"Min of S: "<<*std::min_element(S.h_values, S.h_values + S.nnz);
		std::cout<<"Max of S: "<<*std::max_element(S.h_values, S.h_values + S.nnz);

		if(VERBOSE){
			print_csr(g.size,
					S.nnz,
					S,
					"Transformed Objective Matrix");
		}
	}

	log("Computing M");


        /* Section 5: Compute M = D^{-1/2} * S * D^{-1/2} */
	/* Procedure
	   1. Compute M' = D' * S
	   2. Compute M = M' * D'
	*/

	/* Step 1: Compute M' = D' * S */
	log("Computing M' = D' * S");
	csr M_;
	multiply_csr(&degree_csr, &S, &M_, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&M_, M_.nnz, g.size);
		if(VERBOSE)
		{	print_csr(g.size, M_.nnz, M_, "M' = D' * M");}
	}
	

	/* Step 3: Compute X = X' * D' */
	log("Computing M = M' * D");
	csr M;
	multiply_csr(&M_, &degree_csr, &M, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&M, M.nnz, g.size);
		if(VERBOSE)
			{print_csr(g.size, M.nnz, M, "M = M' * D");}
	}

	/* Section 6: Compute M'' = log(max(M,1)) */
	
	/* Procedure 
	   1. Remove all negative elements from M
	   2. Compute buffer size
	   3. Comput number of non-zeros
	   4. Apply threshold & prune
	   5. Compute log M
	*/

	/* Step 1: Removing all negative elements */
	log("Pruning M");

        if(DEBUG){
                device2host(&M, M.nnz, g.size);

                int num_ones = 0;
                int num_l = 0;
                int num_g = 0;

                for(int i=0;i<M.nnz;i++){
                        if(M.h_values[i] < 1){
                                num_l++;
                        }else if(M.h_values[i] > 1){
                                num_g++;
                        }else{
                                num_ones++;
                        }
                }

                std::cout<<"1s: "<<num_ones<<std::endl;
                std::cout<<"<1s: "<<num_l<<std::endl;
                std::cout<<">1s: "<<num_g<<std::endl;

                if(VERBOSE)
                        {print_csr(g.size, M.nnz, M, "M = M' * D");}
        }

	//prune_m<<<grids,threads>>>(M.d_values, M.nnz);
	//cudaDeviceSynchronize();
	//log("Prunied M");

	/* TODO: Move this to GPU DEBUG THIS ASAP */
        device2host(&M, M.nnz, g.size);
        for(int i=0;i<M.nnz;i++){
        	if(M.h_values[i] < 1){
			M.h_values[i] = 1;
                }
		M.h_values[i] = log(M.h_values[i]);
	}

	host2device(&M, M.nnz, g.size);


        if(DEBUG){
               // device2host(&M, M.nnz, g.size);

                int num_ones = 0;
                int num_l = 0;
                int num_g = 0;

                for(int i=0;i<M.nnz;i++){
                        if(M.h_values[i] < 0){
                                num_l++;
                        }else if(M.h_values[i] > 0){
                                num_g++;
                        }else{
                                num_ones++;
                        }
                }

                std::cout<<"1s: "<<num_ones<<std::endl;
                std::cout<<"<1s: "<<num_l<<std::endl;
                std::cout<<">1s: "<<num_g<<std::endl;

                if(VERBOSE)
                        {print_csr(g.size, M.nnz, M, "M = M' * D");}
        }

	
	/* Step 2: Compute buffersize */
	log("Computing buffer size for pruning");
	double threshold = 0.00;

	csr M_cap;
	allocate_csr_row(&M_cap, g.size);
	size_t lworkInBytes = 0;
	char *d_work=NULL;
	cusparseDpruneCsr2csr_bufferSizeExt(cusparse_handle,
        					g.size,g.size,
        					M.nnz, mat_descr,
        					M.d_values,M.d_rowIndices,M.d_colIndices,
        					&threshold,
        					mat_descr,
						M_cap.d_values, M_cap.d_rowIndices, M_cap.d_colIndices,
        					&lworkInBytes);	
	cudaMalloc((void**)&d_work, lworkInBytes);

	if(DEBUG){
    		printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
	}

	/* Step 3: Compute NNZ */
	log("Computing Number of non zeros for pruned matrix");
	cusparseDpruneCsr2csrNnz(cusparse_handle,
					g.size,g.size,
        				M.nnz,mat_descr,
  					M.d_values,M.d_rowIndices,M.d_colIndices,
        				&threshold,
        				mat_descr,
        				M_cap.d_rowIndices, &M_cap.nnz, /* host */
        				d_work);

    	if (0 == M_cap.nnz ){
        	printf("C is empty \n");
        	return 0;
    	}

	if(DEBUG){
		printf("nnzC = %d\n", M_cap.nnz);
	}


	/* Step 4: Convert CSR matrix to CSR matrix */
	
	allocate_csr_col_val(&M_cap, M_cap.nnz);
	cusparseDpruneCsr2csr(cusparse_handle,
   				g.size,g.size,
        			M.nnz, mat_descr,
        			M.d_values, M.d_rowIndices, M.d_colIndices,
        			&threshold,
        			mat_descr,
        			M_cap.d_values,M_cap.d_rowIndices, M_cap.d_colIndices,
        			d_work);


	device2host(&M_cap, M_cap.nnz, g.size);
	std::cout<<"NNZ in M_cap: "<<M_cap.nnz<<std::endl;
	if(VERBOSE){
		print_csr(g.size,
				M_cap.nnz,
				M_cap, 
				"M cap"
			 );
	}
			
	/* Section 7: Compute SVD of objective matrix */	

	char whichS = 'L';
	char whichV = 'L';

	MKL_INT pm[128];
	mkl_sparse_ee_init(pm);
	//pm[1] = 100;
	//pm[2] = 2;
	//pm[4] = 60;

	MKL_INT mkl_rows = g.size;
	MKL_INT mkl_cols = g.size;


	MKL_INT rows_start[mkl_rows];
	MKL_INT rows_end[mkl_rows];

	for(int i=0;i<mkl_rows;i++){
		rows_start[i] = M_cap.h_rowIndices[i];
		rows_end[i] = M_cap.h_rowIndices[i+1];
	}

	
	MKL_INT mkl_col_idx[M_cap.nnz];
	for(int i=0;i<M_cap.nnz;i++)
		mkl_col_idx[i] = M_cap.h_colIndices[i];


	sparse_matrix_t M_mkl;
	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

	mkl_sparse_d_create_csr(&M_mkl, indexing,
					mkl_rows, mkl_cols,
					rows_start, rows_end,
					mkl_col_idx, M_cap.h_values);

	log("Created MKL sparse");

	matrix_descr mkl_descrM;
	mkl_descrM.type = SPARSE_MATRIX_TYPE_GENERAL;	
	mkl_descrM.mode = SPARSE_FILL_MODE_UPPER;
	mkl_descrM.diag = SPARSE_DIAG_NON_UNIT;

	MKL_INT k0 = dimension;
	MKL_INT k;

	double *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;

	E_mkl = (double *)mkl_malloc(k0 * sizeof(double), 128);
	K_L_mkl = (double *)mkl_malloc( k0*mkl_rows*sizeof( double), 128 );
        K_R_mkl = (double *)mkl_malloc( k0*mkl_cols*sizeof( double), 128 );
        res_mkl = (double *)mkl_malloc( k0*sizeof( double), 128 );

	memset(E_mkl, 0 , k0);
	memset(K_L_mkl, 0 , k0);
	memset(K_R_mkl, 0 , k0);
	memset(res_mkl, 0 , k0);

	int mkl_status = 0;

	log("Computing SVD via MKL");
	mkl_status = mkl_sparse_d_svd(&whichS, &whichV, pm,
			M_mkl, mkl_descrM,
			k0, &k,
			E_mkl,
			K_L_mkl,
			K_R_mkl,
			res_mkl);
	log("Computed SVD via MKL");

	std::cout<<"Number of singular found: "<<k<<std::endl;
	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";

	double *U_device, *Si_device;
	double *U_host, *Si_host;
	double *E_device, *E_host;

	cudaMalloc(&U_device, g.size * dimension * sizeof(double));
	cudaMalloc(&E_device, g.size * dimension * sizeof(double));
	cudaMalloc(&Si_device, dimension * sizeof(double));

	U_host = (double *) malloc(g.size * dimension * sizeof(double));
	E_host = (double *) malloc(g.size * dimension * sizeof(double));
	Si_host = (double *) malloc(dimension * sizeof(double));

	// Convert to column major order
	//U_host = (double *) malloc(g.size * dimension * sizeof(double));
	//for(int i=0;i<g.size;i++){
	//	for(int j=0;j<dimension;j++){
	//		U_host[j*g.size+i] = K_L_mkl[i*g.size + j];
	//	}
	//}

	cudaMemcpy(U_device, K_L_mkl, g.size * dimension * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Si_device, E_mkl, dimension * sizeof(double), cudaMemcpyHostToDevice);

	transform_si<<<grids,threads>>>(Si_device, dimension);

	cudaMemcpy(Si_host, Si_device, dimension * sizeof(double), cudaMemcpyDeviceToHost);

//	for(int i=0;i<dimension;i++){
//		std::cout<<Si_host[i]<<" ";	
//	}

//	std::cout<<"\nSing vect"<<std::endl;
//	for(int j=0;j<g.size;j++){
//		for(int i=0;i<dimension;i++){
//			std::cout<<K_L_mkl[i*dimension + j]<<" ";
//		}
//		std::cout<<"\n";
//	}
	

	std::cout<<"\n";
	cublasDdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, dimension,
		U_device, g.size, 
		Si_device, 1.0,
		E_device, g.size);

	cudaMemcpy(E_host, E_device, g.size * dimension * sizeof(double), cudaMemcpyDeviceToHost);

//	for(int i=0;i<g.size;i++){
//		for(int j=0;j<dimension;j++){
//			std::cout<<E_host[j*g.size + i]<<" ";
//		}
//		std::cout<<"\n";
//	}
	write_embeddings("ppi.emb",E_host, g.size, dimension);
}
