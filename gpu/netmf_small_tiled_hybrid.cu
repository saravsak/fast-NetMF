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
		
	if(degree[id] == 0){
			degree[id] = 1.00;
			adj[id*size + id] = 1.00;	
	}else{
			adj[id * size + id] = 0.0;
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

	if(id >= size * size) return;

	double mem = S[id];
	
	S[id] = mem * val; 
}

__global__ void prune_m(double* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;

	if(M[id] <= 1)
		M[id] = 0;
	else
		M[id] = log(M[id]);	
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

int main (int argc, char *argv[] ){
	/**************
 	* NetMF small *
	**************/
	/* Argument order 
	1. Dataset name
	2. Window Size
	3. Dimension
	4. B
	5. Input
	6. Output
	7. Mapping file
	*/
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
        Clock::time_point begin, end;
        Clock::time_point overall_begin, overall_end;
	info profile; 
	profile.dataset = argv[1];
	profile.algo = "small-dense";
	/* Section 0: Preliminaries */

	/* Settings */
	int window_size = std::atoi(argv[2]);
	int dimension = std::atoi(argv[3]);
	int b = std::atoi(argv[4]);

	profile.window_size = window_size;
	profile.dimension = dimension;

	/* Load graph */
        log("Reading data from file");
	
	//Graph g =  read_graph("../data/test/small_test.csv","edgelist");
	Graph g =  read_graph(argv[5],"edgelist", argv[7]);
	begin = Clock::now(); 
	//Graph g =  read_graph("../../nrl-data/wikipedia.edgelist","edgelist");
	end = Clock::now();

	profile.iptime = std::chrono::duration_cast<milliseconds>(end - begin);

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
	begin = Clock::now();
	float num_threads = 128;
	dim3 threads(num_threads);
	dim3 grids((int)ceil((float)(g.size*g.size)/num_threads));

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

	end = Clock::now();
	profile.init = std::chrono::duration_cast<milliseconds>(end - begin);


	/* Section 1. Move data to device */	

	/* Procedure 
	   1. Create dense adjacency and degree matrix on device
	   2. Allocate space for adjacency and degree matrix on device
	   3. Copy dense matrix from host to device
	   4. Preprocess degree and adjacency matrix for laplacian computation
	   5. Create CSR data structure for both matrices
	   6. Compute nnz/row of dense matrix
	   7. Apply Dense2CSR
	 */
	
	begin = Clock::now();
	/* Step 1: Create dense adjacency matrix and degree matrixx on device */
	log("Creating dense device array");
	double *adj_device_dense;	
	double *degree_device_dense; 
	//double *adj_host_dense;	
	//double *degree_host_dense; 

	/* Step 2a: Allocate space for adjacency and degree matrix on device*/
	log("Allocating space for degree and adjacency mat on device");
	cudaMalloc(&adj_device_dense, 
			g.size * g.size * sizeof(double)); 	
	cudaMalloc(&degree_device_dense, 
			g.size * sizeof(double)); 

	/* Step 2a: Allocate space for adjacency and degree matrix on host */
	log("Allocating space for degree and adjacency matrix on host");
	//adj_host_dense = (double *) malloc(g.size * g.size * sizeof(double));
	//degree_host_dense = (double *) malloc(g.size * sizeof(double));

	/* Step 3: Copy dense matrix from host to device */
	log("Copying dense matrix from host to device");	
	cudaMemcpy(adj_device_dense, 
			g.adj, 
			g.size * g.size * sizeof(double), 
			cudaMemcpyHostToDevice);	
	cudaMemcpy(degree_device_dense, 
			g.degree1D, 
			g.size * sizeof(double), 
			cudaMemcpyHostToDevice);

	/*REMOVE*/
	std::sort(g.degree1D,g.degree1D + g.size);

	/*Step 4: Compute volume and preprocess degree */
	preprocess_laplacian<<<grids,threads>>>(adj_device_dense, degree_device_dense, g.size);
	end = Clock::now();
	profile.gpuio = std::chrono::duration_cast<milliseconds>(end - begin);

	begin = Clock::now();
	log("Moved data from host to device");

	/* Section 2: Compute X = D^{-1/2} * A * D^{-1/2} */
	/* Procedure
	   1. Compute D' = D^{-1/2}
	   2. Compute X' = D' * A
	   3. Compute X = X' * D'
	*/
	
	/* Step 1: Compute D' = D^{-1/2} */
	log("Computing normalized D");
	compute_d<<<grids, threads>>>(degree_device_dense, g.size);
	cudaDeviceSynchronize();
	end = Clock::now();
	profile.compute_d = std::chrono::duration_cast<milliseconds>(end - begin);

	log("Computed normalized D");
	overall_begin = Clock::now();
	begin = Clock::now();
	/* Step 2: Compute X' = D' * A */
	log("Computing X' = D' * A");
	double *X_temp_device;
	//double *X_temp_host;

	cudaMalloc(&X_temp_device, g.size * g.size * sizeof(double));
	cudaMemset(X_temp_device, 0, g.size * g.size);

	cublasDdgmm(cublas_handle, CUBLAS_SIDE_LEFT,
		g.size, g.size,
		adj_device_dense, g.size, 
		degree_device_dense, 1,
		X_temp_device, g.size);
	cudaDeviceSynchronize();
	

//	/* Step 3: Compute X = X' * D */
	log("Computing X = X' * D");
	double *X_device;
	//double *X_host;
	cudaMalloc(&X_device, g.size * g.size * sizeof(double));
	cudaMemset(X_device, 0, g.size * g.size);

	cublasDdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, g.size,
		X_temp_device, g.size, 
		degree_device_dense, 1,
		X_device, g.size);
	
	cudaDeviceSynchronize();	
	cudaFree(X_temp_device);
	cudaFree(adj_device_dense);
	end = Clock::now();
	profile.compute_x = std::chrono::duration_cast<milliseconds>(end - begin);


	/* Section 3: Compute S = sum(X^{0}....X^{window_size}) */
	/* Procedure
	  1. Copy X to S
	  2. Copy X to W
	  3. W' = W * X
	  4. S' = S + W'  
	  5. W = W'
	  6. S = S'
	*/
	
	/* Step 0: Declare all variables */
	double *S_device;
        //double *S_host;
	double *W_device;
        //double *W_host;
	double *S_temp_device;
        //double *S_temp_host;
	double *W_temp_device;
        //double *W_temp_host;	

	const double alpha = 1.00;
	double beta = 1.00;

	begin = Clock::now();
	/* Step 1: Copy X to S */
	log("Copying X to S");

	cudaMalloc(&S_temp_device, g.size * g.size * sizeof(double));
	cudaMalloc(&S_device, g.size * g.size *sizeof(double));
	cudaMemcpy(S_device, X_device, g.size * g.size * sizeof(double), cudaMemcpyDeviceToDevice);

	/* Step 2: Copy X to temp */
	log("Copying X to W");
	
	cudaMalloc(&W_temp_device, g.size * g.size * sizeof(double));
	cudaMalloc(&W_device, g.size * g.size *sizeof(double));
	cudaMemcpy(W_device, X_device, g.size * g.size * sizeof(double), cudaMemcpyDeviceToDevice);

	for(int i=2;i<=window_size;i++){
		/* Step 3: temp' = temp * X */
		log("Computing W' = W * X");
		cudaMemset(W_temp_device, 0, g.size * g.size);
		beta = 0;
		cublasDgemm(cublas_handle, 
				CUBLAS_OP_N, CUBLAS_OP_N,
				g.size, g.size, g.size,
				&alpha,
				W_device, LDA,
				X_device, LDA,
				&beta,
				W_temp_device, LDA);

		/* Step 4: S = S + temp */
		log("Computing S' = S + W'");
		cudaMemset(S_temp_device, 0, g.size * g.size);
		beta = 1;
		cublasDgeam(cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				g.size, g.size,
				&alpha,
				S_device, LDA,
				&beta,
				W_temp_device, LDA,
				S_temp_device, LDA);

		/* Step 5: temp = temp' */
		log("Copying W' to W");
		cudaMemset(W_device, 0, g.size * g.size);
		cudaMemcpy(W_device, W_temp_device, g.size * g.size * sizeof(double), cudaMemcpyDeviceToDevice);

		/* Step 6: S = S' */
		log("Copying S' to S");
		cudaMemset(S_device, 0, g.size * g.size);
		cudaMemcpy(S_device, S_temp_device, g.size * g.size * sizeof(double), cudaMemcpyDeviceToDevice);
	}

	cudaFree(S_temp_device);
	cudaFree(W_temp_device);
	cudaFree(W_device);

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

	cublasDscal(cublas_handle, g.size * g.size,
                    &val,
                    S_device, 1);

	//S_host = (double *) malloc(g.size * g.size * sizeof(double));
	end = Clock::now();
	profile.compute_s = std::chrono::duration_cast<milliseconds>(end - begin);

	begin = Clock::now();
	log("Computing M");

        /* Section 5: Compute M = D^{-1/2} * S * D^{-1/2} */
	/* Procedure
	   1. Compute M' = D' * S
	   2. Compute M = M' * D'
	*/

	/* Step 1: Compute M' = D' * S */
	log("Computing M' = D' * S");

	double *M_temp_device;
	//double *M_host;

	cudaMalloc(&M_temp_device, g.size * g.size * sizeof(double));

	cublasDdgmm(cublas_handle, CUBLAS_SIDE_LEFT,
		g.size, g.size,
		S_device, g.size, 
		degree_device_dense, 1,
		M_temp_device, g.size);

	cudaFree(S_device);

	log("Computing M = M' * S'");
	double *M_device;
	cudaMalloc(&M_device, g.size * g.size * sizeof(double));

	cublasDdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, g.size,
		M_temp_device, g.size, 
		degree_device_dense, 1,
		M_device, g.size);

	cudaFree(M_temp_device);

	/* Section 6: Compute M'' = log(max(M,1)) */
	
	/* Procedure 
	   1. Prune M and take log
	   2. Create CSR struct for M''
	   3. Compute nnzPerVector for M''
	*/

	/* Step 1: Prune M and take log */
	log("Pruning M");

	prune_m<<<grids,threads>>>(M_device, g.size);
       	cudaDeviceSynchronize(); 

	log("Pruned M");

	/* Step 2: Create CSR struct for both matrices */
	log("Converting dense matrix to CSR format");	
	csr M_cap;    /* Variable to hold adjacency matrix in CSR format */

	M_cap.nnz = 0; /* Initialize number of non zeros in adjacency matrix */

	/* Step 6: Compute nnz/row of dense matrix */	
	log("Computing nnzPerVector for M''");

	cudaMalloc(&M_cap.d_nnzPerVector, 
			g.size * sizeof(int));
	cusparseDnnz(cusparse_handle, 
			CUSPARSE_DIRECTION_ROW, 
			g.size, g.size, 
			mat_descr, 
			M_device, LDA, 
			M_cap.d_nnzPerVector, &M_cap.nnz);
	M_cap.h_nnzPerVector = (int *)malloc(g.size * sizeof(int));
	cudaMemcpy(M_cap.h_nnzPerVector, 
			M_cap.d_nnzPerVector, 
			g.size * sizeof(int), 
			cudaMemcpyDeviceToHost); 
	if(DEBUG){
    		printf("Number of nonzero elements in dense adjacency matrix = %i\n", M_cap.nnz);
    		
		if(VERBOSE)
		for (int i = 0; i < g.size; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, M_cap.h_nnzPerVector[i]);
	}


	/* Step 6: Convert dense matrix to sparse matrices */
	allocate_csr(&M_cap, M_cap.nnz, g.size);
	cusparseDdense2csr(cusparse_handle, 
			g.size, g.size, 
			mat_descr,
		       	M_device,	
			LDA, 
			M_cap.d_nnzPerVector, 
			M_cap.d_values, M_cap.d_rowIndices, M_cap.d_colIndices); 
	if(VERBOSE){
		device2host(&M_cap, M_cap.nnz, g.size);	
		print_csr(
    			g.size,
    			M_cap.nnz,
    			M_cap,
    			"Adjacency matrix");
	}

	cudaFree(M_device);

	device2host(&M_cap, M_cap.nnz, g.size);
	log("Completed conversion of data from dense to sparse");
	end = Clock::now();
	profile.compute_m = std::chrono::duration_cast<milliseconds>(end - begin);
		
	/* Section 7: Compute SVD of objective matrix */	

	begin = Clock::now();
	char whichS = 'L';
	char whichV = 'L';

	MKL_INT pm[128];
	mkl_sparse_ee_init(pm);
	//pm[1] = 100;
	//pm[2] = 2;
	//pm[4] = 60;

	MKL_INT mkl_rows = g.size;
	MKL_INT mkl_cols = g.size;


	//MKL_INT rows_start[mkl_rows];
	//MKL_INT rows_end[mkl_rows];

	MKL_INT *rows_start;
	MKL_INT *rows_end;

	rows_start = (MKL_INT *)mkl_malloc(mkl_rows * sizeof(MKL_INT),64);
	rows_end = (MKL_INT *)mkl_malloc(mkl_rows * sizeof(MKL_INT),64);

	for(int i=0;i<mkl_rows;i++){
		rows_start[i] = M_cap.h_rowIndices[i];
		rows_end[i] = M_cap.h_rowIndices[i+1];
	}

	
	//MKL_INT mkl_col_idx[M_cap.nnz];

	MKL_INT *mkl_col_idx;
	mkl_col_idx = (MKL_INT*)mkl_malloc(M_cap.nnz * sizeof(MKL_INT), 64);

	int mkl_temp;
	for(int i=0;i<M_cap.nnz;i++){
		mkl_temp = M_cap.h_colIndices[i];
		mkl_col_idx[i] = mkl_temp;
	}


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
	
	if(mkl_status){
		std::cout<<"SVD failed"<<std::endl;
		exit(0);	
	}


	log("Computed SVD via MKL");

	if(DEBUG){
	std::cout<<"Number of singular found: "<<k<<std::endl;
	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";
	}

	double *U_device, *Si_device;
	//double *U_host;
	double *Si_host;
	double *E_device, *E_host;

	cudaMalloc(&U_device, g.size * dimension * sizeof(double));
	cudaMalloc(&E_device, g.size * dimension * sizeof(double));
	cudaMalloc(&Si_device, dimension * sizeof(double));

	//U_host = (double *) malloc(g.size * dimension * sizeof(double));
	E_host = (double *) malloc(g.size * dimension * sizeof(double));
	Si_host = (double *) malloc(dimension * sizeof(double));

	cudaMemcpy(U_device, K_L_mkl, g.size * dimension * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Si_device, E_mkl, dimension * sizeof(double), cudaMemcpyHostToDevice);

	transform_si<<<grids,threads>>>(Si_device, dimension);

	cudaMemcpy(Si_host, Si_device, dimension * sizeof(double), cudaMemcpyDeviceToHost);

	std::cout<<"\n";
	cublasDdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, dimension,
		U_device, g.size, 
		Si_device, 1.0,
		E_device, g.size);

	cudaMemcpy(E_host, E_device, g.size * dimension * sizeof(double), cudaMemcpyDeviceToHost);

	end = Clock::now();
	profile.svd = std::chrono::duration_cast<milliseconds>(end - begin);


	overall_end = Clock::now();
	profile.emb = std::chrono::duration_cast<milliseconds>(overall_end - overall_begin);


	write_profile("profile.txt", profile);
	write_embeddings(argv[6],E_host, g.size, dimension);

	mkl_free(rows_start);	
	mkl_free(rows_end);	
	mkl_free(mkl_col_idx);	

}
