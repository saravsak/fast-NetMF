#include<stdlib.h>
#include<iostream>
#include<time.h>
#include<chrono>
#include<algorithm>
#include<numeric>
#include<math.h>
#include<stdio.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include <cusparse_v2.h> 

#include "../utils/graph.h"
#include "../utils/io.h"

#include "../utils/model.h"

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
__global__ void preprocess_laplacian(DT* adj, DT *degree, int size){
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
__global__ void compute_d(DT* deg, int size){
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

__global__ void transform_si(DT* S, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = sqrt(S[id]); 
}

__global__ void transform_s(DT* S, float val, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;

	DT mem = S[id];
	
	S[id] = mem * val; 
}

__global__ void prune_m(DT* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;

	if(M[id] <= 1)
		M[id] = 0;
	else
		M[id] = log(M[id]);	
}


__global__ void transform_m(DT* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	M[id] =log(M[id]);
}

__global__ void sqrt_si(float* S, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = sqrt((float) S[id]);
}

void print_matrix(DT* S, int size){
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
	csr_mat->h_values = (DT *) malloc(nnz * sizeof(DT));
	csr_mat->h_colIndices = (int *) malloc(nnz * sizeof(int));

	cudaMalloc(&csr_mat->d_values, nnz*sizeof(DT));
	cudaMalloc(&csr_mat->d_colIndices, nnz * sizeof(int));

}

void allocate_csr(csr *csr_mat, int nnz, int num_rows){
	csr_mat->h_values = (DT *) malloc(nnz * sizeof(DT));
	csr_mat->h_colIndices = (int *) malloc(nnz * sizeof(int));
	csr_mat->h_rowIndices = (int *) malloc((num_rows+1) * sizeof(int));

	cudaMalloc(&csr_mat->d_values, nnz*sizeof(DT));
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
	memcpy(to_mat->h_values, from_mat->h_values, to_mat->nnz * sizeof(DT));
	memcpy(to_mat->h_colIndices, from_mat->h_colIndices, to_mat->nnz * sizeof(int));
	memcpy(to_mat->h_rowIndices, from_mat->h_rowIndices, (num_rows + 1) * sizeof(int));


	/* Copy device variables */
	cudaMemcpy(to_mat->d_values, from_mat->d_values, to_mat->nnz * sizeof(DT), cudaMemcpyDeviceToDevice);
	cudaMemcpy(to_mat->d_colIndices, from_mat->d_colIndices, to_mat->nnz * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(to_mat->d_rowIndices, from_mat->d_rowIndices, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

}

void device2host(csr *csr_mat, int nnz, int num_rows){
	cudaMemcpy(csr_mat->h_values, csr_mat->d_values, 
			nnz * sizeof(DT), cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_mat->h_colIndices, csr_mat->d_colIndices, 
			nnz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_mat->h_rowIndices, csr_mat->d_rowIndices, 
			(num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
}

void host2device(csr *csr_mat, int nnz, int num_rows){
	cudaMemcpy(csr_mat->d_values, csr_mat->h_values, 
			nnz * sizeof(DT), cudaMemcpyHostToDevice);
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

	DT alf = 1.0;
	DT bet = 1.0;

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

	cusparseScsrgeam(context, m, n,
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

	cusparseScsrgemm(context, 
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
	/***********************
 	* NetMF small dense nmf*
	***********************/
	/* Argument order 
	1. Dataset name
	2. Window Size
	3. Dimension
	4. B
	5. Input
	6. Output
	7. Mapping file
	*/
	
	/* Setting args */
	char *arg_dataset = argv[1];
	char *arg_window = argv[2];
	char *arg_dimension = argv[3];
	char *arg_b = argv[4];
	char *arg_input = argv[5];
	char *arg_output = argv[6];
	char *argv_mapping = argv[7];
	char *argv_tile_size = argv[8];
	char *argv_n_iters = argv[9];

	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
        Clock::time_point begin, end;
        Clock::time_point overall_begin, overall_end;
	info profile; 
	profile.dataset = argv[1];
	profile.algo = "small-dense-nmf";
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
	DT *adj_device_dense;	
	DT *degree_device_dense; 
	//DT *adj_host_dense;	
	//DT *degree_host_dense; 

	/* Step 2a: Allocate space for adjacency and degree matrix on device*/
	log("Allocating space for degree and adjacency mat on device");
	cudaMalloc(&adj_device_dense, 
			g.size * g.size * sizeof(DT)); 	
	cudaMalloc(&degree_device_dense, 
			g.size * sizeof(DT)); 

	/* Step 2a: Allocate space for adjacency and degree matrix on host */
	log("Allocating space for degree and adjacency matrix on host");
	//adj_host_dense = (DT *) malloc(g.size * g.size * sizeof(DT));
	//degree_host_dense = (DT *) malloc(g.size * sizeof(DT));

	/* Step 3: Copy dense matrix from host to device */
	log("Copying dense matrix from host to device");	
	cudaMemcpy(adj_device_dense, 
			g.adj, 
			g.size * g.size * sizeof(DT), 
			cudaMemcpyHostToDevice);	
	cudaMemcpy(degree_device_dense, 
			g.degree1D, 
			g.size * sizeof(DT), 
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
	DT *X_temp_device;
	//DT *X_temp_host;

	cudaMalloc(&X_temp_device, g.size * g.size * sizeof(DT));
	cudaMemset(X_temp_device, 0, g.size * g.size);

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_LEFT,
		g.size, g.size,
		adj_device_dense, g.size, 
		degree_device_dense, 1,
		X_temp_device, g.size);
	cudaDeviceSynchronize();
	

//	/* Step 3: Compute X = X' * D */
	log("Computing X = X' * D");
	DT *X_device;
	//DT *X_host;
	cudaMalloc(&X_device, g.size * g.size * sizeof(DT));
	cudaMemset(X_device, 0, g.size * g.size);

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
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
	DT *S_device;
        //DT *S_host;
	DT *W_device;
        //DT *W_host;
	DT *S_temp_device;
        //DT *S_temp_host;
	DT *W_temp_device;
        //DT *W_temp_host;	

	const DT alpha = 1.00;
	DT beta = 1.00;

	begin = Clock::now();
	/* Step 1: Copy X to S */
	log("Copying X to S");

	cudaMalloc(&S_temp_device, g.size * g.size * sizeof(DT));
	cudaMalloc(&S_device, g.size * g.size *sizeof(DT));
	cudaMemcpy(S_device, X_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToDevice);

	/* Step 2: Copy X to temp */
	log("Copying X to W");
	
	cudaMalloc(&W_temp_device, g.size * g.size * sizeof(DT));
	cudaMalloc(&W_device, g.size * g.size *sizeof(DT));
	cudaMemcpy(W_device, X_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToDevice);

	for(int i=2;i<=window_size;i++){
		/* Step 3: temp' = temp * X */
		log("Computing W' = W * X");
		cudaMemset(W_temp_device, 0, g.size * g.size);
		beta = 0;
		cublasSgemm(cublas_handle, 
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
		cublasSgeam(cublas_handle,
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
		cudaMemcpy(W_device, W_temp_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToDevice);

		/* Step 6: S = S' */
		log("Copying S' to S");
		cudaMemset(S_device, 0, g.size * g.size);
		cudaMemcpy(S_device, S_temp_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToDevice);
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
	const DT val = ((DT) g.volume)/(((DT) window_size) * ((DT) b));

	if(DEBUG){
		std::cout<<"Mult value"<<val<<std::endl;
	}

	/* Step 2: Compute S[i] = S[i] * val */

	cublasSscal(cublas_handle, g.size * g.size,
                    &val,
                    S_device, 1);

	//S_host = (DT *) malloc(g.size * g.size * sizeof(DT));
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

	DT *M_temp_device;
	//DT *M_host;

	cudaMalloc(&M_temp_device, g.size * g.size * sizeof(DT));

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_LEFT,
		g.size, g.size,
		S_device, g.size, 
		degree_device_dense, 1,
		M_temp_device, g.size);

	cudaFree(S_device);

	log("Computing M = M' * S'");
	DT *M_device;
	cudaMalloc(&M_device, g.size * g.size * sizeof(DT));

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
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

	DT *M_cap = (DT *) malloc(g.size * g.size * sizeof(DT));
	cudaMemcpy(M_cap, M_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToHost);


	model nmf;

	int nmf_argc = 11;
	char *nmf_argv[nmf_argc];

	nmf_argv[0] = "-est_nmf_gpu";
	log("Set Prameters");
	nmf_argv[1] = "-K";
	nmf_argv[2] = "80";
	log("Set Prameters");
	nmf_argv[3] = "-tile_size";
	nmf_argv[4] = "10";
	log("Set Prameters");
	nmf_argv[5] = "-V";
	sprintf(nmf_argv[6], "%d", g.size);
	log("Set Prameters");
	nmf_argv[7] = "-D";
	sprintf(nmf_argv[8], "%d", g.size);
	log("Set Prameters");
	nmf_argv[9] = "-niters";
	nmf_argv[10] = "10";

	log("Set Prameters");

	int v = g.size;
	int d = g.size;
	double *M_doub = (double *) malloc(v * d * sizeof(double));


	for(int i=0;i<v;i++){
        	for(int j=0;j<d;j++){
                	M_doub[i * d + j] = M_cap[i * d + j];
        	}
    	}

	int nnz = 0;
	for(int i=0;i<g.size * g.size;i++){
		if(M_doub[i] !=0)
			nnz++;
	}

	std::cout<<"Number of nnz in pruned M"<<nnz<<std::endl;

	nmf.init(nmf_argc,nmf_argv);
	nmf.estimate_HALS_GPU(M_doub);

	write_embeddings(argv[6], nmf.DT, g.size, 80);	

}