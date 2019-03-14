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
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
			adj[id*size+id] = 1.00;	
	}else{
			adj[id*size+id] = 0.0;
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

__global__ void unbalanced_dgmm(DT *diag, DT *mat, int m, int n, int k){
	int row = blockIdx.y * blockDim.x + threadIdx.y;
	int col = blockIdx.x * blockDim.y + threadIdx.x;

	if(row >= n || col >= k) return;

	mat[row + col * n] = mat[row+col*n] * diag[row];
	
}


__global__ void deepwalk_filter(DT *A, int size, DT window){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;

	if(A[id] >= 1){
		A[id] = 1;
	}else{
		A[id] = (A[id] * (1 - pow(A[id], window))) / ((1-A[id]) * window);
	}

	if(A[id] <= 0)
		A[id] = 0;

	A[id] = sqrt(A[id]);

}

int main ( int argc, char **argv ){
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
	8. Eigen rank
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
	int rank = std::atoi(argv[8]);

	profile.window_size = window_size;
	profile.dimension = dimension;

	/* Load graph */
        log("Reading data from file");
	
	begin = Clock::now(); 
	Graph g =  read_graph(argv[5],"edgelist", argv[7]);
	end = Clock::now();

	profile.iptime = std::chrono::duration_cast<milliseconds>(end - begin);


	/* CUDA housekeeping */
	float num_threads = 128;
	dim3 threads(num_threads);
	dim3 grids((int)ceil((float)(g.size*g.size)/num_threads));

	/* CuSolver housekeeping */
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);

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

	/* Step 1: Create dense adjacency matrix and degree matrixx on device */
	log("Creating dense device array");
	DT *adj_device_dense;	
	DT *degree_device_dense; 
	DT *adj_host_dense;	
	DT *degree_host_dense; 

	/* Step 2a: Allocate space for adjacency and degree matrix on device*/
	log("Allocating space for degree and adjacency mat on device");
	cudaMalloc(&adj_device_dense, 
			g.size * g.size * sizeof(DT)); 	
	cudaMalloc(&degree_device_dense, 
			g.size * sizeof(DT)); 

	/* Step 2a: Allocate space for adjacency and degree matrix on host */
	log("Allocating space for degree and adjacency matrix on host");
	adj_host_dense = (DT *) malloc(g.size * g.size * sizeof(DT));
	degree_host_dense = (DT *) malloc(g.size * sizeof(DT));

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
	//std::sort(g.degree1D, g.degree1D + g.size);
	//cudaMemcpy(degree_host_dense, degree_device_dense, g.size * sizeof(DT), cudaMemcpyDeviceToHost);

	/*Step 4: Compute volume and preprocess degree */
	preprocess_laplacian<<<grids,threads>>>(adj_device_dense, degree_device_dense, g.size);

	/*REMOVE*/
	//cudaMemcpy(g.degree1D, degree_device_dense, g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	//std::sort(g.degree1D, g.degree1D + g.size);

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

	/*REMOVE*/
	//cudaMemcpy(degree_host_dense, degree_device_dense, g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	//std::sort(g.degree1D, g.degree1D + g.size);
	//log("Computed normalized D");

	/* Step 2: Compute X' = D' * A */
	log("Computing X' = D' * A");
	DT *X_temp_device;
	DT *X_temp_host;

	cudaMalloc(&X_temp_device, g.size * g.size * sizeof(DT));
	cudaMemset(X_temp_device, 0, g.size * g.size);

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_LEFT,
		g.size, g.size,
		adj_device_dense, g.size, 
		degree_device_dense, 1,
		X_temp_device, g.size);
	cudaFree(adj_device_dense);


//	/* Step 3: Compute X = X' * D */
	log("Computing X = X' * D");
	DT *X_device;
	DT *X_host;
	cudaMalloc(&X_device, g.size * g.size * sizeof(DT));
	cudaMemset(X_device, 0, g.size * g.size);

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, g.size,
		X_temp_device, g.size, 
		degree_device_dense, 1,
		X_device, g.size);
	
	cudaFree(X_temp_device);

	/*REMOVE*/
	//X_host = (DT *) malloc(sizeof(DT) * g.size * g.size);
	//cudaMemcpy(X_host, X_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	//std::sort(X_host, X_host + (g.size * g.size));	
	//cudaMemcpy(degree_host_dense, degree_device_dense, g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	/* Section 3: Compute Eig(X) */	

	log("Eigen(X)");
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;	
	int h_meig = 0;
	int lwork = 0;
	int *devInfo;
	DT *d_work;
	cudaMalloc(&devInfo, sizeof(int));

	DT *e_device;
	DT *EV_device;
	cudaMalloc(&e_device, g.size * sizeof(DT));
	cudaMalloc(&EV_device, g.size * rank * sizeof(DT));

	log("Computing buffersize");
	cusolverDnSsyevdx_bufferSize(
	    cusolverH,
	    jobz, 
	    range,
	    uplo, 
	    g.size,
	    X_device,
	    LDA,
	    0.00,
	    0.00,
	    1,
	    rank,
	    &h_meig,
	    e_device,
	    &lwork);

	cudaMalloc(&d_work, sizeof(DT) * lwork);

	log("Computing Eigen");
	cusolverDnSsyevdx(
	    cusolverH,
	    jobz, 
	    range,
	    uplo, 
	    g.size,
	    X_device,
	    LDA,
	    0.00,
	    0.00,
	    g.size - rank + 1,
	    g.size,
	    &h_meig,
	    e_device, 
	    d_work,
	    lwork,
	    devInfo);
	cudaMemcpy(EV_device, X_device, g.size * rank * sizeof(DT), cudaMemcpyDeviceToDevice);
	log("Computed eigen");

	/*REMOVE*/
	DT *e_host;
	e_host = (DT *)malloc(g.size * sizeof(DT));
	////cudaMemcpy(e_host, e_device, g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	//DT *EV_host;
	//EV_host = (DT *) malloc(sizeof(DT) * g.size * rank);
	//gpuErrchk(cudaMemcpy(EV_host, X_device, g.size * rank * sizeof(DT), cudaMemcpyDeviceToHost));
	//
	//cudaMemcpy(degree_host_dense, degree_device_dense, g.size * sizeof(float), cudaMemcpyDeviceToHost);

	num_threads = 32;
	dim3 threads2D(num_threads,num_threads);
	dim3 grids2D(ceil(((float)g.size)/num_threads), ceil(((float)rank/num_threads)));

	DT *D_rt_invU_device;
	cudaMalloc(&D_rt_invU_device, g.size * rank * sizeof(DT));
	cublasSdgmm(cublas_handle, CUBLAS_SIDE_LEFT,
		g.size, rank,
		EV_device, g.size, 
		degree_device_dense, 1,
		EV_device, g.size);

	cudaDeviceSynchronize();

	deepwalk_filter<<<grids,threads>>>(e_device, rank, window_size);	
	cudaDeviceSynchronize();

	cudaMemcpy(e_host, e_device, rank * sizeof(DT), cudaMemcpyDeviceToHost);
	
	for(int i=0;i<rank;i++){
		std::cout<<e_host[i]<<" ";
	}

	///*REMOVE*/
	//cudaMemcpy(EV_host, EV_device, g.size * rank * sizeof(DT), cudaMemcpyDeviceToHost);
	//std::cout<<std::endl;
	//for(int i=0;i<g.size * rank;i++){
	//	std::cout<<"EV "<<i<<":"<<EV_host[i]<<std::endl;
	//}	

	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, rank,
		EV_device, g.size, 
		e_device, 1,
		EV_device, g.size);
	cudaDeviceSynchronize();
		
	/*REMOVE*/
	//cudaMemcpy(EV_host, EV_device, g.size * rank * sizeof(DT), cudaMemcpyDeviceToHost);
	//for(int i=0;i<g.size * rank;i++){
	//	std::cout<<"EV "<<i<<":"<<EV_host[i]<<std::endl;
	//}	


	log("Computing M");

        /* Section 5: Compute M = (EV_device * EV_device.T) * (vol/b) */

	DT *M_device;
	cudaMalloc(&M_device, g.size * g.size * sizeof(DT));

	DT alf = 1.00;
	DT beta = 0.00;

	cublasSgemm(cublas_handle,
                           CUBLAS_OP_N, CUBLAS_OP_T,
                           g.size, g.size, rank,
                           &alf,
                           EV_device, g.size,
                           EV_device, g.size,
                           &beta,
			   M_device, g.size
                           );
	cudaDeviceSynchronize();

	/*REMOVE*/
	//std::cout<<std::endl;
	//DT *M_host;
	//M_host = (DT *) malloc(sizeof(DT) * g.size * g.size);
	//cudaMemcpy(M_host, M_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	//for(int i=0;i<g.size * g.size;i++){
	//	std::cout<<"M "<<i<<":"<<M_host[i]<<std::endl;
	//}

	
	DT val = ((DT) g.volume) / ((DT) b);
	cublasSscal(cublas_handle, g.size * g.size,
			&val,
			M_device, 1);
	
	/*REMOVE*/
	//std::cout<<std::endl;
	//cudaMemcpy(M_host, M_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToHost);
	//for(int i=0;i<g.size * g.size;i++){
	//	std::cout<<"M "<<i<<":"<<M_host[i]<<std::endl;
	//}


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
	cusparseSnnz(cusparse_handle, 
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
	cusparseSdense2csr(cusparse_handle, 
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

	/* REMOVE*/
	//for(int i=0;i<M_cap.nnz;i++){
	//	std::cout<<"M_cap: "<<M_cap.h_values[i]<<std::endl;
	//}

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

	mkl_sparse_s_create_csr(&M_mkl, indexing,
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

	DT *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;

	E_mkl = (DT *)mkl_malloc(k0 * sizeof(DT), 128);
	K_L_mkl = (DT *)mkl_malloc( k0*mkl_rows*sizeof( DT), 128 );
        K_R_mkl = (DT *)mkl_malloc( k0*mkl_cols*sizeof( DT), 128 );
        res_mkl = (DT *)mkl_malloc( k0*sizeof( DT), 128 );

	memset(E_mkl, 0 , k0);
	memset(K_L_mkl, 0 , k0);
	memset(K_R_mkl, 0 , k0);
	memset(res_mkl, 0 , k0);

	int mkl_status = 0;

	log("Computing SVD via MKL");
	mkl_status = mkl_sparse_s_svd(&whichS, &whichV, pm,
			M_mkl, mkl_descrM,
			k0, &k,
			E_mkl,
			K_L_mkl,
			K_R_mkl,
			res_mkl);
	log("Computed SVD via MKL");

	if(DEBUG){
	std::cout<<"Number of singular found: "<<k<<std::endl;
	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";
	}

	DT *U_device, *Si_device;
	DT *U_host;
	DT *Si_host;
	DT *E_device, *E_host;

	cudaMalloc(&U_device, g.size * dimension * sizeof(DT));
	cudaMalloc(&E_device, g.size * dimension * sizeof(DT));
	cudaMalloc(&Si_device, dimension * sizeof(DT));

	U_host = (DT *) malloc(g.size * dimension * sizeof(DT));
	E_host = (DT *) malloc(g.size * dimension * sizeof(DT));
	Si_host = (DT *) malloc(dimension * sizeof(DT));

	cudaMemcpy(U_device, K_L_mkl, g.size * dimension * sizeof(DT), cudaMemcpyHostToDevice);
	cudaMemcpy(Si_device, E_mkl, dimension * sizeof(DT), cudaMemcpyHostToDevice);

	transform_si<<<grids,threads>>>(Si_device, dimension);

	cudaMemcpy(Si_host, Si_device, dimension * sizeof(DT), cudaMemcpyDeviceToHost);

	std::cout<<"\n";
	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
		g.size, dimension,
		U_device, g.size, 
		Si_device, 1.0,
		E_device, g.size);

	cudaMemcpy(E_host, E_device, g.size * dimension * sizeof(DT), cudaMemcpyDeviceToHost);

	write_embeddings("blogcatalog.emb",E_host, g.size, dimension);

	mkl_free(rows_start);	
	mkl_free(rows_end);	
	mkl_free(mkl_col_idx);	

	cudaDeviceReset();
}
