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

	/* General housekeeping */
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
        Clock::time_point begin, end;
	info profile; 
	profile.dataset = "blogcatalog";
	profile.algo = "small";

	/* Load graph */
        log("Reading data from file");

        begin = Clock::now(); 
	Graph g =  read_graph("../data/test/small_test.csv","edgelist");
        end = Clock::now(); 

	profile.iptime = std::chrono::duration_cast<milliseconds>(end - begin);	

	/* CUDA housekeeping */
	log("Running Initialization routine");
	log("Defining Threads");

	begin = Clock::now();
	float num_threads = 128;	
	dim3 threads(num_threads);
	dim3 grid((int)ceil((float)g.size/num_threads));

	/* cuBlas housekeeping */
	log("Creating cuBlas variables");
	cublasHandle_t handle;
	cublasCreate(&handle);
	float al=1.0f;
	float bet=1.0f;

	/* cuSparse housekeeping */
	log("Creating cuSparse variables");
	cusparseHandle_t sp_handle;
	cusparseCreate(&sp_handle);

	/* cuSolver housekeeping */
	log("Setting up cuSolver");	
	int lwork = 0;
	signed char jobu = 'A';
	signed char jobvt = 'N';
	float *d_work, *d_rwork;
	int *devInfo;
	
	cudaMalloc(&devInfo, sizeof(int));
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	cusolverDnSgesvd_bufferSize(cusolverH,g.size, g.size,&lwork);
	cudaMalloc(&d_work, sizeof(float) * lwork);
	cudaMalloc(&d_rwork, sizeof(float) *( g.size - 1));

	/* Initialize and allocate variables */
	log("Setting up host variables");
	float *U, *VT, *Si, *W;
	float *X;
	float *S;
	float *M;
	
	int window_size = 3;
	profile.window_size = window_size;
	int size = g.size * g.size * sizeof(float);
	int b = 1;
	int dimension = 2;
	profile.dimension = dimension;
	//int *devInfoH;
	
	X = (float *)malloc(size);
	S = (float *)malloc(size);
	M = (float *)malloc(size);	
	U = (float*)malloc(g.size * dimension *sizeof(float));
	VT = (float*)malloc(size);
	W = (float*)malloc(size);
	Si = (float*)malloc(g.size * sizeof(float));
	//devInfoH = (int *)malloc(sizeof(int));

	log("Setting up device variables");
	float *D_device;
	float *temp_device;
	float *temp1_device;
	float *X_device;
	float *A_device;
	float *S_device;
	float *M_device;
	float *M_temp_device;
	float *U_device, *VT_device, *Si_device;
	float *W_device; //auxillary device array

	cudaMalloc(&U_device, g.size * dimension *sizeof(float));
	cudaMalloc(&Si_device, g.size * sizeof(float));
	cudaMalloc(&VT_device, size);
	cudaMalloc(&W_device, size);

	cudaMalloc(&D_device, size);
	cudaMalloc(&A_device, size);
	cudaMalloc(&X_device, size);
	cudaMalloc(&temp_device, size);
	cudaMalloc(&temp1_device, size);
	cudaMalloc(&S_device, size);
	cudaMalloc(&M_device, size);
	cudaMalloc(&M_temp_device, size);

	cudaMemset(A_device, 0, size);
	cudaMemset(D_device, 0, size);
	cudaMemset(X_device, 0, size);
	cudaMemset(S_device, 0, size);
	cudaMemset(M_device, 0, size);
	cudaMemset(M_temp_device, 0, size);
	cudaMemset(temp_device, 0, size);
	cudaMemset(temp1_device, 0, size);
	end = Clock::now();
	profile.init = std::chrono::duration_cast<milliseconds>(end - begin);
	
	/* Copy necessary variables to device */
	/* 
	   Note: Make this a non-blocking operation using
	   using Async since g.degree, g.adj and g.size 
	   are available at the very beginning
	*/
	log("Moving data to device");
	begin = Clock::now();
	cudaMemcpy(D_device, g.degree, size, cudaMemcpyHostToDevice);	
	cudaMemcpy(A_device, g.adj, size , cudaMemcpyHostToDevice);	
	end = Clock::now();
	profile.gpuio = std::chrono::duration_cast<milliseconds>(end - begin);;

	/* Compute D = D^{-1/2} */
	begin = Clock::now();
	log("Computing normalized D");
	compute_d<<<grid, threads>>>(D_device, g.size);
	cudaDeviceSynchronize();
	end = Clock::now();
	profile.compute_d = std::chrono::duration_cast<milliseconds>(end - begin);;
	
	/* Compute X = D^{-1/2}AD^{-1/2} */
	log("Computing X");
	begin = Clock::now();
	cublasSgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, g.size,
		    &al,
	            A_device,g.size, 
		    D_device, g.size,
		    &bet, 
		    temp_device, g.size);
	cudaDeviceSynchronize();	
	cublasSgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, g.size,
		    &al,
		    D_device, g.size,
	            temp_device,g.size, 
		    &bet, 
		    X_device, g.size);
	cudaDeviceSynchronize();

	end = Clock::now();
	profile.compute_x = std::chrono::duration_cast<milliseconds>(end - begin);;	
	/* Compute S = sum(X^{0}....X^{window_size}) */
	
	cudaMemcpy(S_device, X_device, size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(temp_device, X_device, size, cudaMemcpyDeviceToDevice);

	begin = Clock::now();
	for(int i=2;i<=window_size;i++){
		std::cout<<"Computing X^"<<i<<std::endl;
		cublasSgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, g.size,
		    &al,
		    X_device, g.size,
	            temp_device,g.size, 
		    &bet, 
		    temp1_device, g.size);
		cudaDeviceSynchronize();		
		
		// Use cublas addition functions
		cublasSgeam(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, 
		    &al,
		    S_device, g.size,
		    &bet, 
	            temp1_device,g.size, 
		    S_device, g.size);
		cudaDeviceSynchronize();
		
		cudaMemcpy(X_device, temp1_device, size, cudaMemcpyDeviceToDevice);
		cudaMemset(temp1_device,0,size);
	}

		
	
	// Compute S = S * (vol / (window_size * b))
	transform_s<<<grid,threads>>>(S_device,g.volume, window_size, b, g.size);
        cudaDeviceSynchronize();
	end = Clock::now();
	profile.compute_s = std::chrono::duration_cast<milliseconds>(end - begin);;
	
	
	// Compute M = D^{-1/2} * S * D^{-1/2}
	cudaMemset(temp_device, 0, size); 
	
	begin = Clock::now();
	log("Computing M");
	cublasSgemm(handle, 
	    CUBLAS_OP_N, CUBLAS_OP_N, 
	    g.size, g.size, g.size,
	    &al,
	    S_device, g.size,
	    D_device,g.size, 
	    &bet, 
	    temp_device, g.size);
        cudaDeviceSynchronize();

	cublasSgemm(handle, 
	    CUBLAS_OP_N, CUBLAS_OP_N, 
	    g.size, g.size, g.size,
	    &al,
	    D_device, g.size,
	    temp_device,g.size, 
	    &bet, 
	    M_device, g.size);
        cudaDeviceSynchronize();
	
	// Compute M = log(max(Mi,1))
	log("Transforming M");
	transform_m<<<grid,threads>>>(M_device, g.size);
        cudaDeviceSynchronize();
	
	end = Clock::now();
	profile.compute_m = std::chrono::duration_cast<milliseconds>(end - begin);;
	
	// Perform SVD on M
	begin = Clock::now();
	log("Performing SVD");
	cudaMemcpy(M, M_device, size, cudaMemcpyDeviceToHost);

	log("Printing M_CAP");
	//print_matrix(M, g.size);

    	cusparseMatDescr_t descrM;      
	cusparseCreateMatDescr(&descrM);
    	cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    	cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);

	int nnzM = 0;

	const int lda = g.size;
   	int *d_nnzPerVectorM;   
	cudaMalloc(&d_nnzPerVectorM, g.size * sizeof(*d_nnzPerVectorM));
	cusparseSnnz(sp_handle, CUSPARSE_DIRECTION_ROW, g.size, g.size, descrM, M_device, lda, d_nnzPerVectorM, &nnzM);

	int *h_nnzPerVectorM = (int *)malloc(g.size * sizeof(*h_nnzPerVectorM));
	cudaMemcpy(h_nnzPerVectorM, d_nnzPerVectorM, g.size * sizeof(*h_nnzPerVectorM), cudaMemcpyDeviceToHost);

    	//std::cout <<"Number of nonzero elements in dense matrix A = "<<nnzM<<std::endl;
    //for (int i = 0; i < g.size; ++i) std::cout<<"Number of nonzero elements in row"<<i << "for matrix = "<<h_nnzPerVectorM[i]<<std::endl;

    float *d_M;            cudaMalloc(&d_M, nnzM * sizeof(*d_M));
    int *d_M_RowIndices;   cudaMalloc(&d_M_RowIndices, (g.size + 1) * sizeof(*d_M_RowIndices));
    int *d_M_ColIndices;   cudaMalloc(&d_M_ColIndices, nnzM * sizeof(*d_M_ColIndices));
    cusparseSdense2csr(sp_handle, g.size, g.size, descrM, M_device, lda, d_nnzPerVectorM, d_M, d_M_RowIndices, d_M_ColIndices);
    float *h_M = (float *)malloc(nnzM * sizeof(*h_M));        
    int *h_M_RowIndices = (int *)malloc((g.size + 1) * sizeof(*h_M_RowIndices));
    int *h_M_ColIndices = (int *)malloc(nnzM * sizeof(*h_M_ColIndices));
    cudaMemcpy(h_M, d_M, nnzM * sizeof(*h_M), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_M_RowIndices, d_M_RowIndices, (g.size + 1) * sizeof(*h_M_RowIndices), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_M_ColIndices, d_M_ColIndices, nnzM * sizeof(*h_M_ColIndices), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < (g.size + 1); ++i) std::cout<<"h_A_RowIndices:"<<i<<":"<<h_M_RowIndices[i]<<std::endl; 
    //for (int i = 0; i < nnzM; ++i) std::cout<<"h_A_ColIndices"<<":"<<i <<":"<<h_M_ColIndices[i] <<std::endl;  
    //for (int i = 0; i < nnzM; ++i) std::cout<<"h_A_Val"<<":"<<i <<":"<<h_M[i] <<std::endl;  


	char whichS = 'L';
	char whichV = 'L';
	MKL_INT pm[128];
	mkl_sparse_ee_init(pm);

	MKL_INT mkl_rows = g.size;
	MKL_INT mkl_cols = g.size;

	MKL_INT rows_start[mkl_rows];
	MKL_INT rows_end[mkl_rows];

	for(int i=0;i<mkl_rows;i++){
		rows_start[i] = h_M_RowIndices[i];
		rows_end[i] = h_M_RowIndices[i+1];
	}
	

	MKL_INT mkl_col_idx[nnzM];
	for(int i=0;i<nnzM;i++){
		mkl_col_idx[i] = h_M_ColIndices[i];
	}
	
	sparse_matrix_t M_mkl;
	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;	
		
	mkl_sparse_s_create_csr(&M_mkl, indexing,
				mkl_rows, mkl_cols,
				rows_start,rows_end,
				mkl_col_idx, h_M);

	log("Created MKL sparse");

	matrix_descr mkl_descrM;
	mkl_descrM.type = SPARSE_MATRIX_TYPE_GENERAL;  
	mkl_descrM.mode = SPARSE_FILL_MODE_UPPER;
	mkl_descrM.diag = SPARSE_DIAG_NON_UNIT;

	MKL_INT  k0 = dimension;
	MKL_INT k;
	float *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;

	E_mkl = (float *)mkl_malloc( k0*sizeof( float), 64 );
	K_L_mkl = (float *)mkl_malloc( k0*mkl_rows*sizeof( float), 64 );
	K_R_mkl = (float *)mkl_malloc( k0*mkl_cols*sizeof( float), 64 );
	res_mkl = (float *)mkl_malloc( k0*sizeof( float), 64 );
	
	mkl_sparse_s_svd(&whichS, &whichV, pm, 
			M_mkl, mkl_descrM, 
			k0, &k, 
			E_mkl, 
			K_L_mkl,
			K_R_mkl, 
			res_mkl); 
	log("computed SVD via MKL");
	
	end = Clock::now();
	profile.svd = std::chrono::duration_cast<milliseconds>(end - begin);	

	cudaMemcpy(Si_device, E_mkl, dimension * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(U_device, K_L_mkl, dimension * g.size * sizeof(float), cudaMemcpyHostToDevice);
	
	begin = Clock::now();
	log("Transforming Si");
	sqrt_si<<<grid, threads>>>(Si_device, dimension);	
        cudaDeviceSynchronize();

	cudaMemcpy(Si, Si_device, dimension * sizeof(float), cudaMemcpyDeviceToHost);
	log("Printing sqrt S");
	for(int i=0;i<dimension;i++)
		std::cout<<Si[i]<<" ";

	cudaMemcpy(U, U_device, dimension * g.size * sizeof(float), cudaMemcpyDeviceToHost);
	log("Printing U");
	for(int i=0;i<g.size;i++){
		for(int j=0;j<dimension;j++){
			std::cout<<U[i*dimension + j]<<" ";
		}
		std::cout<<"\n";
	}

	log("Generating embeddings");
	cublasSdgmm(handle, 
	    CUBLAS_SIDE_LEFT, 
	    g.size, dimension,
	    U_device, g.size,
	    Si_device,1, 
	    W_device, g.size);
        cudaDeviceSynchronize();
	end = Clock::now();
	profile.emb = std::chrono::duration_cast<milliseconds>(end - begin);;

	cudaMemcpy(W, W_device, size, cudaMemcpyDeviceToHost);

	write_embeddings("blogcatalog.emb",W, g.size, dimension);	
	write_profile("profile.txt", profile);		
	log("Done");
	/***********
	* Clean up *
	***********/

	free(X);
	free(S);
	free(M);
	
	free(U); 
	free(VT);
	free(Si);
	free(W);

	// DEVICE
	cudaFree(D_device);
	cudaFree(temp_device); 
	cudaFree(temp1_device); 
	cudaFree(X_device);
	cudaFree(A_device);
	cudaFree(S_device);
	cudaFree(M_device);

	cudaFree(U_device);
	cudaFree(VT_device); 
	cudaFree(Si_device);
	cudaFree(W_device); //auxillary device array

	cudaFree(d_work);
	cudaFree(d_rwork);
	cudaFree(devInfo);

}
