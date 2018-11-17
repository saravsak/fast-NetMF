/* 
TODO: 
1. Change all doubles to float 
2. USe cuBlas for addition
*/

/* 
Question for prof
1. Copy stuff within GPU
2. Launch kernel without copy
3. Is it better to do more redundant work in one thread or one more kernle to do it once?
4. Results are wrong if I use same variable as result. Why?
*/
#include<stdlib.h>
#include<iostream>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>

#include "../utils/graph.h"
#include "../utils/graphio.h"

__global__ void compute_d(double* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	id = id * size + id;
	
	// Make assumption here that graph is connected and every node has degree atleast 1.
	deg[id] = sqrt(1/deg[id]); 
}

__global__ void compute_s(double* S, double* X, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] += X[id]; 
}

__global__ void transform_s(double* S, int volume, int window_size, int b, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] = (S[id] * float(volume))/ ((float) window_size * (float) b); 
}

__global__ void transform_m(double* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	M[id] =logf(M[id] > 1?M[id]:1);
}

__global__ void sqrt_si(double* S, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;
	
	S[id] = sqrt((float) S[id]);
}


int main ( void ){

	/**************
 	* NetMF small *
	**************/
	
	/* Load graph */
	Graph g =  read_graph("../data/test/small_test.csv","edgelist");

	/* CUDA housekeeping */
	dim3 threads(128);
	dim3 grid((int)ceil((float)g.size/128));

	/* cuBlas housekeeping */	
	cublasHandle_t handle;
	cublasCreate(&handle);
	double al=1.0f;
	double bet=1.0f;

	/* Initialize and allocate variables */
	// HOST
	double *X;
	double *S;
	double *M;
	
	int window_size = 10;
	int size = g.size * g.size * sizeof(double);
	int b = 1;
	int dimension = 2;
	
	X = (double *)malloc(size);
	S = (double *)malloc(size);
	M = (double *)malloc(size);	

	// DEVICE
	double *D_device;
	double *temp_device, *temp1_device, *X_device;
	double *A_device;
	double *S_device;
	double *M_device;

	cudaMalloc(&D_device, size);
	cudaMalloc(&A_device, size);
	cudaMalloc(&X_device, size);
	cudaMalloc(&temp_device, size);
	cudaMalloc(&temp1_device, size);
	cudaMalloc(&S_device, size);
	cudaMalloc(&M_device, size);

	cudaMemset(A_device, 0, size);
	cudaMemset(D_device, 0, size);
	cudaMemset(X_device, 0, size);
	cudaMemset(S_device, 0, size);
	cudaMemset(M_device, 0, size);
	cudaMemset(temp_device, 0, size);
	cudaMemset(temp1_device, 0, size);

	/* Copy necessary variables to device */
	cudaMemcpy(D_device, g.degree, size, cudaMemcpyHostToDevice);	
	cudaMemcpy(A_device, g.adj, size , cudaMemcpyHostToDevice);	

	/* Compute D = D^{-1/2} */
	compute_d<<<grid, threads>>>(D_device, g.size);
	cudaDeviceSynchronize();

	/* Compute X = D^{-1/2}AD^{-1/2} */

	cublasDgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, g.size,
		    &al,
	            A_device,g.size, 
		    D_device, g.size,
		    &bet, 
		    temp_device, g.size);
	
	cublasDgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, g.size,
		    &al,
		    D_device, g.size,
	            temp_device,g.size, 
		    &bet, 
		    X_device, g.size);
	
	/* Compute S = sum(X^{0}....X^{window_size}) */
	
	// This might be too slow. Experiment to see if you can use a custom kernel 
	cudaMemcpy(S_device, X_device, size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(temp_device, X_device, size, cudaMemcpyDeviceToDevice);

	for(int i=2;i<=window_size;i++){
		cublasDgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    g.size, g.size, g.size,
		    &al,
		    X_device, g.size,
	            temp_device,g.size, 
		    &bet, 
		    temp1_device, g.size);
		
		// Use cublas addition functions
		compute_s<<<grid, threads>>>(S_device, temp1_device, g.size);
		cudaMemcpy(temp_device, temp1_device, size, cudaMemcpyDeviceToDevice);
		cudaMemset(temp1_device,0,size);
	}

	// Compute S = S * (vol / (window_size * b))
	transform_s<<<grid,threads>>>(S_device,g.volume, window_size, b, g.size);
	cudaMemcpy(S, S_device, size, cudaMemcpyDeviceToHost);
	
	std::cout<<std::endl<<std::endl;
	for(int i=0;i<g.size;i++){
		for(int j=0;j<g.size;j++){
			std::cout<<S[i*g.size + j]<<" ";
		}
		std::cout<<std::endl;
	}
	
	// Compute M = D^{-1/2} * S * D^{-1/2}
	cudaMemset(temp_device, 0, size); 

	cublasDgemm(handle, 
	    CUBLAS_OP_N, CUBLAS_OP_N, 
	    g.size, g.size, g.size,
	    &al,
	    S_device, g.size,
	    D_device,g.size, 
	    &bet, 
	    temp_device, g.size);

	cublasDgemm(handle, 
	    CUBLAS_OP_N, CUBLAS_OP_N, 
	    g.size, g.size, g.size,
	    &al,
	    D_device, g.size,
	    temp_device,g.size, 
	    &bet, 
	    M_device, g.size);
		
	// Compute M = log(max(Mi,1))
	transform_m<<<grid,threads>>>(M_device, g.size);
	cudaMemcpy(M, M_device, size, cudaMemcpyDeviceToHost);

	// Perform SVD on M
	double *U, *VT, *Si;
	U = (double*)malloc(size);
	VT = (double*)malloc(size);
	Si = (double*)malloc(g.size * sizeof(double));

	double *U_device, *VT_device, *Si_device;
	double *W_device; //auxillary device array

	cusolverDnHandle_t cusolverH;


	cudaMalloc(&U_device, size);
	cudaMalloc(&Si_device, g.size * sizeof(double));
	cudaMalloc(&VT_device, size);
	cudaMalloc(&W_device, size);

	int lwork = 0;
	double *d_work, *d_rwork;

	cusolverDnDgesvd_bufferSize(cusolverH,g.size, g.size,&lwork);
	cudaMalloc(&d_work, sizeof(double) * lwork);

	signed char jobu = 'A';
	signed char jobvt = 'A';
	int *devInfo;

	cusolverDnDgesvd(cusolverH, jobu, jobvt, 
			g.size, g.size, A_device, 
			g.size, Si_device, 
			U_device, g.size, 
			VT_device, g.size, 
			d_work, lwork, d_rwork, devInfo); 
	
	cudaDeviceSynchronize();
	
	// TODO: Clip vector to be of dimension D.

	sqrt_si<<<grid, threads>>>(Si_device, g.size);	
	cublasDdgmm(handle, 
	    CUBLAS_SIDE_LEFT, 
	    g.size, g.size,
	    U_device, g.size,
	    Si_device,1, 
	    W_device,  g.size);
	
	
	/***********
	* Clean up *
	***********/

	//free(g);

	//cudaFree(D_device);

	// Function to print matrix
/*
	std::cout<<std::endl<<std::endl;
	for(int i=0;i<g.size;i++){
		for(int j=0;j<g.size;j++){
			std::cout<<S[i*g.size + j]<<" ";
		}
		std::cout<<std::endl;
	}
*/
}
