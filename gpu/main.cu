/* 
TODO: 
1. Change thread architecture
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

__global__ void compute_d(float* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	id = id * size + id;
	
	// Make assumption here that graph is connected and every node has degree atleast 1.
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
	
	/* Load graph */
        std::cout<<"Reading data from file"<<std::endl;
	Graph g =  read_graph("../data/blogcatalog/edges.csv","edgelist");

        std::cout<<"Initializing"<<std::endl;
	/* CUDA housekeeping */
	dim3 threads(128);
	dim3 grid((int)ceil((float)g.size/128));

	/* cuBlas housekeeping */	
	cublasHandle_t handle;
	cublasCreate(&handle);
	float al=1.0f;
	float bet=1.0f;

	/* cuSolver housekeepinh */
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
	// HOST
	float *U, *VT, *Si, *W;
	float *X;
	float *S;
	float *M;
	
	int window_size = 10;
	int size = g.size * g.size * sizeof(float);
	int b = 1;
	int dimension = 128;
	int *devInfoH;
	
	X = (float *)malloc(size);
	S = (float *)malloc(size);
	M = (float *)malloc(size);	
	U = (float*)malloc(size);
	VT = (float*)malloc(size);
	W = (float*)malloc(size);
	Si = (float*)malloc(g.size * sizeof(float));
	devInfoH = (int *)malloc(sizeof(int));

	// DEVICE
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

	cudaMalloc(&U_device, size);
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

	/* Copy necessary variables to device */
	std::cout<<"Moving data to device"<<std::endl;
	cudaMemcpy(D_device, g.degree, size, cudaMemcpyHostToDevice);	
	cudaMemcpy(A_device, g.adj, size , cudaMemcpyHostToDevice);	

	/* Compute D = D^{-1/2} */
	std::cout<<"Computing normalized D"<<std::endl;
	compute_d<<<grid, threads>>>(D_device, g.size);
	cudaDeviceSynchronize();

	/* Compute X = D^{-1/2}AD^{-1/2} */
	std::cout<<"Computing X"<<std::endl;
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
	
	/* Compute S = sum(X^{0}....X^{window_size}) */
	
	// This might be too slow. Experiment to see if you can use a custom kernel 
	//cudaMemcpy(X, X_device, size, cudaMemcpyDeviceToHost);
	//print_matrix(X, g.size);

	cudaMemcpy(S_device, X_device, size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(temp_device, X_device, size, cudaMemcpyDeviceToDevice);

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
		
		// Use cublas addition functions
		compute_s<<<grid, threads>>>(S_device, temp1_device, g.size);
		//cudaMemcpy(temp_device, temp1_device, size, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		cudaMemset(temp1_device,0,size);
	}

	// Compute S = S * (vol / (window_size * b))
	std::cout<<"Transforming S"<<std::endl;
	transform_s<<<grid,threads>>>(S_device,g.volume, window_size, b, g.size);
	//cudaMemcpy(S, S_device, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        //print_matrix(S, g.size);
	
	// Compute M = D^{-1/2} * S * D^{-1/2}
	cudaMemset(temp_device, 0, size); 

	std::cout<<"Computing M"<<std::endl;
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
	std::cout<<"Transforming M"<<std::endl;
	transform_m<<<grid,threads>>>(M_device, g.size);
        cudaDeviceSynchronize();

	// Do need to transpose M, since M is symmetric matrix
	
	// Perform SVD on M
	std::cout<<"SVD"<<std::endl;
	cusolverDnSgesvd(cusolverH, jobu, jobvt, 
			g.size, g.size, M_device, g.size, 
			Si_device, 
			U_device, g.size, 
			VT_device, g.size, 
			d_work, 
			lwork, 
			d_rwork, 
			devInfo); 
	
	//cudaMemcpy(U, U_device, size, cudaMemcpyDeviceToHost);
	//print_matrix(U, g.size);	
	//cudaMemcpy(VT, VT_device, size, cudaMemcpyDeviceToHost);
	//print_matrix(VT, g.size);	
        cudaDeviceSynchronize();
	
	//cudaMemcpy(Si, Si_device, sizeof(float) * g.size, cudaMemcpyDeviceToHost);
	//std::cout<<std::endl<<std::endl;
	//for(int i=0;i<g.size;i++){
	//	std::cout<<Si[i]<<" ";
	//}

	//cudaMemcpy(devInfoH, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	//std::cout<<"\nDev Info:"<<*devInfoH<<std::endl;

	//std::cout<<std::endl<<std::endl;
	
	std::cout<<"Transforming Si"<<std::endl;
	sqrt_si<<<grid, threads>>>(Si_device, dimension);	
        cudaDeviceSynchronize();

	std::cout<<"Computing W"<<std::endl;
	cublasSdgmm(handle, 
	    CUBLAS_SIDE_LEFT, 
	    g.size, dimension,
	    U_device, g.size,
	    Si_device,1, 
	    W_device, g.size);
        cudaDeviceSynchronize();

	cudaMemcpy(W, W_device, size, cudaMemcpyDeviceToHost);
	
	std::cout<<std::endl<<std::endl;
	//for(int i=0;i<dimension;i++){
	//	for(int j=0;j<g.size;j++){
	//		std::cout<<W[i*g.size + j]<<" ";
	//	}
	//	std::cout<<std::endl;
	//}
	
	std::cout<<"Done"<<std::endl<<std::endl;
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
