// Questionf for prof
// 1. Do we need to store in column major order for cuBlas
// 2. Do we need to copy everytime or once is enough
// 3. Ask if DAD calculation is correct

# include<stdlib.h>
# include<cuda_runtime.h>
# include<cublas_v2.h>

#include "../utils/graph.h"
#include "../utils/graphio.h"

#include<iostream>

__global__ void compute_d(double* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;


	if(id >= size) return;

	id = id * size + id;
	
	// Make assumption here that graph is connected and every node has degree atleast 1.
	deg[id] = sqrt(1/deg[id]); 
}

int main ( void ){

	/**************
 	* NetMF small *
	**************/

	/* Load graph */
	Graph g =  read_graph("../data/test/small_test.csv","edgelist");
	std::cout<<"Printing i/p graph"<<std::endl;
	g.print_graph();
	std::cout<<std::endl;
	std::cout<<std::endl;
	std::cout<<"Printing degree matrix"<<std::endl;
	g.print_degree();


	// Initialize all host and device variables
	double *D_device;
	double *temp_device, *X, *X_device;
	double *A_device;
	
	X = (double *)malloc(g.size * g.size * sizeof(double));
	
	cudaMalloc(&D_device, g.size * g.size * sizeof(double));
	cudaMalloc(&X_device, g.size * g.size * sizeof(double));
	cudaMalloc(&A_device, g.size * g.size * sizeof(double));
	cudaMalloc(&temp_device, g.size * g.size * sizeof(double));

	cudaMemset(X_device, 0, g.size * g.size * sizeof(double));
	cudaMemset(temp_device, 0, g.size * g.size * sizeof(double));

	/* Compute D = D^{-1/2} */
	// Create and allocate device variable to hold degree matrix

	// Copy degree matrix to device
	cudaMemcpy(D_device, g.degree, g.size * g.size * sizeof(double), cudaMemcpyHostToDevice);	
	
	// Call GPU Kernel

	std::cout<<"number of grids: "<<(int)ceil((float)g.size/128)<<std::endl;	

	dim3 threads(128);
	dim3 grid((int)ceil((float)g.size/128));
	compute_d<<<grid, threads>>>(D_device, g.size);

	// Copy from device to host
	cudaMemcpy(g.degree, D_device, g.size * g.size * sizeof(double), cudaMemcpyDeviceToHost);

	// Print results
	std::cout<<std::endl;
	std::cout<<std::endl;
	std::cout<<"Printing D^{-1/2}"<<std::endl;
	g.print_degree();

	/* Compute X = D^{-1/2}AD^{-1/2} */
	// Declare and allocate X matrix


	// Copy A and D to device 
	cudaMemcpy(D_device, g.degree, g.size * g.size * sizeof(double), cudaMemcpyHostToDevice);	
	cudaMemcpy(A_device, g.adj, g.size * g.size * sizeof(double), cudaMemcpyHostToDevice);	
	
	// Compute product on GPU
	cublasHandle_t handle;

	cublasCreate(&handle);
	double al=1.0f;
	double bet=1.0f;
	int size = g.size;

	/* NOTE: cuBlas takes matrices in col major order */
	/* So instead of X = D^{-1/2}AD^{-1/2} we do      */
	/* X = AD^{-1/2} -> X = X             */

	cublasDgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    size, size, size,
		    &al,
	            A_device,size, 
		    D_device, size,
		    &bet, 
		    temp_device, size);
	
	cublasDgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_N, 
		    size, size, size,
		    &al,
		    D_device, size,
	            temp_device,size, 
		    &bet, 
		    X_device, size);

	// Copy X to host
	cudaMemcpy(X, X_device, g.size * g.size * sizeof(double), cudaMemcpyDeviceToHost);
	
	// Print X
	std::cout<<std::endl;
	std::cout<<std::endl;
	std::cout<<"Printing X"<<std::endl;
	for(int i=0;i<g.size;i++){
		for(int j=0;j<g.size;j++){
			std::cout<<X[i*g.size + j]<<" ";
		}
		std::cout<<std::endl;
	}
	
	/* Compute S = sum(X^{0}....X^{window_size}) */
		
	// Compute S = S * (vol / (window_size * b))
	
	// Compute M = D^{-1/2} * S * D^{-1/2}
	
	// Compute M = log(max(Mi,1))

	/*****************
 	*  CUBLAS SAMPLE *
	*****************/

//	cudaError_t cudaStat;
//	cublasStatus_t stat;
//	cublasHandle_t handle;
//	int j;
//	float *x;
//	x = (float*) malloc(n*sizeof(*x));
//	for(j=0;j<n;j++)
//		x[j] = (float) j;
//
//	float *d_x;
//
//	cudaStat = cudaMalloc((void**)&d_x, n*sizeof(*x));
//
//	stat = cublasCreate(&handle);
//
//	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
//	int result;
//
//	stat = cublasIsamax(handle, n, d_x, 1, &result);
//
//	std::cout<<x[result-1]<<std::endl;
//
//	cudaFree(d_x);
//	cublasDestroy(handle);
//	free(x);
//	return EXIT_SUCCESS;
//

	/***********
	* Clean up *
	***********/

	//free(g);

	//cudaFree(D_device);
}
