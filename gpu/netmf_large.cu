/* 
TODO: 
1. Change thread architecture
2. Use cuBlas for addition
3. Async copy to device
*/
/*
1. Questions for prof.
In kernel filter_E, is it possible to copy without a kernel?
*/
#include<stdlib.h>
#include<iostream>
#include<time.h>
#include<chrono>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>

#include "../utils/graph.h"
#include "../utils/io.h"

__global__ void compute_d(float* deg, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	id = id * size + id;
	
	/* Make assumption here that graph is  	*/
	/* connected and every node has degree 	*/
        /* atleast 1. 		       		*/

	deg[id] = sqrt(1/deg[id]); 
}

// Make this kernel 2D
__global__ void compute_s(float* S, float* X, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] += X[id]; 
}

// Make this kernel 2D
__global__ void transform_s(float* S, int volume, int window_size, int b, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size * size) return;
	
	S[id] = (S[id] * float(volume))/ ((float) window_size * (float) b); 
}

//Make this kernel 2D
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

__global__ void filter_e(float *W, float *e, int size, int window_size, int rank){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= rank) return;

	float el = W[size - rank + id];
	float val = 0;

	if(el >= 1)
		e[id] = 1;
	else{
	      val = (el * (1 - powf(el, window_size))) / ((1-el) * window_size);
	      e[id] = 0 > val ? 0: val;
	}
	e[id] = sqrt(e[id]);

}

// Make this kernel 2D
__global__ void filter_E(float *X, float *E, int size, int rank){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id>=rank) return;
	
	for(int i=0;i<size;i++){
		E[id * size + i] = X[(size - rank + id) * size + i];
	}
		
	
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
 	* NetMF large *
	**************/

	/* General housekeeping */
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
        Clock::time_point begin, end;
	info profile; 
	profile.dataset = "blogcatalog";
	profile.algo = "large";

	/* Load graph */
        log("Reading data from file");

        begin = Clock::now(); 
	Graph g =  read_graph("../data/blogcatalog/edges.csv","edgelist");
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

	/* cuSolver housekeeping */
	log("Setting up cuSolver");	
	int lwork = 0;
	float *d_work, *d_rwork;
	int *devInfo;
	signed char jobu = 'A';
	signed char jobvt = 'N';
	
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cudaMalloc(&devInfo, sizeof(int));
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);

	/* Initialize and allocate variables */
	log("Setting up host variables");
	int window_size = 10;
	profile.window_size = window_size;
	int b = 1;
	int rank = 2;
	int dimension = 2;
	const float scale = float(g.volume)/float(b);
	int size = g.size * g.size * sizeof(float);
	profile.dimension = dimension;
	
	float *X;
	float *MMT;
	float *Embedding;

	X = (float *)malloc(size);
	MMT = (float *) malloc(size);
	Embedding = (float *)malloc(g.size * dimension * sizeof(float));
	
	memset(MMT, 0, size);
	memset(X, 0, size);
	memset(Embedding, 0, g.size * dimension *sizeof(float));
	
	log("Setting up device variables");
	float *D_device;
	float *A_device;
	float *temp_device;
	float *X_device;
	float *M_device;
	float *U_device, *VT_device, *Si_device;
	float *W_device;
	float *e_device;
	float *E_device;
	float *MMT_device;
	float *Embedding_device;
	
	cudaMalloc((void **)&Embedding_device, g.size * dimension *sizeof(float));
	cudaMalloc((void **)&MMT_device, size);
	cudaMalloc((void **)&W_device, sizeof(float) * g.size);
	cudaMalloc((void **)&e_device, sizeof(float) * rank);
	cudaMalloc((void **)&E_device, sizeof(float) * g.size * rank);
	cudaMalloc((void**)&U_device, size);
	cudaMalloc((void**)&Si_device, g.size * sizeof(float));
	cudaMalloc((void**)&VT_device, size);
	cudaMalloc((void**)&D_device, size);
	cudaMalloc((void**)&A_device, size);
	cudaMalloc((void**)&X_device, size);
	cudaMalloc((void**)&temp_device, size);
	cudaMalloc((void**)&M_device, size);

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

	/* Eigen decomposition of X */
	log("Eigen Decomposition of X");

	cusolverDnSsyevd_bufferSize(cusolverH,jobz, uplo, g.size, X_device, g.size, W_device, &lwork);
	cudaMalloc(&d_work, sizeof(float) * lwork);

	begin = Clock::now();
	cusolverDnSsyevd(cusolverH, 
			jobz, uplo, g.size, 
			X_device, g.size, 
			W_device, d_work, 
			lwork, devInfo);
	
	cudaDeviceSynchronize();
	end = Clock::now();
	profile.compute_s = std::chrono::duration_cast<milliseconds>(end - begin);;	

	begin = Clock::now();
	log("Filtering eigenvalues and eigen vectors");
	filter_e<<<grid, threads>>>(W_device, e_device, g.size, window_size, rank);
	cudaDeviceSynchronize();	
	
	filter_E<<<grid, threads>>>(X_device, E_device, g.size, rank);	
	cudaDeviceSynchronize();	

	cudaMemset(temp_device, 0, g.size * g.size * sizeof(float));
	
	cublasSdgmm(handle,
		    CUBLAS_SIDE_LEFT,
		    g.size, rank,
		    E_device, g.size,
		    D_device, g.size + 1,
		    temp_device, g.size);
	cudaDeviceSynchronize();	

	cublasSdgmm(handle,
		    CUBLAS_SIDE_RIGHT,
		    g.size, rank,
		    temp_device, g.size,
		    e_device, 1,
		    M_device, g.size);
	cudaDeviceSynchronize();	

	cublasSgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_T, 
		    g.size, g.size, g.size,
		    &al,
	            M_device,g.size, 
		    M_device, g.size,
		    &bet, 
		    MMT_device, g.size);

	cudaDeviceSynchronize();	
	cublasSscal(handle, g.size * g.size,
			&scale,
			MMT_device, 
			1);
	cudaDeviceSynchronize();	
	
	transform_m<<<grid,threads>>>(MMT_device, g.size);
	cudaDeviceSynchronize();	
	end = Clock::now();
	profile.compute_m = std::chrono::duration_cast<milliseconds>(end - begin);	

	begin = Clock::now();	
	cusolverDnSgesvd(cusolverH, jobu, jobvt, 
			g.size, g.size, MMT_device, g.size, 
			Si_device, 
			U_device, g.size, 
			VT_device, g.size, 
			d_work, 
			lwork, 
			d_rwork, 
			devInfo); 
	cudaDeviceSynchronize();	
	end = Clock::now();
	profile.svd = std::chrono::duration_cast<milliseconds>(end - begin);	

	begin = Clock::now();
	sqrt_si<<<grid, threads>>>(Si_device, dimension);
	cudaDeviceSynchronize();	
	cublasSdgmm(handle, 
	    CUBLAS_SIDE_RIGHT, 
	    g.size, dimension,
	    U_device, g.size,
	    Si_device,-1, 
	    Embedding_device, g.size);
		
        cudaDeviceSynchronize();
	

	cudaMemcpy(Embedding, Embedding_device,sizeof(float)* g.size * dimension, cudaMemcpyDeviceToHost);
	end = Clock::now();
	profile.emb = std::chrono::duration_cast<milliseconds>(end - begin);	

	write_embeddings("blogcatalog.emb",Embedding, g.size, dimension);	
	write_profile("profile.txt", profile);		
	log("Done");
	/***********
	* Clean up *
	***********/

	free(X);

	// DEVICE
	cudaFree(D_device);
	cudaFree(A_device);
	cudaFree(X_device);
	cudaFree(temp_device); 

	cudaFree(d_work);
	cudaFree(d_rwork);
	cudaFree(devInfo);

}
