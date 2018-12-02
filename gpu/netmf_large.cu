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

	/* cuSolver housekeeping */
	log("Setting up cuSolver");	
	int lwork = 0;
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	float *d_work, *d_rwork;
	int *devInfo;
	
	cudaMalloc(&devInfo, sizeof(int));
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);

	/* Initialize and allocate variables */
	log("Setting up host variables");
	float *X;
	
	int window_size = 10;
	profile.window_size = window_size;

	int size = g.size * g.size * sizeof(float);
	int b = 1;
	int dimension = 2;
	profile.dimension = dimension;
	int *devInfoH;
	
	X = (float *)malloc(size);
	devInfoH = (int *)malloc(sizeof(int));

	log("Setting up device variables");
	float *D_device;
	float *A_device;
	float *temp_device;
	float *X_device;
	float *M_device;
	float *U_device, *VT_device, *Si_device;

	cudaMalloc(&U_device, size);
	cudaMalloc(&Si_device, g.size * sizeof(float));
	cudaMalloc(&VT_device, size);
	cudaMalloc(&D_device, size);
	cudaMalloc(&A_device, size);
	cudaMalloc(&X_device, size);
	cudaMalloc(&temp_device, size);
	cudaMalloc(&M_device, size);

	cudaMemset(A_device, 0, size);
	cudaMemset(D_device, 0, size);
	cudaMemset(X_device, 0, size);
	cudaMemset(temp_device, 0, size);

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
	float *V;
	float *W;
	float *e;
	float *E;
	int rank = 2;
	
	V = (float *) malloc(size);
	W = (float *) malloc(size);
	e = (float *) malloc(rank * sizeof(float));
	E = (float *) malloc(rank * g.size * sizeof(float));

	float *W_device;
	float *e_device;
	float *E_device;

	cudaMalloc((void **)&W_device, sizeof(float) * g.size);
	cudaMalloc((void **)&e_device, sizeof(float) * rank);
	cudaMalloc((void **)&E_device, sizeof(float) * g.size * rank);

	cusolverDnSsyevd_bufferSize(cusolverH,jobz, uplo, g.size, X_device, g.size, W_device, &lwork);
	cudaMalloc(&d_work, sizeof(float) * lwork);

	cusolverDnSsyevd(cusolverH, 
			jobz, uplo, g.size, 
			X_device, g.size, 
			W_device, d_work, 
			lwork, devInfo);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(W, W_device, g.size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(devInfoH, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	log("Filtering eigenvalues and eigen vectors");
	filter_e<<<grid, threads>>>(W_device, e_device, g.size, window_size, rank);
	
	filter_E<<<grid, threads>>>(X_device, E_device, g.size, rank);	
	//cudaMemcpy(E_device, X_device + ptr, rank * g.size * sizeof(float), cudaMemcpyDeviceToDevice);

	float *D;
	D = (float *)malloc(g.size * g.size * sizeof(float));

	cudaMemcpy(E,E_device, rank*g.size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(D, D_device, g.size * g.size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(e,e_device, rank*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(X,X_device, g.size * g.size*sizeof(float), cudaMemcpyDeviceToHost);

	std::cout<<std::endl<<std::endl<<"Value of D"<<std::endl;
	print_matrix(D, g.size);

	std::cout<<"Value of EV";
	print_matrix(X, g.size);

	std::cout<<std::endl<<std::endl<<"Value of E"<<std::endl;
	for(int i=0;i<rank;i++){
		for(int j=0;j<g.size;j++){
			std::cout<<E[i*g.size + j]<<" ";
		}
		std::cout<<std::endl;
	}		

	std::cout<<std::endl<<std::endl<<"Value of e"<<std::endl;
	for(int i=0;i<g.size;i++){
		std::cout<<e[i]<<" ";
	}

	cudaMemset(temp_device, 0, g.size * g.size * sizeof(float));

	cublasSdgmm(handle,
		    CUBLAS_SIDE_LEFT,
		    g.size, rank,
		    E_device, g.size,
		    D_device, g.size + 1,
		    temp_device, g.size);

	std::cout<<"Value of D_rt_invU"<<std::endl;
	cudaMemcpy(W, temp_device, g.size * rank * sizeof(float), cudaMemcpyDeviceToHost);	

	std::cout<<std::endl<<std::endl;
	for(int i=0;i<g.size;i++){
		for(int j=0; j<g.size; j++){
			std::cout<<W[i*g.size+j]<<" ";
		}
		std::cout<<'\n';
	}	

	cublasSdgmm(handle,
		    CUBLAS_SIDE_RIGHT,
		    g.size, rank,
		    temp_device, g.size,
		    e_device, 1,
		    M_device, g.size);

	float *M = (float *)malloc( g.size * rank * sizeof(float));
	
	std::cout<<"evals * d_rt_invU"<<std::endl;
	cudaMemcpy(M, M_device, g.size * rank * sizeof(float), cudaMemcpyDeviceToHost);

	print_matrix(M, g.size);

	//TODO: Perform MMT here - mmT = T.dot(m, m.T) * (vol/b)

	float *MMT, *MMT_device;
	cudaMalloc(&MMT_device, size);
	cudaMemset(MMT_device, 0, size);
	MMT = (float *) malloc(size);
	memset(MMT, 0, size);
	cublasSgemm(handle, 
		    CUBLAS_OP_N, CUBLAS_OP_T, 
		    g.size, g.size, g.size,
		    &al,
	            M_device,g.size, 
		    M_device, g.size,
		    &bet, 
		    MMT_device, g.size);

	const float scale = float(g.volume)/float(b);
	cudaDeviceSynchronize();	
	cublasSscal(handle, g.size * g.size,
			&scale,
			MMT_device, 
			1);

	cudaMemcpy(MMT, MMT_device, size, cudaMemcpyDeviceToHost);
	std::cout<<"Value of MMT";
	print_matrix(MMT, g.size);
	
	signed char jobu = 'A';
	signed char jobvt = 'N';
	
	float *Embedding, *Embedding_device;

	cudaMalloc(&Embedding_device, g.size * dimension *sizeof(float));
	Embedding = (float *)malloc(g.size * dimension * sizeof(float));
	
	transform_m<<<grid,threads>>>(M_device, g.size);
	
	cusolverDnSgesvd(cusolverH, jobu, jobvt, 
			g.size, g.size, MMT_device, g.size, 
			Si_device, 
			U_device, g.size, 
			VT_device, g.size, 
			d_work, 
			lwork, 
			d_rwork, 
			devInfo); 
		
	sqrt_si<<<grid, threads>>>(Si_device, dimension);	
	cublasSdgmm(handle, 
	    CUBLAS_SIDE_RIGHT, 
	    g.size, dimension,
	    U_device, g.size,
	    Si_device,1, 
	    Embedding_device, g.size);
		
        cudaDeviceSynchronize();

	

	cudaMemcpy(Embedding, Embedding_device, size, cudaMemcpyDeviceToHost);


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
