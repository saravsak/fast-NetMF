/* 
TODO: 
1. Change thread architecture
2. Use cuBlas for addition
3. Async copy to device
*/

#include<stdlib.h>
#include<iostream>
#include<time.h>
#include<chrono>
#include<algorithm>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>

#include "../utils/graph.h"
#include "../utils/io.h"

#include "../lib/RedSVD-h"
#include<Eigen/Core>
#include<Eigen/Dense>

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
	int dimension = 3;
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

	cudaMemcpy(M, M_device, size, cudaMemcpyDeviceToHost);


	log("Printing M_CAP");
	print_matrix(M, g.size);

	log("Creating Eigen matrix");
	Eigen::MatrixXf M_eigen = Eigen::Map<Eigen::MatrixXf>(M, g.size, g.size);

	log("Performing SVD of M");
	REDSVD::RedSVD<Eigen::MatrixXf> svd(M_eigen, dimension);


	log("Sorting singular values");
	 std::vector<unsigned long long int > sorted_indexes(dimension);
	 std::size_t n(0);
	 std::generate(std::begin(sorted_indexes), 
			std::end(sorted_indexes), 
			[&]{ return n++; });

	 std::sort(  std::begin(sorted_indexes), 
	             std::end(sorted_indexes),
	             [&](unsigned long long int  i1, unsigned long long int  i2) { 
				return svd.singularValues()[i1] > svd.singularValues()[i2]; 
			}
		);

	// cout << "\nsort_indexes are \n----------------------------- \n";
	//  for (auto v : sorted_indexes)
	//          std::cout << v << ' ';


	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,int> perm(dimension);	
	for (unsigned long long int  i = 0; i< dimension; i++)
		perm.indices()[i] = sorted_indexes[i];	

	
	std::cout << "\nSingular Values and U are\n------------------------\n";
	std::cout << "\nSingular Values are " << std::endl << svd.singularValues().transpose() * perm << std::endl ;
	std::cout << "\nComputed U  is " << std::endl << (svd.matrixU() * perm).transpose() << std::endl ;

//	cusolverDnSgesvd(cusolverH, jobu, jobvt, 
//			g.size, g.size, M_device, g.size, 
//			Si_device, 
//			U_device, g.size, 
//			VT_device, g.size, 
//			d_work, 
//			lwork, 
//			d_rwork, 
//			devInfo); 
	
//        cudaDeviceSynchronize();
	log("Moving singular values to host");
	Eigen::Map<Eigen::MatrixXf>(Si, 1, dimension) = svd.singularValues().transpose() * perm;
	Eigen::Map<Eigen::MatrixXf>(U, g.size, dimension) = (svd.matrixU() * perm).transpose();

	log("Moving singular values to device");
	cudaMemcpy(Si_device, Si, dimension * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(U_device, U, g.size * dimension * sizeof(float), cudaMemcpyHostToDevice);	

	end = Clock::now();
	profile.svd = std::chrono::duration_cast<milliseconds>(end - begin);;	
	
	begin = Clock::now();
	log("Transforming Si");
	sqrt_si<<<grid, threads>>>(Si_device, dimension);	
        cudaDeviceSynchronize();

	cudaMemcpy(Si, Si_device, dimension * sizeof(float), cudaMemcpyDeviceToHost);
	log("Printing sqrt S");
	for(int i=0;i<dimension;i++)
		std::cout<<Si[i]<<" ";

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
