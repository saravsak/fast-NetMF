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
__global__ void preprocess_laplacian(DT* adj, DT *degree, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= size) return;

	// Remove self loops
	// If deg(v) = 0 -> deg(0) = 1
		
	if(degree[id*size + id] == 0){
			degree[id*size + id] = 1.00;
			adj[id*size + id] = 1.00;	
	}else{
			adj[id*size + id] = 0.0;
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

	if(id >= size) return;
	
	DT mem = S[id];
	
	S[id] = mem * val; 
}

__global__ void prune_m(DT* M, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;

	if(M[id] < 1)
		M[id] = 1;

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

int main ( int argc, char** argv  ){
	/**************
 	* NetMF small *
	**************/
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
        Clock::time_point begin, end;
        Clock::time_point overall_begin, overall_end;
	info profile; 
	profile.dataset = argv[1];
	profile.algo = "small-sparse";
	/* Section 0: Preliminaries */

	overall_begin = Clock::now();

	/* Settings */
	int window_size = std::atoi(argv[2]);
	int dimension = std::atoi(argv[3]);
	int b = std::atoi(argv[4]);

	/* Load graph */
        log("Reading data from file");
	
	begin = Clock::now(); 
	Graph g =  read_graph(argv[5],"csr", argv[7]);

	cudaMalloc(&g.degree_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	cudaMalloc(&g.degree_csr.d_colIndices, (g.degree_csr.nnz) * sizeof(int));
	cudaMalloc(&g.degree_csr.d_values, (g.degree_csr.nnz) * sizeof(DT));

	cudaMalloc(&g.adj_csr.d_rowIndices, (g.size + 1) * sizeof(int));
	cudaMalloc(&g.adj_csr.d_colIndices, (g.adj_csr.nnz) * sizeof(int));
	cudaMalloc(&g.adj_csr.d_values, (g.adj_csr.nnz) * sizeof(DT));

	end = Clock::now();

	profile.iptime = std::chrono::duration_cast<milliseconds>(end - begin);

	begin = Clock::now();
	/* CUDA housekeeping */
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

	host2device(&g.degree_csr, g.degree_csr.nnz, g.size);
	host2device(&g.adj_csr, g.adj_csr.nnz, g.size);

	if(DEBUG){
		device2host(&g.degree_csr, g.degree_csr.nnz, g.size);	
		device2host(&g.adj_csr, g.adj_csr.nnz, g.size);	
	
		std::cout<<"Sample"<<g.adj_csr.h_values[0];	
	
		DT sum = 0;
		for(int i=0;i<g.degree_csr.nnz;i++)
			sum+=g.degree_csr.h_values[i];
		//int sum = std::accumulate(g.degree_csr.h_values, g.degree_csr.h_values + g.degree_csr.nnz);
		std::cout<<"Sum: "<<sum<<std::endl;
		if(VERBOSE){
			print_csr(
    				g.size,
    				g.degree_csr.nnz,
    				g.degree_csr,
    				"Degree matrix");
		}
	}

	log("Completed conversion of data from dense to sparse");
	end = Clock::now();
	profile.gpuio = std::chrono::duration_cast<milliseconds>(end - begin);

	/* Section 2: Compute X = D^{-1/2} * A * D^{-1/2} */
	/* Procedure
	   1. Compute D' = D^{-1/2}
	   2. Compute X' = D' * A
	   3. Compute X = X' * D'
	*/
	
	/* Step 1: Compute D' = D^{-1/2} */
	begin = Clock::now();
	log("Computing normalized D");
	compute_d<<<grids, threads>>>(g.degree_csr.d_values, g.degree_csr.nnz);
	cudaDeviceSynchronize();
	end = Clock::now();
	profile.init = std::chrono::duration_cast<milliseconds>(end - begin);


	log("Computed normalized D");
	if(DEBUG){
		device2host(&g.degree_csr, g.degree_csr.nnz, g.size);
		float sum = 0;

		for(int i=0;i<g.degree_csr.nnz;i++)
			sum+=g.degree_csr.h_values[i];

		std::cout<<"Sum of normalized degree matrix"<<sum<<std::endl;

		std::sort(g.degree_csr.h_values, g.degree_csr.h_values + g.degree_csr.nnz);
		if(VERBOSE){
			print_csr(
				g.size,
				g.degree_csr.nnz,
				g.degree_csr,
				"Normalized Degree Matrix");
		}
	}
			
	begin = Clock::now();
	/* Step 2: Compute X' = D' * A */
	log("Computing X' = D' * A");
	csr X_temp;
	multiply_csr(&g.degree_csr, &g.adj_csr, &X_temp, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&X_temp, X_temp.nnz, g.size);

		int nz = 0;
		int nnz = 0;
		
		for(int i=0;i<X_temp.nnz;i++){
			if(X_temp.h_values[i] == 0)
				nz +=1;
			else
				nnz +=1;
		}	
		std::cout<<"# zeros: "<<nz<<std::endl;
		std::cout<<"# nonzeros: "<<nnz<<std::endl;

		

		if(VERBOSE)
		{	print_csr(g.size, X_temp.nnz, X_temp, "X' = D' * A");}
	}
	

	/* Step 3: Compute X = X' * D */
	log("Computing X = X' * D");
	csr X;
	multiply_csr(&X_temp, &g.degree_csr, &X, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&X, X.nnz, g.size);
		std::cout<<"Min of X: "<<*std::min_element(X.h_values, X.h_values + X.nnz)<<std::endl;
		std::cout<<"Max of X: "<<*std::max_element(X.h_values, X.h_values + X.nnz)<<std::endl;
		int x_nz = 0;
		int x_nnz = 0;
		for(int i=0;i<X.nnz;i++){
			if(X.h_values[i] == 0)
				x_nz +=1;
			else
				x_nnz +=1;
		}	
		std::cout<<"# zeros: "<<x_nz<<std::endl;
		std::cout<<"# nonzeros: "<<x_nnz<<std::endl;
		if(VERBOSE)
			{print_csr(g.size, X.nnz, X, "X = X' * A");}
	}
	end = Clock::now();
	profile.compute_x = std::chrono::duration_cast<milliseconds>(end - begin);

	begin = Clock::now();
	/* Section 3: Compute S = sum(X^{0}....X^{window_size}) */
	/* Procedure
	  1. Copy X to S
	  2. Copy X to temp
	  3. temp' = temp * X
	  4. S' = S + temp'  
	  5. temp = temp'
	  6. S = S'
	*/
	
	/* Step 0: Declare all variables */
	csr S;
	csr temp;
	csr temp_;
	csr S_;
	
	/* Step 1: Copy X to S */
	log("Copying X to S");
	allocate_csr(&S, X.nnz, g.size);
	copy_csr(&X, &S, g.size);

	/* Step 2: Copy X to temp */
	log("Copying X to temp");
	allocate_csr(&temp, X.nnz, g.size);
	copy_csr(&X, &temp, g.size);

	for(int i=2;i<=window_size;i++){
		/* Step 3: temp' = temp * X */
		log("Computing temp' = temp * X");
		multiply_csr(&temp, &X, &temp_, g.size, g.size, g.size, cusparse_handle, mat_descr);


		/* Step 4: S = S + temp */
		log("Computing S = S + temp");
		add_csr(&S, &temp_, &S_, g.size, g.size, cusparse_handle, mat_descr);

		/* Step 5: temp = temp' */
		log("Copying temp' to temp");
		free_csr(&temp);
		allocate_csr(&temp, temp_.nnz, g.size);
		copy_csr(&temp_, &temp, g.size);
		free_csr(&temp_);

		/* Step 6: S = S' */
		log("Copying S' to S");
		free_csr(&S);
		allocate_csr(&S, S_.nnz, g.size);
		copy_csr(&S_, &S, g.size);
		free_csr(&S_);
		
		if(DEBUG){
			device2host(&temp, temp.nnz, g.size);
			device2host(&temp, temp.nnz, g.size);
			//std::sort(temp_.h_values, temp_.h_values + temp_.nnz);
			std::cout<<"Min: "<<*std::min_element(temp.h_values, temp.h_values + temp.nnz)<<std::endl;
			std::cout<<"Max: "<<*std::max_element(temp.h_values, temp.h_values + temp.nnz)<<std::endl;
			device2host(&S, S.nnz, g.size);
			device2host(&S, S.nnz, g.size);
			//std::sort(temp_.h_values, temp_.h_values + temp_.nnz);
			std::cout<<"Min: "<<*std::min_element(S.h_values, S.h_values + S.nnz)<<std::endl;
			std::cout<<"Max: "<<*std::max_element(S.h_values, S.h_values + S.nnz)<<std::endl;
		}
	}

	if(DEBUG){
		device2host(&S, S.nnz, g.size);
		if(VERBOSE){
			print_csr(g.size,
					S.nnz,
					S,
					"Objective matrix");
		}
	}

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

		
	cublasSscal(cublas_handle, S.nnz,
                    &val,
                    S.d_values, 1);
	cudaDeviceSynchronize();
	
	if(DEBUG){
		device2host(&S, S.nnz, g.size);
	
		std::cout<<"Min of S: "<<*std::min_element(S.h_values, S.h_values + S.nnz);
		std::cout<<"Max of S: "<<*std::max_element(S.h_values, S.h_values + S.nnz);

		if(VERBOSE){
			print_csr(g.size,
					S.nnz,
					S,
					"Transformed Objective Matrix");
		}
	}

	end = Clock::now();
	profile.compute_s = std::chrono::duration_cast<milliseconds>(end - begin);
	log("Computing M");

	begin = Clock::now();

        /* Section 5: Compute M = D^{-1/2} * S * D^{-1/2} */
	/* Procedure
	   1. Compute M' = D' * S
	   2. Compute M = M' * D'
	*/

	/* Step 1: Compute M' = D' * S */
	log("Computing M' = D' * S");
	csr M_;
	multiply_csr(&g.degree_csr, &S, &M_, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&M_, M_.nnz, g.size);
		if(VERBOSE)
		{	print_csr(g.size, M_.nnz, M_, "M' = D' * M");}
	}
	

	/* Step 3: Compute X = X' * D' */
	log("Computing M = M' * D");
	csr M;
	multiply_csr(&M_, &g.degree_csr, &M, g.size, g.size, g.size, cusparse_handle, mat_descr);
	
	if(DEBUG){
		device2host(&M, M.nnz, g.size);
		if(VERBOSE)
			{print_csr(g.size, M.nnz, M, "M = M' * D");}
	}

	/* Section 6: Compute M'' = log(max(M,1)) */
	
	/* Procedure 
	   1. Remove all negative elements from M
	   2. Compute buffer size
	   3. Comput number of non-zeros
	   4. Apply threshold & prune
	   5. Compute log M
	*/

	/* Step 1: Removing all negative elements */
	log("Pruning M");

        if(DEBUG){
                device2host(&M, M.nnz, g.size);

                int num_ones = 0;
                int num_l = 0;
                int num_g = 0;

                for(int i=0;i<M.nnz;i++){
                        if(M.h_values[i] < 1){
                                num_l++;
                        }else if(M.h_values[i] > 1){
                                num_g++;
                        }else{
                                num_ones++;
                        }
                }

                std::cout<<"1s: "<<num_ones<<std::endl;
                std::cout<<"<1s: "<<num_l<<std::endl;
                std::cout<<">1s: "<<num_g<<std::endl;

                if(VERBOSE)
                        {print_csr(g.size, M.nnz, M, "M = M' * D");}
        }

	prune_m<<<grids,threads>>>(M.d_values, M.nnz);
	//cudaDeviceSynchronize();
	//log("Prunied M");

	/* TODO: Move this to GPU DEBUG THIS ASAP */
        //device2host(&M, M.nnz, g.size);
        //for(int i=0;i<M.nnz;i++){
        //	if(M.h_values[i] < 1){
	//		M.h_values[i] = 1;
        //        }
	//	M.h_values[i] = log(M.h_values[i]);
	//}

	//host2device(&M, M.nnz, g.size);


        if(DEBUG){
               // device2host(&M, M.nnz, g.size);

                int num_ones = 0;
                int num_l = 0;
                int num_g = 0;

                for(int i=0;i<M.nnz;i++){
                        if(M.h_values[i] < 0){
                                num_l++;
                        }else if(M.h_values[i] > 0){
                                num_g++;
                        }else{
                                num_ones++;
                        }
                }

                std::cout<<"1s: "<<num_ones<<std::endl;
                std::cout<<"<1s: "<<num_l<<std::endl;
                std::cout<<">1s: "<<num_g<<std::endl;

                if(VERBOSE)
                        {print_csr(g.size, M.nnz, M, "M = M' * D");}
        }

	
	/* Step 2: Compute buffersize */
	log("Computing buffer size for pruning");
	DT threshold = 0.00;

	csr M_cap;
	allocate_csr_row(&M_cap, g.size);
	size_t lworkInBytes = 0;
	char *d_work=NULL;
	cusparseSpruneCsr2csr_bufferSizeExt(cusparse_handle,
        					g.size,g.size,
        					M.nnz, mat_descr,
        					M.d_values,M.d_rowIndices,M.d_colIndices,
        					&threshold,
        					mat_descr,
						M_cap.d_values, M_cap.d_rowIndices, M_cap.d_colIndices,
        					&lworkInBytes);	
	cudaMalloc((void**)&d_work, lworkInBytes);

	if(DEBUG){
    		printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
	}

	/* Step 3: Compute NNZ */
	log("Computing Number of non zeros for pruned matrix");
	cusparseSpruneCsr2csrNnz(cusparse_handle,
					g.size,g.size,
        				M.nnz,mat_descr,
  					M.d_values,M.d_rowIndices,M.d_colIndices,
        				&threshold,
        				mat_descr,
        				M_cap.d_rowIndices, &M_cap.nnz, /* host */
        				d_work);

    	if (0 == M_cap.nnz ){
        	printf("C is empty \n");
        	return 0;
    	}

	if(DEBUG){
		printf("nnzC = %d\n", M_cap.nnz);
	}


	/* Step 4: Convert CSR matrix to CSR matrix */
	
	allocate_csr_col_val(&M_cap, M_cap.nnz);
	cusparseSpruneCsr2csr(cusparse_handle,
   				g.size,g.size,
        			M.nnz, mat_descr,
        			M.d_values, M.d_rowIndices, M.d_colIndices,
        			&threshold,
        			mat_descr,
        			M_cap.d_values,M_cap.d_rowIndices, M_cap.d_colIndices,
        			d_work);


	device2host(&M_cap, M_cap.nnz, g.size);
	std::cout<<"NNZ in M_cap: "<<M_cap.nnz<<std::endl;
	if(VERBOSE){
		print_csr(g.size,
				M_cap.nnz,
				M_cap, 
				"M cap"
			 );
	}
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
	MKL_INT k = 0;

	DT *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;

	E_mkl = (DT *)mkl_malloc(k0 * sizeof(DT), 128);
	K_L_mkl = (DT *)mkl_malloc( k0*mkl_rows*sizeof( DT), 128 );
        K_R_mkl = (DT *)mkl_malloc( k0*mkl_cols*sizeof( DT), 128 );
        res_mkl = (DT *)mkl_malloc( k0*sizeof( DT), 128 );

	memset(E_mkl, 0 , k0 * sizeof(DT));
	memset(K_L_mkl, 0 , k0 * sizeof(DT));
	memset(K_R_mkl, 0 , k0 * sizeof(DT));
	memset(res_mkl, 0 , k0 * sizeof(DT));

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

	if(mkl_status){
		std::cout<<"SVD failed "<<mkl_status<<std::endl;
		exit(0);	
	}

	if(DEBUG){
	std::cout<<"Number of singular found: "<<k<<std::endl;
	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";
	}

	DT *U_device, *Si_device;
	//DT *U_host;
	DT *Si_host;
	DT *E_device, *E_host;

	cudaMalloc(&U_device, g.size * dimension * sizeof(DT));
	cudaMalloc(&E_device, g.size * dimension * sizeof(DT));
	cudaMalloc(&Si_device, dimension * sizeof(DT));

	//U_host = (DT *) malloc(g.size * dimension * sizeof(DT));
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
