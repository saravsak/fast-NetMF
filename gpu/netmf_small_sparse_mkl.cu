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

#include "omp.h"

#define NUM_STREAMS 4
#define TILE_SIZE 2

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void make_tile(DT* A, DT *B, int om, int on, int m, int n, int row_offset, int col_offset){
        int row = 0, col = 0;
        row_offset = row_offset * m;
        col_offset = col_offset * n;

	#pragma omp parallel
	{
		#pragma for collapse(2)
        	for(int i=0;i<m;i++){
                	row = row_offset + i;
                	for(int j=0;j<n;j++){
                        	col = col_offset + j;
                        	B[i + j * m] = A[row + col * om];
                	}
        	}
	}	
}

void copy_tile(DT* A, DT *B, int om, int on, int m, int n, int row_offset, int col_offset){

        int row = 0, col = 0;
        row_offset = row_offset * m;
        col_offset = col_offset * n;

	#pragma omp parallel
	{
		#pragma for collapse(2)
        	for(int i=0;i<m;i++){
        	        row = row_offset + i;
        	        for(int j=0;j<n;j++){
        	                col = col_offset + j;
        	                A[row + col * om] = B[i + j * m];
        	        }
        	}
	}
}

void tiled_dgmm(DT *diag, DT *A, DT *C, int m, int n){

	cudaStream_t streams[NUM_STREAMS];

	for(int i=0; i<NUM_STREAMS;i++){
		cudaStreamCreate(&streams[i]);
	}

	DT *tile_d[NUM_STREAMS];
	DT *tile_h[NUM_STREAMS];
	
	DT *diag_d[NUM_STREAMS];
	DT *diag_h[NUM_STREAMS];

	DT *res_d[NUM_STREAMS];
	DT *res_h[NUM_STREAMS];
	
	int num_tiles = ceil(((float) m)/(TILE_SIZE));

	log("Check 1");

        for(int i=0;i<NUM_STREAMS;i++){
                gpuErrchk(cudaMalloc(&tile_d[i],
                                        TILE_SIZE * TILE_SIZE * sizeof(DT)));
                gpuErrchk(cudaMalloc(&res_d[i],
                                        TILE_SIZE * TILE_SIZE * sizeof(DT)));
                gpuErrchk(cudaMalloc(&diag_d[i],
                                        TILE_SIZE * sizeof(DT)));

		gpuErrchk(cudaHostAlloc(&tile_h[i],
                                TILE_SIZE*TILE_SIZE*sizeof(DT),
                                cudaHostAllocDefault));
		gpuErrchk(cudaHostAlloc(&res_h[i],
                                TILE_SIZE*TILE_SIZE*sizeof(DT),
                                cudaHostAllocDefault));
		gpuErrchk(cudaHostAlloc(&diag_h[i],
                                TILE_SIZE*sizeof(DT),
                                cudaHostAllocDefault));
        }
	
	log("Check 2");

	cublasHandle_t cublasH;	

	int row=0,col=0;

	while(row < num_tiles){
		col = 0;
		while(col < num_tiles){
			int ret_col = col;
			for(int i=0;i<NUM_STREAMS;i++){
				if(col > num_tiles){
					break;
				}
				std::cout<<"Working on block: "<<row * col<<std::endl;
				make_tile(A, tile_h[(row * col) % NUM_STREAMS], m, n, TILE_SIZE, TILE_SIZE, row, col);
				log("Check 3");
					
				gpuErrchk(cudaMemcpyAsync(tile_d[(row * col) % NUM_STREAMS],
							tile_h[(row * col) % NUM_STREAMS],
							TILE_SIZE * TILE_SIZE * sizeof(DT),
							cudaMemcpyHostToDevice,
							streams[i]));
				gpuErrchk(cudaMemcpyAsync(diag_d[row % NUM_STREAMS],
							diag_h + (row * TILE_SIZE),
							TILE_SIZE * sizeof(DT),
							cudaMemcpyHostToDevice,
							streams[i]));
				log("Check 4");
				cublasSetStream(cublasH, streams[i]);
				
				log("Check 5");
				cublasSdgmm(cublasH, CUBLAS_SIDE_LEFT, m, n,
					tile_d[(row*col) % NUM_STREAMS], m,
					diag_d[row%NUM_STREAMS],1,
					res_d[(row*col) % NUM_STREAMS],m);
				log("Check 6");
				col++;
			}

			for(int i=0;i<NUM_STREAMS;i++){
				if(ret_col > num_tiles){
					break;
				}
				cudaStreamSynchronize(streams[i]);

				gpuErrchk(cudaMemcpyAsync(res_h[(row * ret_col) % NUM_STREAMS],
							res_d[(row * ret_col) % NUM_STREAMS],
							TILE_SIZE * TILE_SIZE * sizeof(DT),
							cudaMemcpyHostToDevice,
							streams[i]));
				copy_tile(C, res_h[(row*ret_col) % NUM_STREAMS],m,n, TILE_SIZE, TILE_SIZE, row, ret_col);
				ret_col++;

			}	
		}
		row++;
	}


}

void mkl_sparse_multiply(csr *A, csr *B, csr *C,int  m, int n, int k){
	sparse_matrix_t A_handle, B_handle, C_handle;

	mkl_sparse_s_create_csr(&A_handle, SPARSE_INDEX_BASE_ZERO,
			m, k,
			A->h_rowIndices, A->h_rowEndIndices,
			A->h_colIndices, A->h_values);

	mkl_sparse_s_create_csr(&B_handle, SPARSE_INDEX_BASE_ZERO,
			k, n,
			B->h_rowIndices, B->h_rowEndIndices,
			B->h_colIndices, B->h_values);

	mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
			A_handle, B_handle, &C_handle);

	sparse_index_base_t indexing;
	int rows, cols;

	mkl_sparse_s_export_csr(C_handle,
		       &indexing, &rows, &cols, 
		       &C->h_rowIndices, &C->h_rowEndIndices,
		       &C->h_colIndices, &C->h_values);

}

//void copy_mkl_csr(csr *A, csr *B, int rows){
//	std::memcpy(B->h_rowIndices, A->h_rowIndices, rows * sizeof(int));
//	std::memcpy(B->h_rowEndIndices, A->h_rowEndIndices, rows * sizeof(int));
//	std::memcpy(B->h_colIndices, A->h_colIndices, A.nnz * sizeof(int));
//	std::memcpy(B->h_values, A->h_values, A.nnz * sizeof(DT));
//}

void allocate_mkl_csr(csr *A,int nnz, int rows){
	A->h_rowIndices	= (int *) malloc(rows * sizeof(int));
	A->h_rowEndIndices = (int *) malloc(rows * sizeof(int));
	A->h_colIndices = (int *) malloc(nnz * sizeof(int));
	A->h_values = (DT *) malloc(nnz * sizeof(DT));
}

//struct matrix_descr{
//	sparse_matrix_type_t type;
//	sparse_fill_mode_t mode;
//	sparse_diag_type_t diag;
//}mkl_descr;	


int main(int argc, char *argv[]){
	/********************
        * NetMF small tiled *
        *********************/
        /* Argument order 
        1. Dataset name
        2. Window Size
        3. Dimension
        4. B
        5. Input
        6. Output
        7. Mapping file
        */

	/* Parse arguments */
	char *arg_dataset = argv[1];
	char *arg_window = argv[2];
	char *arg_dimension = argv[3];
	char *arg_b = argv[4];
	char *arg_input = argv[5];
	char *arg_output = argv[6];
	char *arg_mapping = argv[7];

	/* Assign arguments */
	int window_size = std::atoi(arg_window);
	int b = std::atoi(arg_b);
	int dimension = std::atoi(arg_dimension);

	/* Read graph */
	log("Reading data from file");
	Graph g = read_graph(arg_input, "mkl-sparse", arg_mapping);
	log("Successfully read data from file");

//	/* Preprocess graph */
//	//log("Preprocessing graph");
//
//	//#pragma omp parallel
//	//{
//	//	#pragma omp for
//	//	for(int i=0;i<g.size;i++){
//	//		if(g.degree[i * g.size + i] == 0){
//	//			g.degree[i * g.size + i] = 1.00;
//	//			g.adj[i * g.size + i] = 1.00;
//	//		}else{
//	//			g.adj[i * g.size + i] = 0.00;
//	//		}
//	//	}
//	//}	
//
	/* Compute D' = D^{-1/2} */
	log("Computing normalized degree matrix");

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<g.size;i++){
			g.degree_csr.h_values[i] = 1.00 / sqrt(g.degree_csr.h_values[i]);
		}
	}

	/* Compute X = D' * A * D' */
	log("Computing X' = D' * A");
	csr X_;
	mkl_sparse_multiply(&g.degree_csr, &g.adj_csr, &X_, g.size, g.size, g.size);

	log("Computing X = X' * D");
	csr X;
	mkl_sparse_multiply(&X_, &g.degree_csr, &X, g.size, g.size, g.size);

	/* Compute S = sum(X^{0}....X^{window_size}) */
	log("Computing S");
	DT *X_pow = (DT *)malloc(g.size * g.size * sizeof(DT));
	memset(X_pow, 0, g.size * g.size * sizeof(DT));

	MKL_INT cols = g.size;

	for(int i=0;i<g.size;i++){
		for(int j=X.h_rowIndices[i]; j<X.h_rowEndIndices[i]; j++){
			X_pow[i * cols + X.h_colIndices[j]] = X.h_values[j];		
		}
	}

	DT *S = (DT *) malloc(g.size * g.size * sizeof(DT));
	DT *X_temp = (DT *)malloc(g.size * g.size * sizeof(DT));

	std::memcpy(S, X_pow, g.size * g.size * sizeof(DT));

	sparse_matrix_t X_handle;
	mkl_sparse_s_create_csr(&X_handle, SPARSE_INDEX_BASE_ZERO, 
			g.size, g.size, X.h_rowIndices, X.h_rowEndIndices, X.h_colIndices, X.h_values);	

	matrix_descr mkl_descr;
	mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
	mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
	

	for(int i=2;i<=window_size;i++){
		std::cout<<"Calculating "<<i<<"th power"<<std::endl;
	
		mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
				1.00, X_handle, mkl_descr,
				SPARSE_LAYOUT_ROW_MAJOR, X_pow,
				g.size, g.size, 
				0.00, X_temp, g.size);
		std::memcpy(X_pow, X_temp, g.size * g.size * sizeof(DT));
	
		vsAdd(g.size * g.size, S, X_pow, S);	

	}

	/* Compute S = S * (vol / (window_size * b)) */
	log("Transforming S");
	DT val = (g.volume / (window_size * b));
	cblas_sscal(g.size * g.size, val, S, 1);

	/* Compute M = D^{-1/2} * S * D^{-1/2} */
	log("Computing M");
	DT *M_ = (DT *) malloc(g.size * g.size * sizeof(DT));
	DT *M = (DT *) malloc(g.size * g.size * sizeof(DT)); 

	sparse_matrix_t degree_handle;
	mkl_sparse_s_create_csr(&degree_handle, SPARSE_INDEX_BASE_ZERO,
			g.size, g.size,
			g.degree_csr.h_rowIndices, g.degree_csr.h_rowEndIndices,
			g.degree_csr.h_colIndices, g.degree_csr.h_values);

	mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
			1.00, degree_handle, mkl_descr,
			SPARSE_LAYOUT_ROW_MAJOR, S,
			g.size, g.size, 
			0.00, M_, g.size);

	// Nice hack:  (A*B)' = B'*A'
	mkl_sparse_s_mm(SPARSE_OPERATION_TRANSPOSE,
			1.00, degree_handle, mkl_descr,
			SPARSE_LAYOUT_COLUMN_MAJOR, M_,
			g.size, g.size, 
			0.00, M, g.size);
	
	free(M_);
	free(S);

	/* Compute M'' = log(max(M,1)) */
	log("Filtering M");
	
		for(int i=0;i<g.size * g.size;i++){
			if(M[i]<=1){
				M[i] = 0;
			}else{
				M[i] = log(M[i]);
			}
		}


	/* Convert M'' to sparse */
	log("Converting M to sparse");
	int nnz = 0;
		for(int i=0;i<g.size * g.size; i++){
			if(M[i]!=0)
				nnz++;
		}
	std::cout<<"# Non zero in M_Cap "<<nnz<<std::endl;
	sparse_matrix_t M_cap;

	MKL_INT rows = g.size;
	cols = g.size;

	//TODO: How to set alignment?
	MKL_INT *rows_start = (MKL_INT *) mkl_malloc(rows * sizeof(MKL_INT), 64);
	MKL_INT *rows_end = (MKL_INT *) mkl_malloc(rows * sizeof(MKL_INT), 64);
	MKL_INT *col_idx = (MKL_INT *) mkl_malloc(nnz * sizeof(MKL_INT), 64);
	DT *vals = (DT *) mkl_malloc(nnz * sizeof(DT), 64);

	//TODO: Optimize: Highly inefficient
	int idx = 0;

	// Cannot be parallel since idx increaments serially
		for(int i=0;i<rows;i++){
			rows_start[i] = idx;
			for(int j=0;j<cols;j++){
				if(M[i*cols + j]!=0){
					col_idx[idx] = j;
					vals[idx] = M[i*cols+j];
					idx++;				
				}	
			}
			rows_end[i] = idx;
		}

	mkl_sparse_s_create_csr(&M_cap, SPARSE_INDEX_BASE_ZERO, 
			rows, cols, rows_start, rows_end, col_idx, vals);	

	/* Compute SVD of M'' */
	log("Making space for SVD");

	char whichS = 'L';
	char whichV = 'L';

	MKL_INT pm[128];
	mkl_sparse_ee_init(pm);

	matrix_descr descrM;
	descrM.type = SPARSE_MATRIX_TYPE_GENERAL;
	descrM.mode = SPARSE_FILL_MODE_UPPER; 
	descrM.diag = SPARSE_DIAG_NON_UNIT;

	MKL_INT k0 = dimension;
	MKL_INT k;

	DT *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;

        E_mkl = (DT *)mkl_malloc(k0 * sizeof(DT), 128);
        K_L_mkl = (DT *)mkl_malloc( k0*rows*sizeof( DT), 128 );
        K_R_mkl = (DT *)mkl_malloc( k0*cols*sizeof( DT), 128 );
        res_mkl = (DT *)mkl_malloc( k0*sizeof( DT), 128 );

        memset(E_mkl, 0 , k0);
        memset(K_L_mkl, 0 , k0);
        memset(K_R_mkl, 0 , k0);
        memset(res_mkl, 0 , k0);

	int mkl_status = 0;

	log("Computing SVD via MKL");
        mkl_status = mkl_sparse_s_svd(&whichS, &whichV, pm,
                        M_cap, descrM,
                        k0, &k,
                        E_mkl,
                        K_L_mkl,
                        K_R_mkl,
                        res_mkl);

	if(mkl_status){
		std::cout<<"STD failed with status: "<<mkl_status<<std::endl;
		exit(0);
	}

	std::cout<<"Number of singular found: "<<k<<std::endl;
	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";

	/* Transform singular values */
	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<dimension;i++){
			E_mkl[i] = sqrt(E_mkl[i]);
		}
	}

	/* Generate Embeddings */
	DT *ev = (DT *)mkl_malloc(dimension * dimension * sizeof(DT), 64);

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<dimension;i++){
			ev[i*dimension + i] = E_mkl[i];
		}
	}

	DT *embeddings = (DT *)mkl_malloc(dimension * g.size * sizeof(DT), 64);

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
        		g.size, dimension, dimension, 1.0, K_L_mkl, g.size, ev, dimension, 0.00, embeddings, g.size);	
	//mkl_simatcopy('R', 'T', g.size, dimension, 1.00, embeddings, dimension, g.size);
	/* Save Embeddings */
	write_embeddings(arg_output, embeddings, g.size, dimension);
	

}
