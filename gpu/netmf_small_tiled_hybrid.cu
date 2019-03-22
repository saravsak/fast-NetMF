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

	/* Read graph */
	log("Reading data from file");
	Graph g = read_graph(arg_input, "edgelist", arg_mapping);
	log("Successfully read data from file");

	/* Preprocess graph */
	log("Preprocessing graph");

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<g.size;i++){
			if(g.degree1D[i] == 0){
				g.degree1D[i * g.size + i] = 1.00;
				g.adj[i * g.size + i] = 1.00;
			}else{
				g.adj[i * g.size + i] = 1.00;
			}
		}
	}	

	/* Compute D' = D^{-1/2} */
	log("Computing normalized degree matrix");

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<g.size;i++){
			g.degree1D[i] = 1.00 / sqrt(g.degree1D[i]);
		}
	}

	/* Compute X = D' * A * D' */
	log("Computing X' = D' * A");
	DT *X_ = (DT *) malloc(g.size * g.size * sizeof(DT));

	

	/* Compute S = sum(X^{0}....X^{window_size}) */

	/* Compute S = S * (vol / (window_size * b)) */

	/* Compute M = D^{-1/2} * S * D^{-1/2} */

	/* Compute M'' = log(max(M,1)) */

	/* Convert M'' to sparse */

	/* Compute SVD of M'' */

	/* Transform singular values */

	/* Generate Embeddings */

	/* Save Embeddings */

	

}
