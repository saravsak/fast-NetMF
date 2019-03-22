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
				g.degree[i * g.size + i] = 1.00;
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
			for(int j=0;j<g.size;j++){
				g.degree[i * g.size + j] = 1.00 / sqrt(g.degree[i * g.size + j]);
			}
		}
	}

	/* Compute X = D' * A * D' */
	log("Computing X' = D' * A");
	DT *X_ = (DT *) mkl_malloc( g.size * g.size * sizeof(DT *), 64);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			g.size, g.size, g.size,
			1.00, g.degree, g.size,
			g.adj, g.size,
			0.00, X_, g.size );		

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
