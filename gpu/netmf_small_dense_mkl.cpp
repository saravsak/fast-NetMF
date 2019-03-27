#include<stdlib.h>
#include<iostream>
#include<time.h>
#include<chrono>
#include<algorithm>
#include<numeric>
#include<math.h>

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

	/* Assign arguments */
	int window_size = std::atoi(arg_window);
	int b = std::atoi(arg_b);
	int dimension = std::atoi(arg_dimension);

	/* Read graph */
	log("Reading data from file");
	Graph g = read_graph(arg_input, "mkl-dense", arg_mapping);
	log("Successfully read data from file");

	/* Preprocess graph */
	//log("Preprocessing graph");

	//#pragma omp parallel
	//{
	//	#pragma omp for
	//	for(int i=0;i<g.size;i++){
	//		if(g.degree[i * g.size + i] == 0){
	//			g.degree[i * g.size + i] = 1.00;
	//			g.adj[i * g.size + i] = 1.00;
	//		}else{
	//			g.adj[i * g.size + i] = 0.00;
	//		}
	//	}
	//}	

	/* Compute D' = D^{-1/2} */
	log("Computing normalized degree matrix");

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<g.size;i++){
			g.degree_mkl[i * g.size + i] = 1.00 / sqrt(g.degree_mkl[i* g.size + i]);
		}
	}

	/* Compute X = D' * A * D' */
	log("Computing X' = D' * A");
	DT *X_ = (DT *) malloc(g.size * g.size * sizeof(DT));
	DT *X = (DT *) malloc(g.size * g.size * sizeof(DT));

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        		g.size, g.size, g.size, 1.0, g.degree_mkl, g.size, g.adj_mkl, g.size, 0.00, X_, g.size);	
	log("Computing X = X' * D");
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        		g.size, g.size, g.size, 1.0, X_, g.size, g.degree_mkl, g.size, 0.00, X, g.size);	

	free(g.adj_mkl);

	/* Compute S = sum(X^{0}....X^{window_size}) */
	log("Computing S");
	DT *S = (DT *) malloc(g.size * g.size * sizeof(DT));
	std::memcpy(X_, X, g.size * g.size * sizeof(DT));
	std::memcpy(S, X, g.size * g.size * sizeof(DT));

	DT *W = (DT *) malloc(g.size * g.size * sizeof(DT));

	for(int i=2;i<=window_size;i++){
		std::cout<<"Calculating "<<i<<"th power"<<std::endl;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        		g.size, g.size, g.size, 1.0, X_, g.size, X, g.size, 0.00, W, g.size);
		vsAdd(g.size * g.size, W, S, S);
		std::memcpy(X, W, g.size * g.size * sizeof(DT));
	}

	/* Compute S = S * (vol / (window_size * b)) */
	log("Transforming S");
	DT val = (g.volume / (window_size * b));
	cblas_sscal(g.size * g.size, val, S, 1);

	/* Compute M = D^{-1/2} * S * D^{-1/2} */
	log("Computing M");
	DT *M_ = (DT *) malloc(g.size * g.size * sizeof(DT));
	DT *M = (DT *) malloc(g.size * g.size * sizeof(DT)); 
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        	g.size, g.size, g.size, 1.0, g.degree_mkl, g.size, S, g.size, 0.00, M_, g.size);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        	g.size, g.size, g.size, 1.0, M_, g.size, g.degree_mkl, g.size, 0.00, M, g.size);
	
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
	MKL_INT cols = g.size;

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
