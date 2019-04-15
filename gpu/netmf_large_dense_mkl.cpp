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

void print_matrix(DT* S, int size){
	std::cout<<std::endl<<std::endl;
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			std::cout<<S[i*size + j]<<" ";
		}
		std::cout<<std::endl;
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

int main ( int argc, char **argv ){
	/**************
 	* NetMF small *
	**************/
	/* Argument order 
	1. Dataset name
	2. Window Size
	3. Dimension
	4. B
	5. Input
	6. Output
	7. Mapping file
	8. Eigen rank
	*/
	/* Parse arguments */
	char *arg_dataset = argv[1];
	char *arg_window = argv[2];
	char *arg_dimension = argv[3];
	char *arg_b = argv[4];
	char *arg_input = argv[5];
	char *arg_output = argv[6];
	char *arg_mapping = argv[7];
	char *arg_rank = argv[8];
	
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
        Clock::time_point begin, end;
        Clock::time_point overall_begin, overall_end;
	info profile; 
	profile.dataset = arg_dataset;
	profile.algo = "large-mkl";
	/* Section 0: Preliminaries */

	/* Settings */
	int window_size = std::atoi(arg_window);
	int dimension = std::atoi(arg_dimension);
	int b = std::atoi(arg_b);
	int rank = std::atoi(arg_rank);

	profile.window_size = window_size;
	profile.dimension = dimension;

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

	log("Eigen(X)");
	sparse_matrix_t X_handle;
	mkl_sparse_s_create_csr(&X_handle, SPARSE_INDEX_BASE_ZERO, 
			g.size, g.size, X.h_rowIndices, X.h_rowEndIndices, X.h_colIndices, X.h_values);	

	char which = 'L';
	
	MKL_INT k;

	MKL_INT pm[128];
	mkl_sparse_ee_init(pm);

	matrix_descr mkl_descr;
	mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	mkl_descr.mode = SPARSE_FILL_MODE_UPPER;
	mkl_descr.diag = SPARSE_DIAG_NON_UNIT;

	DT *evals = (DT *) malloc(rank * sizeof(DT));
	DT *EV = (DT *) malloc(g.size * rank * sizeof(DT));
	DT *res = (DT *) malloc(rank * sizeof(DT));
	
	sparse_status_t mkl_status;

	mkl_status = mkl_sparse_s_ev(&which, pm, 
			X_handle, mkl_descr, rank, &k, evals, EV, res);

	if(mkl_status != SPARSE_STATUS_SUCCESS){
		std::cout<<"Sparse eigen failed with status: "<<mkl_status<<std::endl;
	}

	if(DEBUG){
		for(int i=0;i<rank;i++) std::cout<<evals[i]<<" "; std::cout<<'\n';
	}

	log("Computed eigen");

	log("Computing D_rt_invU = D * EV * D");	

	DT *D_rt_inv = (DT *)malloc(g.size * rank * sizeof(DT));
	DT *D_rt_invU = (DT *)malloc(g.size * rank * sizeof(DT));

	sparse_matrix_t degree_handle;
	mkl_sparse_s_create_csr(&degree_handle, SPARSE_INDEX_BASE_ZERO,
			g.size, g.size,
			g.degree_csr.h_rowIndices, g.degree_csr.h_rowEndIndices,
			g.degree_csr.h_colIndices, g.degree_csr.h_values);

	mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
			1.00, degree_handle, mkl_descr,
			SPARSE_LAYOUT_COLUMN_MAJOR, EV,
			rank, g.size, 
			0.00, D_rt_invU, g.size);

	log("Filtering eigenvalues");
	for(int i=0;i<rank;i++){
		if(evals[i] >= 1){
			evals[i] = 1;
		}else{
			evals[i] = (evals[i] * (1 - pow(evals[i], window_size))) / ((1-evals[i]) * window_size);
		}
	}

	for(int i=0;i<rank;i++){
		std::cout<<evals[i]<<" ";
	}

	DT *diag_ev = (DT *)malloc(rank * rank * sizeof(DT));

	for(int i=0;i<rank;i++){
		diag_ev[i*rank + i] = evals[i];
	}

	DT *M = (DT *) malloc(g.size * rank * sizeof(DT));	

	cblas_sgemm(CblasNoTrans, CblasNoTrans,
			g.size, rank, rank,
			1.00, D_rt_invU, rank,
			diag_ev, rank,
			0.00, M, rank);

	cblas_sgemm(CblasNoTrans, CblasTrans,
			g.size, g.size, rank,
			1.00, M, rank,
			M, rank,
			0.00, M, g.size);
	


//	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
//		g.size, rank,
//		EV_device, g.size, 
//		e_device, 1,
//		EV_device, g.size);
//	cudaDeviceSynchronize();
//		
//	/*REMOVE*/
//	//cudaMemcpy(EV_host, EV_device, g.size * rank * sizeof(DT), cudaMemcpyDeviceToHost);
//	//for(int i=0;i<g.size * rank;i++){
//	//	std::cout<<"EV "<<i<<":"<<EV_host[i]<<std::endl;
//	//}	
//
//
//	log("Computing M");
//
//        /* Section 5: Compute M = (EV_device * EV_device.T) * (vol/b) */
//
//	DT *M_device;
//	cudaMalloc(&M_device, g.size * g.size * sizeof(DT));
//
//	DT alf = 1.00;
//	DT beta = 0.00;
//
//	cublasSgemm(cublas_handle,
//                           CUBLAS_OP_N, CUBLAS_OP_T,
//                           g.size, g.size, rank,
//                           &alf,
//                           EV_device, g.size,
//                           EV_device, g.size,
//                           &beta,
//			   M_device, g.size
//                           );
//	cudaDeviceSynchronize();
//
//	/*REMOVE*/
//	//std::cout<<std::endl;
//	//DT *M_host;
//	//M_host = (DT *) malloc(sizeof(DT) * g.size * g.size);
//	//cudaMemcpy(M_host, M_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToHost);
//	//for(int i=0;i<g.size * g.size;i++){
//	//	std::cout<<"M "<<i<<":"<<M_host[i]<<std::endl;
//	//}
//
//	
//	DT val = ((DT) g.volume) / ((DT) b);
//	cublasSscal(cublas_handle, g.size * g.size,
//			&val,
//			M_device, 1);
//	
//	/*REMOVE*/
//	//std::cout<<std::endl;
//	//cudaMemcpy(M_host, M_device, g.size * g.size * sizeof(DT), cudaMemcpyDeviceToHost);
//	//for(int i=0;i<g.size * g.size;i++){
//	//	std::cout<<"M "<<i<<":"<<M_host[i]<<std::endl;
//	//}
//
//
//	/* Section 6: Compute M'' = log(max(M,1)) */
//	
//	/* Procedure 
//	   1. Prune M and take log
//	   2. Create CSR struct for M''
//	   3. Compute nnzPerVector for M''
//	*/
//
//	/* Step 1: Prune M and take log */
//	log("Pruning M");
//
//	prune_m<<<grids,threads>>>(M_device, g.size);
//       	cudaDeviceSynchronize(); 
//
//	log("Pruned M");
//
//	/* Step 2: Create CSR struct for both matrices */
//	log("Converting dense matrix to CSR format");	
//	csr M_cap;    /* Variable to hold adjacency matrix in CSR format */
//
//	M_cap.nnz = 0; /* Initialize number of non zeros in adjacency matrix */
//
//	/* Step 6: Compute nnz/row of dense matrix */	
//	log("Computing nnzPerVector for M''");
//
//	cudaMalloc(&M_cap.d_nnzPerVector, 
//			g.size * sizeof(int));
//	cusparseSnnz(cusparse_handle, 
//			CUSPARSE_DIRECTION_ROW, 
//			g.size, g.size, 
//			mat_descr, 
//			M_device, LDA, 
//			M_cap.d_nnzPerVector, &M_cap.nnz);
//	M_cap.h_nnzPerVector = (int *)malloc(g.size * sizeof(int));
//	cudaMemcpy(M_cap.h_nnzPerVector, 
//			M_cap.d_nnzPerVector, 
//			g.size * sizeof(int), 
//			cudaMemcpyDeviceToHost); 
//	if(DEBUG){
//    		printf("Number of nonzero elements in dense adjacency matrix = %i\n", M_cap.nnz);
//    		
//		if(VERBOSE)
//		for (int i = 0; i < g.size; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, M_cap.h_nnzPerVector[i]);
//	}
//
//
//	/* Step 6: Convert dense matrix to sparse matrices */
//	allocate_csr(&M_cap, M_cap.nnz, g.size);
//	cusparseSdense2csr(cusparse_handle, 
//			g.size, g.size, 
//			mat_descr,
//		       	M_device,	
//			LDA, 
//			M_cap.d_nnzPerVector, 
//			M_cap.d_values, M_cap.d_rowIndices, M_cap.d_colIndices); 
//	if(VERBOSE){
//		device2host(&M_cap, M_cap.nnz, g.size);	
//		print_csr(
//    			g.size,
//    			M_cap.nnz,
//    			M_cap,
//    			"Adjacency matrix");
//	}
//
//	cudaFree(M_device);
//
//	device2host(&M_cap, M_cap.nnz, g.size);
//	log("Completed conversion of data from dense to sparse");
//
//	/* REMOVE*/
//	//for(int i=0;i<M_cap.nnz;i++){
//	//	std::cout<<"M_cap: "<<M_cap.h_values[i]<<std::endl;
//	//}
//
//	/* Section 7: Compute SVD of objective matrix */	
//
//	char whichS = 'L';
//	char whichV = 'L';
//
//	MKL_INT pm[128];
//	mkl_sparse_ee_init(pm);
//	//pm[1] = 100;
//	//pm[2] = 2;
//	//pm[4] = 60;
//
//	MKL_INT mkl_rows = g.size;
//	MKL_INT mkl_cols = g.size;
//
//
//	//MKL_INT rows_start[mkl_rows];
//	//MKL_INT rows_end[mkl_rows];
//
//	MKL_INT *rows_start;
//	MKL_INT *rows_end;
//
//	rows_start = (MKL_INT *)mkl_malloc(mkl_rows * sizeof(MKL_INT),64);
//	rows_end = (MKL_INT *)mkl_malloc(mkl_rows * sizeof(MKL_INT),64);
//
//	for(int i=0;i<mkl_rows;i++){
//		rows_start[i] = M_cap.h_rowIndices[i];
//		rows_end[i] = M_cap.h_rowIndices[i+1];
//	}
//	
//	//MKL_INT mkl_col_idx[M_cap.nnz];
//
//	MKL_INT *mkl_col_idx;
//	mkl_col_idx = (MKL_INT*)mkl_malloc(M_cap.nnz * sizeof(MKL_INT), 64);
//
//	int mkl_temp;
//	for(int i=0;i<M_cap.nnz;i++){
//		mkl_temp = M_cap.h_colIndices[i];
//		mkl_col_idx[i] = mkl_temp;
//	}
//
//
//	sparse_matrix_t M_mkl;
//	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
//
//	mkl_sparse_s_create_csr(&M_mkl, indexing,
//					mkl_rows, mkl_cols,
//					rows_start, rows_end,
//					mkl_col_idx, M_cap.h_values);
//
//	log("Created MKL sparse");
//
//	matrix_descr mkl_descrM;
//	mkl_descrM.type = SPARSE_MATRIX_TYPE_GENERAL;	
//	mkl_descrM.mode = SPARSE_FILL_MODE_UPPER;
//	mkl_descrM.diag = SPARSE_DIAG_NON_UNIT;
//
//	MKL_INT k0 = dimension;
//	MKL_INT k;
//
//	DT *E_mkl, *K_L_mkl, *K_R_mkl, *res_mkl;
//
//	E_mkl = (DT *)mkl_malloc(k0 * sizeof(DT), 128);
//	K_L_mkl = (DT *)mkl_malloc( k0*mkl_rows*sizeof( DT), 128 );
//        K_R_mkl = (DT *)mkl_malloc( k0*mkl_cols*sizeof( DT), 128 );
//        res_mkl = (DT *)mkl_malloc( k0*sizeof( DT), 128 );
//
//	memset(E_mkl, 0 , k0);
//	memset(K_L_mkl, 0 , k0);
//	memset(K_R_mkl, 0 , k0);
//	memset(res_mkl, 0 , k0);
//
//	int mkl_status = 0;
//
//	log("Computing SVD via MKL");
//	mkl_status = mkl_sparse_s_svd(&whichS, &whichV, pm,
//			M_mkl, mkl_descrM,
//			k0, &k,
//			E_mkl,
//			K_L_mkl,
//			K_R_mkl,
//			res_mkl);
//	log("Computed SVD via MKL");
//
//	if(DEBUG){
//	std::cout<<"Number of singular found: "<<k<<std::endl;
//	for(int i=0;i<k0;i++){ std::cout<<E_mkl[i]<<" ";} std::cout<<"\n";
//	}
//
//	DT *U_device, *Si_device;
//	DT *U_host;
//	DT *Si_host;
//	DT *E_device, *E_host;
//
//	cudaMalloc(&U_device, g.size * dimension * sizeof(DT));
//	cudaMalloc(&E_device, g.size * dimension * sizeof(DT));
//	cudaMalloc(&Si_device, dimension * sizeof(DT));
//
//	U_host = (DT *) malloc(g.size * dimension * sizeof(DT));
//	E_host = (DT *) malloc(g.size * dimension * sizeof(DT));
//	Si_host = (DT *) malloc(dimension * sizeof(DT));
//
//	cudaMemcpy(U_device, K_L_mkl, g.size * dimension * sizeof(DT), cudaMemcpyHostToDevice);
//	cudaMemcpy(Si_device, E_mkl, dimension * sizeof(DT), cudaMemcpyHostToDevice);
//
//	transform_si<<<grids,threads>>>(Si_device, dimension);
//
//	cudaMemcpy(Si_host, Si_device, dimension * sizeof(DT), cudaMemcpyDeviceToHost);
//
//	std::cout<<"\n";
//	cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
//		g.size, dimension,
//		U_device, g.size, 
//		Si_device, 1.0,
//		E_device, g.size);
//
//	cudaMemcpy(E_host, E_device, g.size * dimension * sizeof(DT), cudaMemcpyDeviceToHost);
//
//	write_embeddings("blogcatalog.emb",E_host, g.size, dimension);
//
//	mkl_free(rows_start);	
//	mkl_free(rows_end);	
//	mkl_free(mkl_col_idx);	
//
//	cudaDeviceReset();
}
