#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "mkl_spblas.h"
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cmath>
#include <cassert>
#include <typeinfo>
#include <map>
#include "constants.h"

#include "nmf-utils.h"
#include "model.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector> 
#include <list>
#include <set>

#include <sstream>
#include <string>
#include <numeric>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <memory.h>
#include <string.h>
#include "cuda_util.h"

#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "update_kernel.h"

#define MM_TYPE mat_type

using namespace std;
using namespace CUDAUtil;


inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    if(result != cudaSuccess)
        exit (-1);
  }
  return result;
}

double rtclock(void) {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


bool myfunction (int i,int j) { return (i<j); }

struct myclass {
    bool operator() (int i,int j) { return (i<j);}
} myobject;


model::~model() {

    // Tile_size = TS;

    if (model_status == MODEL_STATUS_FAST_HALS_CPU  || model_status == MODEL_STATUS_FAST_HALS_GPU) {

        if (WT) {
          for (int w = 0; w < V; w++) {
            if (WT[w]) {
              delete WT[w];
            }
          }
        }
        if (DT) {
          for (int d = 0; d < D; d++) {
            if (DT[d]) {
              delete DT[d];
            }
          }
        }

        if (trainData) {
          for (int w = 0; w < V; w++) {
            if (trainData[w]) {
              delete trainData[w];
            }
          }
        }

        if (temp_error) {
          for (int v = 0; v < V; v++) {
            if (temp_error[v]) {
              delete temp_error[v];
            }
          }
        }
    }

}


void model::set_default_values() {
    model_status = MODEL_STATUS_UNKNOWN;
    V = 0;
    D = 0;
    K = 0;
    niters = 0;
    liter = 0;
}

int model::parse_args(int argc, char ** argv) {
    return utils::parse_args(argc, argv, this);
}

int model::init(int argc, char ** argv) {
    if (parse_args(argc, argv)) {
        return 1;
    }
    return 0;
}


int model::init_est() {
    printf("NMF initialization skipped\n");
    return 0;
}




void model::estimate_HALS_CPU(double *data) {

    printf("Starting NMF based on FAST-HALS algorithm - CPU version\n\n");

    Tile_size = TS;

    double *m_W_old, *m_W_new, *m_H_old, *m_H_new, *m_temp_p, *m_temp_q, *m_temp_r, *m_temp_s;

    m_W_old = (double *)mkl_malloc( V*K*sizeof( double ), 64 );
    m_W_new = (double *)mkl_malloc( V*K*sizeof( double ), 64 );
    m_H_old = (double *)mkl_malloc( D*K*sizeof( double ), 64 );
    m_H_new = (double *)mkl_malloc( D*K*sizeof( double ), 64 );

    m_temp_p = (double *)mkl_malloc( V*K*sizeof( double ), 64 );
    m_temp_q = (double *)mkl_malloc( K*K*sizeof( double ), 64 );
    m_temp_r = (double *)mkl_malloc( D*K*sizeof( double ), 64 );
    m_temp_s = (double *)mkl_malloc( K*K*sizeof( double ), 64 );


    double eps_W_H = 1e-5;
    srand48(0L);
    WT = new double*[V];
    for (int v = 0; v < V; v++) {
        WT[v] = new double[K];
        for (int k = 0; k < K; k++) {
            m_W_old[v*K+k] = 0.1 * drand48();
            if (m_W_old[v*K+k] >= 1) {
                m_W_old[v*K+k] = m_W_old[v*K+k] - eps_W_H;
            }
            WT[v][k] = m_W_old[v*K+k];
            m_W_new[v*K+k] = m_W_old[v*K+k];
        }
    }

    srand48(0L);
    DT = new double*[D];
    for (int d = 0; d < D; d++) {
        DT[d] = new double[K];
        for (int k = 0; k < K; k++) {
            m_H_old[d*K+k] = 0.1 * drand48();
            if (m_H_old[d*K+k] >= 1) {
                m_H_old[d*K+k] = m_H_old[d*K+k] - eps_W_H;
            }
            DT[d][k] = m_H_old[d*K+k];
            m_H_new[d*K+k] = m_H_old[d*K+k];
        }
    }

   /* vector <vector <int> > data;
    ifstream infile(train_file);

    while (infile)
    {
        string s;
        if (!getline( infile, s )) break;
        istringstream ss( s );
        vector <int> record;
        while (ss)
        {
            string s;
            if (!getline( ss, s, ',' )) break;
            int a = atoi(s.c_str());
            record.push_back( a );
        }
        data.push_back( record );
    }

    if (!infile.eof())
    {
        cerr << "Fooey!\n";
    }*/


    int total_nnz = 0;

    trainData = new double*[V];

    for (int v = 0; v < V; v++) {
      trainData[v]  = new double[D];
      for (int d = 0; d < D; d++) {
        trainData[v][d] = data[v*D + d];
        if (trainData[v][d] != 0) {
                total_nnz += 1;
        }
      }
    }

    printf("total number of nnz = %d\n",total_nnz);

    double *m_denseData;
    m_denseData = (double *)mkl_malloc( V*D*sizeof( double ), 64 );

    for (int v = 0; v < V; v++) {
        for (int d = 0; d < D; d++) {
            m_denseData[v*D+d] = trainData[v][d];
        }
    }


    double *trainData_sparse;
    MKL_INT *trainData_sparse_cols;
    MKL_INT *trainData_sparse_ptrB;
    MKL_INT *trainData_sparse_ptrE;


    double* csr_val_trainData;
    int* csr_col_trainData;
    int* csr_ptrB_trainData;
    int* csr_ptrE_trainData;

    csr_val_trainData = new double[total_nnz];
    csr_col_trainData = new int[total_nnz];
    csr_ptrB_trainData = new int[V];
    csr_ptrE_trainData = new int[V];

    int nnz_idx = 0;
    for (int v = 0; v < V; v++) {
        csr_ptrB_trainData[v] = nnz_idx;
        for (int d = 0; d < D; d++) {
            if (trainData[v][d] != 0) {
                csr_val_trainData[nnz_idx] = trainData[v][d];
                csr_col_trainData[nnz_idx] = d;
                nnz_idx++;
            }
        }
        csr_ptrE_trainData[v] = nnz_idx;
    }

    trainData_sparse = (double *)mkl_malloc( total_nnz*sizeof(double), 64);
    trainData_sparse_cols = (MKL_INT *)mkl_malloc(total_nnz*sizeof(MKL_INT), 64);
    trainData_sparse_ptrB = (MKL_INT *)mkl_malloc(V*sizeof(MKL_INT),64);
    trainData_sparse_ptrE = (MKL_INT *)mkl_malloc(V*sizeof(MKL_INT),64);

    for (int i = 0; i < total_nnz; i++) {
        trainData_sparse[i] = csr_val_trainData[i];
        trainData_sparse_cols[i] = csr_col_trainData[i];
    }

    for (int v = 0; v < V; v++) {
        trainData_sparse_ptrB[v] =csr_ptrB_trainData[v];
        trainData_sparse_ptrE[v] = csr_ptrE_trainData[v];
    }

    double ss = 0.0;
    for (int r = 0; r < V; r++) {
        for (int c = 0; c < D; c++) {
            ss += trainData[r][c]*trainData[r][c];
        }
    }
    norm_trainData = sqrt(ss);


    // TEMP matrices for computing relative error
    temp_error = new double*[V];
    for (int v = 0; v < V; v++) {
        temp_error[v] = new double[D];
    }

    int print_error_step = 1;
    int iter_id = 0;
    double rel_errors;
    rel_errors = compute_rel_error();
    printf("initial relative error = %f\n",rel_errors);
    double sum_total_time = 0.0;

    double eps = 1e-16;
    int n_epoch = niters;

    char transa;
    char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};

    double* elapsed_time_5iter;
    elapsed_time_5iter = new double[n_epoch/print_error_step];
    for (int et = 0; et < n_epoch/print_error_step; et++) {
        elapsed_time_5iter[et] = 0.0;
    }

    double* error_5iter;
    error_5iter = new double[n_epoch/print_error_step];
    for (int ep = 0; ep < n_epoch/print_error_step; ep++) {
        error_5iter[ep] = 0.0;
    }

    // int num_tiles = K/Tile_size;
    int num_tiles = (K+Tile_size-1) / Tile_size;

    printf("Number of tile = %d\n",num_tiles);
    printf("Tile size = %d\n",Tile_size);

    printf("%d iterations (CPU, parallel)\n", n_epoch);

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        printf("Iteration %d ...\n", epoch+1);
        rel_errors = 0.0;
        double start = rtclock();

        // Copy old matrices from previous iteration

        #pragma omp parallel for
        for (int v = 0; v< V; v++) {
            for (int k = 0; k < K; k++) {
                m_W_old[v*K+k] = m_W_new[v*K+k];
            }
        }
        #pragma omp parallel for
        for (int d = 0; d< D; d++) {
            for (int k = 0; k < K; k++) {
                m_H_old[d*K+k] = m_H_new[d*K+k];
            }
        }

/********************************updating H************************************/

        transa = 'T';
        MKL_INT ldb = K, ldc=K;
        alpha_mkl = 1.0; beta_mkl = 0.0;

        // mkl_dcsrmm(&transa, &V, &K, &D, &alpha_mkl, matdescra, trainData_sparse, trainData_sparse_cols, trainData_sparse_ptrB, trainData_sparse_ptrE, &m_W_old[0], &ldb, &beta_mkl, &m_temp_r[0], &ldc); // for sparse dataset
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, K, V, alpha_mkl, &m_denseData[0], D, &m_W_old[0], K, beta_mkl, &m_temp_r[0], K); // for dense dataset
        
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, V, alpha_mkl, &m_W_old[0], K, &m_W_old[0], K, beta_mkl, &m_temp_s[0], K);


        alpha_mkl = -1.0; beta_mkl = 1.0;

        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, (tile_id*Tile_size), Tile_size, alpha_mkl, &m_H_old[0]+(tile_id*Tile_size), K, &m_temp_s[0]+(tile_id*Tile_size*K), K, beta_mkl, &m_H_new[0], K);
        }

        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {
                #pragma omp parallel for
                for (int d = 0; d < D; d++) {
                    double tmp = 0;

                    int k = tile_id*Tile_size;
                    #pragma omp simd reduction(+:tmp)
                    for (; k < t; k++) {
                            tmp += (m_H_new[d*K+k]*m_temp_s[t*K+k]);
                    }
                    #pragma omp simd reduction(+:tmp)
                    for (k=t; k < (tile_id+1)*Tile_size; k++) {
                            tmp += (m_H_old[d*K+k]*m_temp_s[t*K+k]);
                    }
                    m_H_new[d*K+t] = max(m_H_new[d*K+t] - tmp + m_temp_r[d*K+t],eps);
                }
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, (K-(tile_id+1)*Tile_size), Tile_size, alpha_mkl, &m_H_new[0]+(tile_id*Tile_size), K, &m_temp_s[0]+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, beta_mkl, &m_H_new[0]+((tile_id+1)*Tile_size), K);
        }


/********************************updating W************************************/


        alpha_mkl = 1.0; beta_mkl = 0.0;
        transa = 'N';

        // mkl_dcsrmm(&transa, &V, &K, &D, &alpha_mkl, matdescra, trainData_sparse, trainData_sparse_cols, trainData_sparse_ptrB, trainData_sparse_ptrE, &m_H_new[0], &ldb, &beta_mkl, &m_temp_p[0], &ldc); // for sparse dataset
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, V, K, D, alpha_mkl, &m_denseData[0], D, &m_H_new[0], K, beta_mkl, &m_temp_p[0], K); // for dense dataset

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, D, alpha_mkl, &m_H_new[0], K, &m_H_new[0], K, beta_mkl, &m_temp_q[0], K);


        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            for (int k = 0; k < K; k++) {
                m_W_new[v*K+k] = m_W_old[v*K+k]*m_temp_q[k*K+k];
            }
        }

        alpha_mkl = -1.0; beta_mkl = 1.0;

        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, V, (tile_id*Tile_size), Tile_size, alpha_mkl, &m_W_old[0]+(tile_id*Tile_size), K, &m_temp_q[0]+(tile_id*Tile_size*K), K, beta_mkl, &m_W_new[0], K);
        }

        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {
                double ss_col = 0;

                #pragma omp parallel for reduction(+:ss_col)
                for (int v = 0; v < V; v++) {
                    double tmp = 0;

                    int k = tile_id*Tile_size;
                    #pragma omp simd reduction(+:tmp)
                    for (; k < t; k++) {
                            tmp += (m_W_new[v*K+k]*m_temp_q[t*K+k]);
                    }
                    #pragma omp simd reduction(+:tmp)
                    for (k=t; k < (tile_id+1)*Tile_size; k++) {
                            tmp += (m_W_old[v*K+k]*m_temp_q[t*K+k]);
                    }
                    m_W_new[v*K+t] = max(m_W_new[v*K+t] - tmp + m_temp_p[v*K+t],eps);
                    ss_col += m_W_new[v*K+t]*m_W_new[v*K+t];
                }

                #pragma omp parallel for
                for (int w = 0; w < V; w++) {
                    m_W_new[w*K+t] = m_W_new[w*K+t]/sqrt(ss_col);
                }
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, V, (K-(tile_id+1)*Tile_size), Tile_size, alpha_mkl, &m_W_new[0]+(tile_id*Tile_size), K, &m_temp_q[0]+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, beta_mkl, &m_W_new[0]+((tile_id+1)*Tile_size), K);
        }


        double end = rtclock();
        sum_total_time += end-start;

        if ((epoch+1) % print_error_step == 0) {

            #pragma omp parallel for
            for (int w = 0; w < V; w++) {
                for (int k = 0; k < K; k++) {
                    WT[w][k] = m_W_new[w*K+k];
                }
            }
            #pragma omp parallel for
            for (int d = 0; d < D; d++) {
                for (int k = 0; k < K; k++) {
                    DT[d][k] = m_H_new[d*K+k];
                }
            }
            rel_errors = compute_rel_error();
            printf("relative error = %f\n",rel_errors);
            error_5iter[iter_id] = rel_errors;

            printf("elpased time = %f\n", sum_total_time);
            elapsed_time_5iter[iter_id] = sum_total_time;
            iter_id++;
        }
    }

    mkl_free(m_W_old);
    mkl_free(m_W_new);

    mkl_free(m_H_old);
    mkl_free(m_H_new);

    mkl_free(m_temp_p);
    mkl_free(m_temp_q);
    mkl_free(m_temp_r);
    mkl_free(m_temp_s);

    mkl_free(trainData_sparse);
    mkl_free(trainData_sparse_cols);
    mkl_free(trainData_sparse_ptrB);
    mkl_free(trainData_sparse_ptrE);

    // save_topic(model_name);

    printf("\n===========ELASPED TIME===========\n\n");
    ofstream myfile_time;
    myfile_time.open ("parallel_cpu_time.txt");
    for (int output_time =0; output_time < n_epoch/print_error_step; output_time++) {
        myfile_time << elapsed_time_5iter[output_time] << endl;
        printf("%f\n",elapsed_time_5iter[output_time]);
    }
    myfile_time.close();

    printf("\n===========RELATIVE ERROR===========\n\n");
    ofstream myfile_error;
    myfile_error.open ("parallel_cpu_rel_error.txt");
    for (int output_error = 0; output_error < n_epoch/print_error_step; output_error++) {
        myfile_error << error_5iter[output_error] << endl;
        printf("%f\n",error_5iter[output_error]);
    }
    myfile_error.close();

    printf("Accumulated Total Time: %fs\n\n", sum_total_time);
}



void model::estimate_HALS_GPU(double *data) {

    printf("Starting NMF based on FAST-HALS algorithm - GPU version\n\n");

    Tile_size = TS;


    double eps_WT_DT = 1e-5;
    srand48(0L);
    WT = new double*[V];
    for (int w = 0; w < V; w++) {
        WT[w] = new double[K];
        for (int k = 0; k < K; k++) {
            WT[w][k] = 0.1 * drand48();
            if (WT[w][k] >= 1) {
                WT[w][k] = WT[w][k] - eps_WT_DT;
            }
            if (WT[w][k] <= 0 || WT[w][k] >= 1) {
                printf("random intialization error \n");
            }
        }
    }
    srand48(0L);
    DT = new double*[D];
    for (int d = 0; d < D; d++) {
        DT[d] = new double[K];
        for (int k = 0; k < K; k++) {
            DT[d][k] = 0.1 * drand48();
            if (DT[d][k] >= 1) {
                DT[d][k] = DT[d][k] - eps_WT_DT;
            }
            if (DT[d][k] <= 0 || DT[d][k] >= 1) {
                printf("random intialization error \n");
            }
        }
    }

    //vector <vector <int> > data;
    //ifstream infile(train_file);

    //while (infile)
    //{
    //    string s;
    //    if (!getline( infile, s )) break;
    //    istringstream ss( s );
    //    vector <int> record;
    //    while (ss)
    //    {
    //        string s;
    //        if (!getline( ss, s, ',' )) break;
    //        int a = atoi(s.c_str());
    //        record.push_back( a );
    //    }
    //    data.push_back( record );
    //}

    //if (!infile.eof())
    //{
    //    cerr << "Fooey!\n";
    //}

    int total_nnz = 0;

    trainData = new double*[V];
    for (int v = 0; v < V; v++) {
      trainData[v]  = new double[D];
      for (int d = 0; d < D; d++) {
        trainData[v][d] = data[v * D + d];
        if (trainData[v][d] != 0) {
                total_nnz += 1;
        }
      }
    }

    double *h_denseData;
    h_denseData = new double[V*D];

    for (int d = 0; d < D; d++) {
        for (int v = 0; v < V; v++) {
            h_denseData[v+d*V] = trainData[v][d];
        }
    }

    double* csr_val_trainData;
    int* csr_col_ind_trainData;
    int* csr_row_ptr_trainData;

    csr_val_trainData = new double[total_nnz];
    csr_col_ind_trainData = new int[total_nnz];
    csr_row_ptr_trainData = new int[V+1];

    int nnz_idx = 0;
    for (int v = 0; v < V; v++) {
        csr_row_ptr_trainData[v] = nnz_idx;
        for (int d = 0; d < D; d++) {
            if (trainData[v][d] != 0) {
                csr_val_trainData[nnz_idx] = trainData[v][d];
                csr_col_ind_trainData[nnz_idx] = d;
                nnz_idx++;
            }
        }
    }
    csr_row_ptr_trainData[V] = nnz_idx;

    double* csr_val_trainData_T;
    int* csr_col_ind_trainData_T;
    int* csr_row_ptr_trainData_T;

    csr_val_trainData_T = new double[total_nnz];
    csr_col_ind_trainData_T = new int[total_nnz];
    csr_row_ptr_trainData_T = new int[D+1];

    int nnz_idx_T = 0;
    for (int d = 0; d < D; d++) {
        csr_row_ptr_trainData_T[d] = nnz_idx_T;
        for (int v = 0; v < V; v++) {
            if (trainData[v][d] != 0) {
                csr_val_trainData_T[nnz_idx_T] = trainData[v][d];
                csr_col_ind_trainData_T[nnz_idx_T] = v;
                nnz_idx_T++;
            }
        }
    }
    csr_row_ptr_trainData_T[D] = nnz_idx_T;


    double *h_H_old, *h_H_new, *h_W_old, *h_W_new;

    h_H_old = new double[D*K];
    h_H_new = new double[D*K];

    h_W_old = new double[V*K];
    h_W_new = new double[V*K];

    for (int k = 0; k < K; k++) {
        for (int d = 0; d < D; d++) {
            h_H_old[d+k*D] = DT[d][k];
            h_H_new[d+k*D] = DT[d][k];
        }
    }

    for (int k = 0; k < K; k++) {
        for (int v = 0; v < V; v++) {
            h_W_old[v+k*V] = WT[v][k];
            h_W_new[v+k*V] = WT[v][k];
        }
    }

    double ss = 0.0;
    for (int r = 0; r < V; r++) {
        for (int c = 0; c < D; c++) {
            ss += trainData[r][c]*trainData[r][c];
        }
    }
    norm_trainData = sqrt(ss);

    // TEMP matrices for computing relative error
    temp_error = new double*[V];
    for (int v = 0; v < V; v++) {
        temp_error[v] = new double[D];
    }

    double *_d_csr_val, *_d_csr_val_T;
    int *_d_csr_col_ind, *_d_row_ptr, *_d_csr_col_ind_T, *_d_row_ptr_T;

    double *_d_denseData, *_d_H_old, *_d_H_new, *_d_temp_r, *_d_temp_r_dense, *_d_temp_s;
    double *_d_W_old, *_d_W_new, *_d_temp_p, *_d_temp_q, *_d_ss_col;

    cudaMalloc((void**)&_d_csr_val, sizeof (double) *total_nnz);
    cudaMemcpy(_d_csr_val, csr_val_trainData, sizeof (double) *total_nnz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_csr_col_ind, sizeof (int) *total_nnz);
    cudaMemcpy(_d_csr_col_ind, csr_col_ind_trainData, sizeof (int) *total_nnz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_row_ptr, sizeof (int) *(V+1));
    cudaMemcpy(_d_row_ptr, csr_row_ptr_trainData, sizeof (int) *(V+1), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_d_csr_val_T, sizeof (double) *total_nnz);
    cudaMemcpy(_d_csr_val_T, csr_val_trainData_T, sizeof (double) *total_nnz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_csr_col_ind_T, sizeof (int) *total_nnz);
    cudaMemcpy(_d_csr_col_ind_T, csr_col_ind_trainData_T, sizeof (int) *total_nnz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_row_ptr_T, sizeof (int) *(D+1));
    cudaMemcpy(_d_row_ptr_T, csr_row_ptr_trainData_T, sizeof (int) *(D+1), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_d_denseData, sizeof (double) *(V*D));
    cudaMemcpy(_d_denseData, h_denseData, sizeof (double) *(V*D), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_d_H_old, sizeof (double) *(D*K));
    cudaMalloc((void**)&_d_H_new, sizeof (double) *(D*K));
    cudaMemcpy(_d_H_old, h_H_old, sizeof (double) *(D*K), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_H_new, h_H_new, sizeof (double) *(D*K), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_temp_r, sizeof (double) *(D*K));
    cudaMemset((void *)_d_temp_r,0, sizeof (double) *(D*K));
    cudaMalloc((void**)&_d_temp_r_dense, sizeof (double) *(D*K));
    cudaMemset((void *)_d_temp_r_dense,0, sizeof (double) *(D*K));
    cudaMalloc((void**)&_d_temp_s, sizeof (double) *(K*K));
    cudaMemset((void *)_d_temp_s,0, sizeof (double) *(K*K));

    cudaMalloc((void**)&_d_W_old, sizeof (double) *(V*K));
    cudaMalloc((void**)&_d_W_new, sizeof (double) *(V*K));
    cudaMemcpy(_d_W_old, h_W_old, sizeof (double) *(V*K), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_W_new, h_W_new, sizeof (double) *(V*K), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_temp_p, sizeof (double) *(V*K));
    cudaMemset((void *)_d_temp_p,0, sizeof (double) *(V*K));
    cudaMalloc((void**)&_d_temp_q, sizeof (double) *(K*K));
    cudaMemset((void *)_d_temp_q,0, sizeof (double) *(K*K));

    cudaMalloc((void**)&_d_ss_col, sizeof (double) *1);
    cudaMemset((void *)_d_ss_col,0, sizeof (double) *1);

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    int n_epoch = niters;
    int print_error_step = 1;

    double* elapsed_time_5iter;
    elapsed_time_5iter = new double[n_epoch/print_error_step];
    for (int et = 0; et < n_epoch/print_error_step; et++) {
        elapsed_time_5iter[et] = 0.0;
    }

    double* error_5iter;
    error_5iter = new double[n_epoch/print_error_step];
    for (int ep = 0; ep < n_epoch/print_error_step; ep++) {
        error_5iter[ep] = 0.0;
    }

    int iter_id = 0;
    double rel_errors;
    double eps = 1e-16;
    double sum_total_time = 0.0;

    char transa;
    char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};

    // int num_tiles = K/Tile_size;
    int num_tiles = (K+Tile_size-1) / Tile_size;
    printf("Number of tile = %d\n",num_tiles);
    printf("Tile size = %d\n",Tile_size);

    printf("%d iterations (GPU, parallel)\n", n_epoch);


    for (int epoch = 0; epoch < n_epoch; epoch++) {
        printf("Iteration %d ...\n", epoch+1);
        rel_errors = 0.0;
        float mili =0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        cudaMemcpy(_d_W_old, _d_W_new, sizeof (double) *(V*K), cudaMemcpyDeviceToDevice);

        cudaMemcpy(_d_H_old, _d_H_new, sizeof (double) *(D*K), cudaMemcpyDeviceToDevice);

/********************************updating H************************************/

        alpha_cuda = 1.0; beta_cuda = 0.0;

        // cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, D, K, V, total_nnz, &alpha_cuda, descr, _d_csr_val_T, _d_row_ptr_T, _d_csr_col_ind_T, _d_W_old, V, &beta_cuda, _d_temp_r, D); // for sparse dataset
        // cudaDeviceSynchronize();

        cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, K, V, &alpha_cuda, _d_denseData, V, _d_W_old, V, &beta_cuda, _d_temp_r, D); // for dense dataset
        cudaDeviceSynchronize();

        cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, K, K, V, &alpha_cuda, _d_W_old, V, _d_W_old, V, &beta_cuda, _d_temp_s, K);
        cudaDeviceSynchronize();


        // update H - PHASE 1
        alpha_cuda = -1.0; beta_cuda = 1.0;


        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, (tile_id*Tile_size), Tile_size, &alpha_cuda, _d_H_old+(tile_id*Tile_size*D), D, _d_temp_s+(tile_id*Tile_size*K), K, &beta_cuda, _d_H_new, D);
            cudaDeviceSynchronize();
        }


        // update H - PHASE 2 & 3
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {
                phase_two_H(_d_H_old, _d_H_new, _d_temp_s, _d_temp_r, t, tile_id, Tile_size, D, K, eps);
                cudaDeviceSynchronize();
            }

            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, (K-(tile_id+1)*Tile_size), Tile_size, &alpha_cuda, _d_H_new+(tile_id*Tile_size*D), D, _d_temp_s+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, &beta_cuda, _d_H_new+((tile_id+1)*Tile_size*D), D);
            cudaDeviceSynchronize();
        }


/********************************updating W************************************/

        alpha_cuda = 1.0; beta_cuda = 0.0;

        // cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, V, K, D, total_nnz, &alpha_cuda, descr, _d_csr_val, _d_row_ptr, _d_csr_col_ind, _d_H_new, D, &beta_cuda, _d_temp_p, V); // for sparse dataset
        // cudaDeviceSynchronize();

        cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, V, K, D, &alpha_cuda, _d_denseData, V, _d_H_new, D, &beta_cuda, _d_temp_p, V); // for dense dataset
        cudaDeviceSynchronize();

        cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, K, K, D, &alpha_cuda, _d_H_new, D, _d_H_new, D, &beta_cuda, _d_temp_q, K);
        cudaDeviceSynchronize();

        cuda_mul_W_old_temp_q(_d_W_old, _d_W_new, _d_temp_q, V, K);
        cudaDeviceSynchronize();

        // update W - PHASE 1
        alpha_cuda = -1.0; beta_cuda = 1.0;
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, (tile_id*Tile_size), Tile_size, &alpha_cuda, _d_W_old+(tile_id*Tile_size*V), V, _d_temp_q+(tile_id*Tile_size*K), K, &beta_cuda, _d_W_new, V);
            cudaDeviceSynchronize();
        }


        // update W - PHASE 2 & 3
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {

                cudaMemset((void *)_d_ss_col,0, sizeof (double) *1);
                phase_two_W(_d_W_old, _d_W_new, _d_temp_q, _d_temp_p, _d_ss_col, t, tile_id, Tile_size, V, K, eps);
                cudaDeviceSynchronize();

                cuda_div_W_new_col(_d_W_new, _d_ss_col, t, V);
                cudaDeviceSynchronize();

            }
            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, (K-(tile_id+1)*Tile_size), Tile_size, &alpha_cuda, _d_W_new+(tile_id*Tile_size*V), V, _d_temp_q+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, &beta_cuda, _d_W_new+((tile_id+1)*Tile_size*V), V);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&mili, start, stop);
        sum_total_time += mili;


        if ((epoch+1) % print_error_step == 0) {
            cudaMemcpy(h_H_new, _d_H_new, sizeof (double) *(D*K), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_W_new, _d_W_new, sizeof (double) *(V*K), cudaMemcpyDeviceToHost);
            #pragma omp parallel for
            for (int w = 0; w < V; w++) {
                for (int k = 0; k < K; k++) {
                    WT[w][k] = h_W_new[k*V+w];
                }
            }
            #pragma omp parallel for
            for (int d = 0; d < D; d++) {
                for (int k = 0; k < K; k++) {
                    DT[d][k] = h_H_new[k*D+d];
                }
            }

            rel_errors = compute_rel_error();
            printf("relative error = %f\n",rel_errors);
            error_5iter[iter_id] = rel_errors;

            printf("elpased time = %f\n", sum_total_time/1000);
            elapsed_time_5iter[iter_id] = sum_total_time/1000;
            iter_id++;
        }
    }
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    cublasDestroy(handle_cublas);

    cudaFree(_d_csr_val);
    cudaFree(_d_csr_col_ind);
    cudaFree(_d_row_ptr);
    cudaFree(_d_csr_val_T);
    cudaFree(_d_csr_col_ind_T);
    cudaFree(_d_row_ptr_T);
    cudaFree(_d_H_old);
    cudaFree(_d_H_new);
    cudaFree(_d_temp_r);
    cudaFree(_d_temp_s);
    cudaFree(_d_W_old);
    cudaFree(_d_W_new);
    cudaFree(_d_temp_p);
    cudaFree(_d_temp_q);
    cudaFree(_d_ss_col);

    // save_topic(model_name);

    printf("\n===========ELASPED TIME===========\n\n");
    ofstream myfile_time;
    myfile_time.open ("parallel_gpu_time.txt");
    for (int output_time =0; output_time < n_epoch/print_error_step; output_time++) {
        myfile_time << elapsed_time_5iter[output_time] << endl;
        printf("%f\n",elapsed_time_5iter[output_time]);
    }
    myfile_time.close();

    printf("\n===========RELATIVE ERROR===========\n\n");
    ofstream myfile_error;
    myfile_error.open ("parallel_gpu_rel_error.txt");
    for (int output_error = 0; output_error < n_epoch/print_error_step; output_error++) {
        myfile_error << error_5iter[output_error] << endl;
        printf("%f\n",error_5iter[output_error]);
    }
    myfile_error.close();

    printf("Accumulated Total Time: %fs\n", sum_total_time/1000);

}


double model::compute_rel_error() {

    double rel_error = 0.0;
    double norm_error = 0.0;

    #pragma omp parallel for
    for (int v = 0; v < V; v++) {
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += WT[v][k]*DT[d][k];
            }
            temp_error[v][d] = trainData[v][d] - sum;
        }
    }

    double ts = 0.0;
    for (int v = 0; v < V; v++) {
        for (int d = 0; d < D; d++) {
            ts += temp_error[v][d]*temp_error[v][d];
        }
    }

    norm_error = sqrt(ts);
    rel_error = norm_error / norm_trainData;

    return rel_error;

}

int model::save_topic(string model_name) {

    int num_top_words = 10;

        string filename = "topics_NMF_K.txt";
        FILE * fout = fopen(filename.c_str(), "w");

        for (int k = 0; k < K; k++) {

            double col_sum = 0.0;
            for (int v = 0; v < V; v++) {
                col_sum += WT[v][k];
            }

            vector<pair<int, double> > words_probs;
            pair<int, double> word_prob;
            for (int w = 0; w < V; w++) {
                word_prob.first = w;
                word_prob.second = WT[w][k] / col_sum;
                //WT[w][k] = WT[w][k] / col_sum;
                words_probs.push_back(word_prob);
            }

            utils::quicksort(words_probs, 0, words_probs.size() - 1);

            //printf("\n==========topic %d==========\n", k);
            //fprintf(fout, "\n==========topic %d==========\n", k);

            for (int top = 0; top < num_top_words; top++) {
                //printf("word index = %d real word = %s word prob = %f\n",words_probs[top].first,vocabmap[words_probs[top].first].c_str(),words_probs[top].second);
                //printf("word index = %d real word = %s\n",words_probs[top].first,vocabmap[words_probs[top].first].c_str());
                //fprintf(fout, "word index = %d real word = %s\n", words_probs[top].first,vocabmap[words_probs[top].first].c_str());
                //fprintf(fout, "%s\n", vocabmap[words_probs[top].first].c_str());
                 if (top < num_top_words-1) {
                    fprintf(fout, "%s ", vocabmap[words_probs[top].first].c_str());
                }
                else {
                    fprintf(fout, "%s", vocabmap[words_probs[top].first].c_str());
                }
            }
            fprintf(fout, "\n");
        }
        fclose(fout);

    return 0;
}

