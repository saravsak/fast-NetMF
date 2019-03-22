#ifndef _MODEL_H
#define _MODEL_H

#include "mkl.h"
#include "constants.h"
#include <iostream>

using namespace std;

// NMF model
class model {
public:
    // fixed options

    int model_status;       // model status:

    // --- model parameters and variables ---

    string model_name;

    int mat_type;

    int V; // vocabulary size
    int D; // number of documents
    int K; // number of topics
    int Tile_size;

    double norm_trainData;

    int TS;
    int niters;
    int liter;
    string train_file;

    string * vocabmap;

    double ** WT;
    double ** DT;
    double ** trainData;

    double ** temp_w;
    double ** temp_v;
    double ** temp_p;
    double ** temp_q;

    double ** temp_error;

    double * p;

    double alpha, beta;
    double alpha_mkl, beta_mkl;
    double alpha_cuda, beta_cuda;


    model() {
    set_default_values();
    }

    ~model();

    void set_default_values();
    int parse_args(int argc, char ** argv);
    int init(int argc, char ** argv);
    int init_est();

    void estimate_HALS_CPU(double *data);
    void estimate_HALS_GPU(double *data);
    double compute_rel_error();
    int save_topic(string model_name);

};

#endif
