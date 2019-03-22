#include <stdio.h>
#include <string>
#include <map>
#include "nmf-utils.h"
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

int utils::parse_args(int argc, char ** argv, model * pmodel) {
    int model_status = MODEL_STATUS_UNKNOWN;
    int K = 0;
    int TS = 0;
    int niters = 0;
    string train_file = "";
    int V = 0;
    int D = 0;
    int mat_type;

    int i = 0;
    while (i < argc) {
    string arg = argv[i];

    if (arg == "-est_nmf_cpu") {
        model_status = MODEL_STATUS_FAST_HALS_CPU;

    } else if (arg == "-est_nmf_gpu") {
        model_status = MODEL_STATUS_FAST_HALS_GPU;

    } else if (arg == "-K") {
        K = atoi(argv[++i]);

    } else if (arg == "-tile_size") {
        TS = atoi(argv[++i]);

    } else if (arg == "-data") {
        train_file = argv[++i];

    } else if (arg == "-V") {
        V = atoi(argv[++i]);

    } else if (arg == "-D") {
        D = atoi(argv[++i]);

    } else if (arg == "-mat_type") {
        mat_type = atoi(argv[++i]);

    } else if (arg == "-niters") {
        niters = atoi(argv[++i]);

    } else {
        // any more?
    }

    i++;
    }

    if (model_status == MODEL_STATUS_FAST_HALS_CPU || model_status == MODEL_STATUS_FAST_HALS_GPU) {
        pmodel->model_status = model_status;
        if (K > 0) {
            pmodel->K = K;
        }

        if (TS > 0) {
            pmodel->TS = TS;
        }

        if (niters > 0) {
            pmodel->niters = niters;
        }

        if (V > 0) {
            pmodel->V = V;
        }

        if (D > 0) {
            pmodel->D = D;
        }

        pmodel->mat_type = mat_type;

        pmodel->train_file = train_file;
    }


    if (model_status == MODEL_STATUS_UNKNOWN) {
    printf("Please specify the task you would like to perform (-est/-estc/-inf)!\n");
    return 1;
    }

    return 0;
}

void utils::quicksort(vector<pair<int, double> > & vect, int left, int right) {
    int l_hold, r_hold;
    pair<int, double> pivot;

    l_hold = left;
    r_hold = right;
    int pivotidx = left;
    pivot = vect[pivotidx];

    while (left < right) {
    while (vect[right].second <= pivot.second && left < right) {
        right--;
    }
    if (left != right) {
        vect[left] = vect[right];
        left++;
    }
    while (vect[left].second >= pivot.second && left < right) {
        left++;
    }
    if (left != right) {
        vect[right] = vect[left];
        right--;
    }
    }

    vect[left] = pivot;
    pivotidx = left;
    left = l_hold;
    right = r_hold;

    if (left < pivotidx) {
    quicksort(vect, left, pivotidx - 1);
    }
    if (right > pivotidx) {
    quicksort(vect, pivotidx + 1, right);
    }
}
