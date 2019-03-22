#include "model.h"

#include <cstdio>
#include <cstdlib>
#include <stdlib.h>

using namespace std;

void show_help();

int main(int argc, char ** argv) {

    srandom(0); // initialize for random number generation

    model nmf;

    if (nmf.init(argc, argv)) {
        show_help();
        return 1;
    }

    double *data;
    int v = nmf.V;
    int d = nmf.D;

    data = (double *)malloc(v * d * sizeof(double));

    for(int i=0;i<v;i++){
    	for(int j=0;j<d;j++){
		if(rand() %10 + 1 > 5){
			data[i * d + j] = 1;
		}else{
			data[i * d + j] = 0;
		}
	}
    }

    if (nmf.model_status == MODEL_STATUS_FAST_HALS_CPU) {
        printf("Optimized FAST HALS on multi-core CPUs\n");
        nmf.estimate_HALS_CPU(data);
    }

    if (nmf.model_status == MODEL_STATUS_FAST_HALS_GPU) {
        printf("Optimized FAST HALS on a single GPU\n");
        nmf.estimate_HALS_GPU(data);
    }

    return 0;
}

void show_help() {
    printf("Command line usage:\n");
}

