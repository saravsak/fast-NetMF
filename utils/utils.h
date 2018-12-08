#pragma once

#include<time.h>
#include<chrono>

typedef std::chrono::milliseconds milliseconds;

struct info {
	const char * dataset;
	const char * algo;
	double dimension;
	double window_size;
	milliseconds iptime;
	milliseconds init;
	milliseconds gpuio;
	milliseconds compute_d;
	milliseconds compute_x;
	milliseconds compute_s;
	milliseconds compute_m;
	milliseconds svd;
	milliseconds emb;
};

void log(const char* message);
double diff_ms(timeval t1, timeval t2);

struct CSR{
	unsigned int row_id;
	unsigned int col_idx;
	float *values;
	unsigned int nnz;
}
