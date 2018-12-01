#pragma once

#include<time.h>
#include<chrono>

typedef std::chrono::milliseconds milliseconds;

struct CSR
{
	unsigned int* row_indx{};
        unsigned int* col_id{}; 	
	float* values{};

	unsigned int nrows{};
	unsigned int ncols{};
	unsigned int nnz{};
};

struct CSC
{
        unsigned int* col_indx{}; 	
	unsigned int* row_id{};
	float* values{};

	unsigned int nrows{};
	unsigned int ncols{};
	unsigned int nnz{};
};

struct COO
{
	unsigned int* row_id{};
        unsigned int* col_id{}; 	
	float* values{};

	unsigned int nrows{};
	unsigned int ncols{};
	unsigned int nnz{};
};

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
