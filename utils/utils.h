#pragma once

#include<time.h>
#include<chrono>

#include "graph.h"

typedef std::chrono::milliseconds milliseconds;

struct info {
	const char * dataset;
	const char * algo;
	DT dimension;
	DT window_size;
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
