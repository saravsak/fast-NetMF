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
	const char * mode;
	milliseconds iptime;
	milliseconds normalization;
	milliseconds compute_s;
	milliseconds compute_m;
	milliseconds svd;
	milliseconds tot;
};

void log(const char* message);
double diff_ms(timeval t1, timeval t2);
