#pragma once

#include "graph.h"
#include "utils.h"

#include<string>

Graph read_graph(std::string, std::string);
Graph read_graph_from_mat(std::string);
Graph read_graph_from_metis(std::string);
Graph read_graph_from_edgelist(std::string);
void write_embeddings(const char *, double *embeddings, int size, int dim);
void write_profile(const char *, info profile);

struct csr{
	// Device variables
	double *d_values;
	int *d_rowIndices;
	int *d_colIndices;
	int *d_nnzPerVector;

	// Host variables
	double *h_values;
	int *h_rowIndices;
	int *h_colIndices;
	int *h_nnzPerVector;
	int nnz;
	int lda;
};
