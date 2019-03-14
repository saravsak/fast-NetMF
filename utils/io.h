#pragma once

#include "graph.h"
#include "utils.h"

#include<string>

Graph read_graph(std::string, std::string, const char *mapping_filename);
Graph read_graph_from_mat(std::string);
Graph read_graph_from_metis(std::string);
Graph read_graph_from_edgelist(std::string, const char* mapping_filename);
void write_embeddings(const char *, DT *embeddings, int size, int dim);
void write_profile(const char *, info profile);

struct csr{
	// Device variables
	DT *d_values;
	int *d_rowIndices;
	int *d_colIndices;
	int *d_nnzPerVector;

	// Host variables
	DT *h_values;
	int *h_rowIndices;
	int *h_colIndices;
	int *h_nnzPerVector;
	int nnz;
	int lda;
};
