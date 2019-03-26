#pragma once

#include "graph.h"
#include "utils.h"

#include<string>

Graph read_graph(std::string, std::string, const char *mapping_filename);
Graph read_graph_from_edgelist(std::string, const char* mapping_filename);
Graph read_graph_from_dense(std::string);
Graph read_graph_from_csr(std::string);
Graph read_graph_from_mkl_dense(std::string);
Graph read_graph_from_mkl_sparse(std::string);
void write_embeddings(const char *, DT *embeddings, int size, int dim);
void write_embeddings(const char *, double **embeddings, int size, int dim);
void write_profile(const char *, info profile);

