#pragma once

#include "graph.h"
#include "utils.h"

#include<string>

//Graph read_graph(std::string, std::string);
//Graph read_graph_from_mat(std::string);
//Graph read_graph_from_metis(std::string);
//Graph read_graph_from_edgelist(std::string);
void write_embeddings(const char *, float *embeddings, int size, int dim);
void write_profile(const char *, info profile);
