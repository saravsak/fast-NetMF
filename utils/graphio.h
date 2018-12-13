#pragma once

#include "graph.h"
#include "spgraph.h"
#include<string>

Graph read_graph(std::string, std::string);
Graph read_graph_from_mat(std::string);
Graph read_graph_from_metis(std::string);
Graph read_graph_from_edgelist(std::string);
