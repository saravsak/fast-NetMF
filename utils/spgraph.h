#pragma once
#include "utils.h"
class SpGraph
{
	public:
		CSR adj, degree;

		float volume;
		int size;
		int num_edges;
		bool directed;

		SpGraph(int, int);
		void add_node(int, int*, float*, int);
		
		void print_degree();
		void print_graph();
		void info();
};
