#pragma once
#include "utils.h"
class SpGraph
{
	public:
		CSR adj, degree;

		float volume;
		int size;
		float num_edges;
		bool directed;

		SpGraph(unsigned int, unsigned int);
		void add_node(unsigned int, unsigned int*, float*, unsigned int);
		
		void print_degree();
		void print_graph();
		void info();
};
