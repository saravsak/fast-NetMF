#pragma once

#include "utils.h"

class Graph
{
	public:
		// Graph metadata
		int size;
		float num_edges;
		float volume;
		bool directed;
		
		// Adjacency and degree matrices
		COO adj;
		COO degree;

		Graph(int, int);
		void add_edge(int, int, float);
		void print_degree();
		void print_graph();
		void info();

	private:
		int current_edge;

};
