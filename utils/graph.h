#pragma once
class Graph
{
	public:
		float *adj, *degree, volume;
		int size;
		float num_edges;
		bool directed;
		Graph(int);
		void add_edge(int, int, float);
		void print_degree();
		void print_graph();
		void info();
};
