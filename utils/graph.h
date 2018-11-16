#pragma once
class Graph
{
	public:
		double *adj, *degree, volume;
		int size;
		bool directed;
		Graph(int);
		void add_edge(int, int, float);
		void print_degree();
		void print_graph();
};
