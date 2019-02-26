#pragma once
#include<string.h>
#include<fstream>
#include<iostream>
#include <bits/stdc++.h> 
#include<vector>
#include "graph.h"
#include<stdio.h>
#include<cstring>
#include <iostream>
class Graph
{
	public:
		double *adj, *degree, volume;
		int size;
		float num_edges;
		bool directed;
		Graph(int);
		void add_edge(int, int, double);
		void print_degree();
		void print_graph();
		void info();
};
