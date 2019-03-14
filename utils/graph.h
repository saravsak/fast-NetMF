#pragma once
#include<string.h>
#include<fstream>
#include<iostream>
#include <bits/stdc++.h> 
#include<vector>
#include<stdio.h>
#include<cstring>
#include <iostream>

//typedef double DT;
typedef float DT;

class Graph
{
	public:
		DT *adj, *degree, *degree1D, volume;
		int size;
		DT num_edges;
		bool directed;
		Graph(int);
		void add_edge(int, int, DT);
		void print_degree();
		void print_graph();
		void info();
};
