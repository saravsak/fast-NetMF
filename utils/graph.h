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

struct csr{
	// Device variables
	DT *d_values;
	int *d_rowIndices;
	int *d_colIndices;
	int *d_nnzPerVector;

	// Host variables
	DT *h_values;
	int *h_rowIndices;
	int *h_colIndices;
	int *h_nnzPerVector;
	int nnz;
	int lda;
};

class Graph
{
	public:
		int size;
		int num_edges;
		DT volume;

		// Dense
		DT *adj, *degree;
		
		// Sparse
		csr adj_csr, degree_csr;	
			
		bool directed;
		Graph(int);
		Graph(int, int);
		void add_edge(int, int, DT);
		void print_degree();
		void print_graph();
		void info();
};
