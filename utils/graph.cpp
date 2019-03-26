#include<string.h>
#include<fstream>
#include<iostream>
#include <bits/stdc++.h> 
#include<vector>
#include "graph.h"
#include<stdio.h>
#include<cstring>
#include "utils.h"

#include "mkl.h"


Graph::Graph(int N )
{
	this->adj = (DT*) malloc (N * N * sizeof(DT));
	this->degree = (DT*) malloc (N * sizeof(DT));
	
	this->size = N;
	this->directed = false; // Always false, since NetMF assumes undirected graphs 

	memset(this->adj, 0, N*N*sizeof(DT));
	memset(this->degree, 0, N*sizeof(DT));

	this->volume = 0;
	//for(int i=0; i < this->size; i++)
	//	for(int j=0; j< this->size; j++)
	//		this->adj[i * this->size + j] = 0;

	//for(int i=0; i < this->size; i++)
	//	this->degree[i * this->size + i] = -1;
	
}

Graph::Graph(int N, int E, bool mode )
{
	this->size = N;
	this->num_edges = E;
	this->volume = 0;
	this->directed = false; // Always false, since NetMF assumes undirected graphs 

	if(mode){
		this->adj_mkl = (DT*) malloc (N * N * sizeof(DT));
		this->degree_mkl = (DT*) malloc (N * N * sizeof(DT));

		memset(this->adj_mkl, 0, N*N*sizeof(DT));
		memset(this->degree_mkl, 0,N* N*sizeof(DT));
	}else{
		this->adj_csr.h_rowIndices = (int *) malloc((N) * sizeof(int));
		this->adj_csr.h_rowEndIndices = (int *) malloc((N) * sizeof(int));
		this->adj_csr.h_colIndices = (int *) malloc(E * sizeof(int));
		this->adj_csr.h_values = (DT *) malloc(E * sizeof(DT));
		this->adj_csr.nnz = E;
	
		this->degree_csr.h_rowIndices = (int *) malloc((N) * sizeof(int));
		this->degree_csr.h_rowEndIndices = (int *) malloc((N) * sizeof(int));
		this->degree_csr.h_colIndices = (int *) malloc(N * sizeof(int));
		this->degree_csr.h_values = (DT *) malloc(N * sizeof(DT));
		this->degree_csr.nnz = N;
	}
	//memset(this->adj, 0, N*N*sizeof(DT));
	//memset(this->degree, 0, N*sizeof(DT));

	//for(int i=0; i < this->size; i++)
	//	for(int j=0; j< this->size; j++)
	//		this->adj[i * this->size + j] = 0;

	//for(int i=0; i < this->size; i++)
	//	this->degree[i * this->size + i] = -1;
	
}


Graph::Graph(int N, int E )
{

	this->size = N;
	this->num_edges = E;
	this->volume = 0;

	this->directed = false; // Always false, since NetMF assumes undirected graphs 
	
	this->adj_csr.h_rowIndices = (int *) malloc((N+1) * sizeof(int));
	this->adj_csr.h_colIndices = (int *) malloc(E * sizeof(int));
	this->adj_csr.h_values = (DT *) malloc(E * sizeof(DT));
	this->adj_csr.nnz = E;

	this->degree_csr.h_rowIndices = (int *) malloc((N+1) * sizeof(int));
	this->degree_csr.h_colIndices = (int *) malloc(N * sizeof(int));
	this->degree_csr.h_values = (DT *) malloc(N * sizeof(DT));
	this->degree_csr.nnz = N;

	//memset(this->adj, 0, N*N*sizeof(DT));
	//memset(this->degree, 0, N*sizeof(DT));

	//for(int i=0; i < this->size; i++)
	//	for(int j=0; j< this->size; j++)
	//		this->adj[i * this->size + j] = 0;

	//for(int i=0; i < this->size; i++)
	//	this->degree[i * this->size + i] = -1;
	
}


void Graph::add_edge(int i, int j, DT weight=1.0)
{
	/* 
  	* NOTE: This function should add only one edge. 
 	* Use this function twice in io to make the 
 	* matrix symmetric. 
 	*/

	//if(i!=j){
		if(this->adj[i * this->size + j]==0){
			this->adj[i * this->size + j] = weight;
			this->volume += weight;
			if(i!=j){
				this->degree[i * this->size + i] += weight;
				//this->degree1D[i] += weight;
			}
		}
		//if(this->degree[i * this->size + i] == -1)
		//	this->degree[i * this->size + i] = 0;
	//}
}
void Graph::info(){
	std::cout<<"**********************************"<<std::endl;
	std::cout<<"Number of nodes: "<<this->size<<std::endl;
	std::cout<<"Number of edges: "<<std::endl;
	std::cout<<"Sparsity: "<<num_edges/((float)this->size)<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}

