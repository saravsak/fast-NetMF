#include "spgraph.h"
#include<iostream>
#include<stdio.h>
#include<cstring>
SpGraph::SpGraph(int N, int nnz)
{
	this->adj.row_id = (int *) malloc((N+1) * sizeof(int));
	this->adj.row_id[N] = nnz; // Set last element to be nnz
	this->adj.col_idx = (int *) malloc(nnz * sizeof(int));
	this->adj.values = (float *) malloc(nnz * sizeof(float));

	this->adj.nnz = nnz;

	this->degree.row_id = (int *) malloc((N+1) * sizeof(int));
	this->degree.row_id[N] = N; // Set last element to be nnz of degree matrix = N
	this->degree.col_idx = (int *) malloc(N * sizeof(int));
	this->degree.values = (float *) malloc(N * sizeof(float));
	this->degree.nnz = N;

	for(int i=0;i<N;i++){
		this->degree.row_id[i] = i;
		this->degree.col_idx[i] = i;
		this->degree.values[i] = 0;
	}

	this->volume = 0;
	this->size = N;
	this->num_edges = 0;
	this->directed = false;

	// Set all memory locations to zero
	// TODO: For arrays
	


}

void SpGraph::add_node(int node, int *neighbors, float *weight, int num_neighbors)
{
	for(int i=0;i<num_neighbors;i++){
		this->volume+=weight[i];
	}

	// Add edges to adj mat
	
	this->adj.row_id[node] = this->num_edges;
	
	for(int i=0;i<num_neighbors;i++){
		this->adj.col_idx[num_edges + i] = neighbors[i];
		this->adj.values[num_edges + i] = weight[i]; 
		// Add edges to degree mat
		this->degree.values[node] += weight[i];
	}

	// Update number of edges
	this->num_edges += num_neighbors;
	
}
void SpGraph::info(){
	std::cout<<"**********************************"<<std::endl;
	std::cout<<"Number of nodes: "<<this->size<<std::endl;
	std::cout<<"Number of edges: "<<std::endl;
	std::cout<<"Sparsity: "<<num_edges/((float)this->size)<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}

