#include "spgraph.h"
#include<iostream>
#include<stdio.h>
#include<cstring>
SpGraph::SpGraph(unsigned int N, unsigned int nnz)
{
	this->adj.row_id = (unsigned int *) malloc((N+1) * sizeof(unsigned int));
	this->adj.row_id[N] = nnz; // Set last element to be nnz
	this->adj.col_idx = (unsigned int *) malloc(nnz * sizeof(unsigned int));
	this->adj.values = (float *) malloc(nnz * sizeof(float));

	this->nnz = nnz;

	this->degree.row_id = (unsigned int *) malloc((N+1) * sizeof(unsigned int));
	this->degree.row_id[N] = N; // Set last element to be nnz of degree matrix = N
	this->degree.col_idx = (unsigned int *) malloc(N * sizeof(unsigned int));
	this->degree.values = (float *) malloc(N * sizeof(float));

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

void SpGraph::add_node(unsigned int node, unsigned int *neighbors, float *weight, unsigned int num_neighbors)
{
	for(unsigned int i=0;i<num_neighbors;i++){
		this->volume+=weight[i];
	}

	// Add edges to adj mat
	
	this->adj.row_id[node] = this->num_edges;
	
	for(unsigned int i=0;i<num_neighbors;i++){
		this->adj.col_idx[num_edges + i] = neighbors[i];
		this->adj.values[num_edges + i] = weight[i]; 
		// Add edges to degree mat
		this->degree[node] += weight[i];
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

