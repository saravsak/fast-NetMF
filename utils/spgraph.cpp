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
	std::cout<<"Number of edges: "<<this->adj.nnz<<std::endl;
	std::cout<<"Sparsity: "<<num_edges * 100/((float)this->size * this->size)<<"%"<<std::endl;
	std::cout<<"Volume: "<<this->volume<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}
void SpGraph::print_degree(){
	std::cout<<"\nPrinting Degree matrix"<<std::endl;
	std::cout<<"Values"<<std::endl;
	for(int i=0;i<this->degree.nnz;i++){
		std::cout<<this->degree.values[i]<<" ";
	}
	std::cout<<"\nCol Ids"<<std::endl;
	for(int i=0;i<this->degree.nnz;i++){
		std::cout<<this->degree.col_idx[i]<<" ";
	}
	std::cout<<"\nRow Ids"<<std::endl;
	for(int i=0;i<=this->size;i++){
		std::cout<<this->degree.row_id[i]<<" ";
	}
	std::cout<<'\n';
}
void SpGraph::print_graph(){
	std::cout<<"\nPrinting Adjacency matrix"<<std::endl;
	std::cout<<"Values"<<std::endl;
	for(int i=0;i<this->adj.nnz;i++){
		std::cout<<this->adj.values[i]<<" ";
	}
	std::cout<<"\nCol Ids"<<std::endl;
	for(int i=0;i<this->adj.nnz;i++){
		std::cout<<this->adj.col_idx[i]<<" ";
	}
	std::cout<<"\nRow Ids"<<std::endl;
	for(int i=0;i<=this->size;i++){
		std::cout<<this->adj.row_id[i]<<" ";
	}
	std::cout<<'\n';
}

