#include "graph.h"
#include "utils.h"

#include<iostream>
#include<stdio.h>
#include<cstring>

Graph::Graph(int N, int nnz)
{	
	// Declare the adjacency matrix and the
	// degree matrix in CSR format
	this->adj.nrows = N;
	this->adj.ncols = N;
	this->adj.nnz = nnz;

	this->adj.values = (float *)malloc(nnz * sizeof(float));
	this->adj.col_id = (unsigned int *)malloc(nnz * sizeof(unsigned int));
	this->adj.row_id = (unsigned int *)malloc(nnz * sizeof(unsigned int));

	this->degree.nrows = N;
	this->degree.ncols = N;
	this->degree.nnz = N;
	
	this->degree.values = (float *)malloc(nnz * sizeof(float));
	this->degree.col_id = (unsigned int *)malloc(nnz * sizeof(unsigned int));
	this->degree.row_id = (unsigned int *)malloc(nnz * sizeof(unsigned int));

	// Initialize all memory locations to value 0
	memset(this->adj.values, 0, nnz * sizeof(float));
	memset(this->adj.col_id, 0, nnz * sizeof(unsigned int));
	memset(this->adj.row_id, 0, nnz * sizeof(unsigned int));

	memset(this->degree.values, 0, nnz * sizeof(float));
	memset(this->degree.col_id, 0, nnz * sizeof(unsigned int));
	memset(this->degree.row_id, 0, nnz * sizeof(unsigned int));

	this->size = N;
	
	// Always false, since NetMF assumes undirected graphs
	this->directed = false; 
	this->current_edge = 0;

}

void Graph::add_edge(int i, int j, float weight=1.0)
{
	/* 
  	* NOTE: This function should add only one edge. 
 	* Use this function twice in io to make the 
 	* matrix symmetric. 
 	*/

	this->adj.values[this->current_edge] = weight;		
	this->adj.row_id[this->current_edge] = i;		
	this->adj.col_id[this->current_edge] = j;		

	this->degree.values[this->current_edge] += weight;
	this->degree.row_id[this->current_edge] += i;
	this->degree.col_id[this->current_edge] += j;


	this->current_edge += 1;

	this->volume += weight;
}
void Graph::info(){
	std::cout<<"**********************************"<<std::endl;
	std::cout<<"Number of nodes: "<<this->size<<std::endl;
	std::cout<<"Number of edges: "<<std::endl;
	std::cout<<"Sparsity: "<<num_edges/((float)this->size)<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}

