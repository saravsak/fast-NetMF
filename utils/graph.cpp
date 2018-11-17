#include "graph.h"
#include<iostream>
#include<stdio.h>
#include<cstring>
Graph::Graph(int N)
{
	this->adj = (double*) malloc (N * N * sizeof(double));
	this->degree = (double*) malloc (N * N * sizeof(double));
	this->size = N;

	for(int i=0; i < this->size; i++)
		for(int j=0; j< this->size; j++)
			this->adj[i * this->size + j] = 0;

	for(int i=0; i < this->size; i++)
		for(int j=0; j< this->size; j++)
			this->degree[i * this->size + j] = 0;
	
	// Always false, since NetMF assumes undirected graphs
	this->directed = false;
}

void Graph::add_edge(int i, int j, float weight=1.0)
{
	/* NOTE: This function should add ONLY one edge. Use this function twice to make the matrix symmetric */

//	if( i > this->size || j > this->size)
//		throw "Cannot add edge. Invalid node.";
//
//	if( adj[i * this->size + j] != 0 || adj[j * this->size + i] != 0)
//		throw "Trying to add duplicate node. Verify dataset";
//
	this->adj[i * this->size + j] = weight;		
	this->degree[i * this->size + i] += weight;

	this->volume += weight;
}
void Graph::print_graph()
{
	for(int i = 0; i < this->size; i++){
		for(int j = 0; j < this->size; j++){
			std::cout << this->adj[i * this->size + j] << " ";
		}
		std::cout << std::endl;
	}
}
void Graph::print_degree()
{
	for(int i = 0; i < this->size; i++){
		for(int j = 0; j < this->size; j++){
			std::cout << this->degree[i * this->size + j] << " ";
		}
		std::cout << std::endl;
	}
}
void Graph::info(){
	std::cout<<"**********************************"<<std::endl;
	std::cout<<"Number of nodes: "<<this->size<<std::endl;
	std::cout<<"Number of edges: "<<std::endl;
	std::cout<<"Sparsity: "<<num_edges/((float)this->size)<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}

