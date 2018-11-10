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
	if( i > this->size || j > this->size)
		throw "Cannot add edge. Invalid node.";

	this->adj[i * this->size + j] = weight;		
	this->adj[j * this->size + i] = weight;

	this->degree[j * this->size + i] += 1;
	this->degree[i * this->size + j] += 1;

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
