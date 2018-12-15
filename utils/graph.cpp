#include "graph.h"
#include<iostream>
#include<stdio.h>
#include<cstring>
Graph::Graph(int N)
{
	this->adj = (float*) malloc (N * N * sizeof(float));
	this->degree = (float*) malloc (N * N * sizeof(float));

	this->size = N;
	this->directed = false; // Always false, since NetMF assumes undirected graphs 

	memset(this->adj, 0, N*N*sizeof(float));
	memset(this->degree, 0, N*N*sizeof(float));

	//for(int i=0; i < this->size; i++)
	//	for(int j=0; j< this->size; j++)
	//		this->adj[i * this->size + j] = 0;

	//for(int i=0; i < this->size; i++)
	//	for(int j=0; j< this->size; j++)
	//		this->degree[i * this->size + j] = 0;
	
}

void Graph::add_edge(int i, int j, float weight=1.0)
{
	/* 
  	* NOTE: This function should add only one edge. 
 	* Use this function twice in io to make the 
 	* matrix symmetric. 
 	*/

	this->adj[i * this->size + j] = weight;		
	this->degree[i * this->size + i] += weight;

	this->volume += weight;
}
void Graph::info(){
	std::cout<<"**********************************"<<std::endl;
	std::cout<<"Number of nodes: "<<this->size<<std::endl;
	std::cout<<"Number of edges: "<<std::endl;
	std::cout<<"Sparsity: "<<num_edges/((float)this->size)<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}

