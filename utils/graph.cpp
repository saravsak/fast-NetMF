#include<string.h>
#include<fstream>
#include<iostream>
#include <bits/stdc++.h> 
#include<vector>
#include "graph.h"
#include<stdio.h>
#include<cstring>

Graph::Graph(int N )
{
	this->adj = (double*) malloc (N * N * sizeof(double));
	this->degree = (double*) malloc (N * N * sizeof(double));
	this->degree1D = (double*) malloc (N * sizeof(double));
	
	this->size = N;
	this->directed = false; // Always false, since NetMF assumes undirected graphs 

	memset(this->adj, 0, N*N*sizeof(double));
	memset(this->degree, 0, N*N*sizeof(double));
	memset(this->degree1D, 0, N*sizeof(double));

	this->volume = 0;
	//for(int i=0; i < this->size; i++)
	//	for(int j=0; j< this->size; j++)
	//		this->adj[i * this->size + j] = 0;

	//for(int i=0; i < this->size; i++)
	//	this->degree[i * this->size + i] = -1;
	
}

void Graph::add_edge(int i, int j, double weight=1.0)
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
		}
		//if(this->degree[i * this->size + i] == -1)
		//	this->degree[i * this->size + i] = 0;
		if(i!=j){
			this->degree[i * this->size + i] += weight;
			this->degree1D[i] += weight;
		}
	//}
}
void Graph::info(){
	std::cout<<"**********************************"<<std::endl;
	std::cout<<"Number of nodes: "<<this->size<<std::endl;
	std::cout<<"Number of edges: "<<std::endl;
	std::cout<<"Sparsity: "<<num_edges/((float)this->size)<<std::endl;
	std::cout<<"**********************************"<<std::endl;
}

