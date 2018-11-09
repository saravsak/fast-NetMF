#include "graph.h"
#include "utils.h"
#include<iostream>

int main(){
	//Graph G = read_graph("../data/blogcatalog/edges.csv","edgelist");	
	Graph G = read_graph("../data/test/test.csv","edgelist");	
	//G.print_graph();
	std::cout<<G.volume;
}
