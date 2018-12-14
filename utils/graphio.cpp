#include "graph.h"
#include "spgraph.h"
#include "graphio.h"

#include<string.h>
#include<fstream>
#include<iostream>
#include <bits/stdc++.h> 
#include<vector>


SpGraph read_sparse_graph(std::string filename, std::string file_format){
	// TODO: Add check for unknown graph format

	if(!file_format.compare("metis"))
		return read_sparse_graph_from_metis(filename);

	if(!file_format.compare("mat"))
		return read_sparse_graph_from_mat(filename);

	if(!file_format.compare("edgelist"))
		return read_sparse_graph_from_edgelist(filename);
}

SpGraph read_sparse_graph_from_metis(std::string filename){
	/* TODO: Placeholder functions */
	SpGraph G(4, 4);
	return G;
}
SpGraph read_sparse_graph_from_mat(std::string filename){
	/* TODO: Placeholder functions */
	SpGraph G(4, 4);
	return G;
}
SpGraph read_sparse_graph_from_edgelist(std::string filename){
	// Open file
	std::ifstream inFile;
	inFile.open(filename);

	std::string line;

	std::map<std::string, int> node_mapping;
	std::map<int, std::vector<int>> adj_list;

	int node_num = 0, source=0, target=0;
	std::vector<std::string> nodes;
	std::string temp;
	int num_edges=0;	
	while(inFile>>line){
		
		// Split the string into 2 nodes
		std::stringstream stream(line);
		nodes.clear();
		
		//TODO: Dynamic delimiters
		while(getline(stream, temp, ',')){
			nodes.push_back(temp);
		}

		// If node does not exist in map, add it
		if(node_mapping.find(nodes[0]) == node_mapping.end()){
			node_mapping.insert( std::pair<std::string, int>( nodes[0], node_num++  ));
		}
		if(node_mapping.find(nodes[1]) == node_mapping.end()){
			node_mapping.insert( std::pair<std::string, int>( nodes[1], node_num++  ));
		}

		
		// Add the node to the adjacency list
		source = node_mapping.find(nodes[0])->second;	
		target = node_mapping.find(nodes[1])->second;

		if(adj_list.find(source) == adj_list.end()){
			adj_list[source] = std::vector<int>();
		}

		if(adj_list.find(target) == adj_list.end()){
			adj_list[target] = std::vector<int>();
		}
		
		adj_list.find(source)->second.push_back(target);
		adj_list.find(target)->second.push_back(source);		
		num_edges+=2;
	}

	std::map<int, std::vector<int>>::iterator it;
        SpGraph G(node_mapping.size(), num_edges);	
	
	for(it = adj_list.begin(); it != adj_list.end(); it++){
		source = it->first;
		float temp[it->second.size()];
		memset(temp, 1.0, it->second.size()); 
		G.add_node(source, &it->second[0], temp, it->second.size());
		delete[] temp;	
	} 

	return G;
}


Graph read_graph(std::string filename, std::string format){
	// TODO: Add check for unknown graph format

	if(!format.compare("metis"))
		return read_graph_from_metis(filename);

	if(!format.compare("mat"))
		return read_graph_from_mat(filename);

	if(!format.compare("edgelist"))
		return read_graph_from_edgelist(filename);
}

Graph read_graph_from_metis(std::string filename){
	/* TODO: Placeholder functions */
	Graph G(4);
	return G;
}

Graph read_graph_from_mat(std::string filename){
	/* TODO: Placeholder functions */
	Graph G(4);
	return G;
}

Graph read_graph_from_edgelist(std::string filename){
	
	// Open file
	std::ifstream inFile;
	inFile.open(filename);

	std::string line;

	std::map<std::string, int> node_mapping;
	std::map<int, std::vector<int>> adj_list;

	int node_num = 0, source=0, target=0;
	std::vector<std::string> nodes;
	std::string temp;
	
	while(inFile>>line){
		
		// Split the string into 2 nodes
		std::stringstream stream(line);
		nodes.clear();
		
		//TODO: Dynamic delimiters
		while(getline(stream, temp, ',')){
			nodes.push_back(temp);
		}

		// If node does not exist in map, add it
		if(node_mapping.find(nodes[0]) == node_mapping.end()){
			node_mapping.insert( std::pair<std::string, int>( nodes[0], node_num++  ));
		}
		if(node_mapping.find(nodes[1]) == node_mapping.end()){
			node_mapping.insert( std::pair<std::string, int>( nodes[1], node_num++  ));
		}

		
		// Add the node to the adjacency list
		source = node_mapping.find(nodes[0])->second;	
		target = node_mapping.find(nodes[1])->second;

		if(adj_list.find(source) == adj_list.end()){
			adj_list[source] = std::vector<int>();
		}

		if(adj_list.find(target) == adj_list.end()){
			adj_list[target] = std::vector<int>();
		}
		
		adj_list.find(source)->second.push_back(target);
		adj_list.find(target)->second.push_back(source);		

	}

	std::map<int, std::vector<int>>::iterator it;
        Graph G(node_mapping.size());	
	
	for(it = adj_list.begin(); it != adj_list.end(); it++){
		source = it->first;
		for(std::vector<int>::iterator nit = it->second.begin(); nit!= it->second.end(); ++nit){
			G.add_edge(source, *nit, 1.0);
		}
	} 

	return G;
}
