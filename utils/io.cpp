#include "graph.h"
#include "io.h"
#include "utils.h"

#include<string.h>
#include<fstream>
#include<iostream>
#include <bits/stdc++.h> 
#include<vector>

bool check_file(const char * fileName){
	std::ifstream infile(fileName);
	return infile.good();
}

void write_embeddings(const char * fileName, DT *embeddings, int size, int dim){
	// Assumes embeddings are stored in column major format
	std::ofstream op;
	op.open(fileName);
	op<<size<<" "<<dim<<std::endl;
	for(int j=0;j<size;j++){
		op << j << " ";
		for(int i=0;i<dim;i++){
			op << embeddings[i*size + j] << " ";	
		}
		op<<std::endl;
	}
	op.close();

}
void write_embeddings(const char * fileName, double **embeddings, int size, int dim){
	// Assumes embeddings are stored in column major format
	std::ofstream op;
	op.open(fileName);
	op<<size<<" "<<dim<<std::endl;
	for(int i=0;i<size;i++){
		op << i << " ";
		for(int j=0;j<dim;j++){
			op << embeddings[i][j] << " ";	
		}
		op<<std::endl;
	}
	op.close();

}
void write_profile(const char * fileName, info profile){
	std::ofstream op;
	double tot = profile.iptime.count()
			+ profile.init.count()
			+ profile.gpuio.count()
			+ profile.compute_d.count()
			+ profile.compute_x.count()
			+ profile.compute_s.count()
			+ profile.compute_m.count()
			+ profile.svd.count()
			+ profile.emb.count();
	
	if(!check_file(fileName)){
		std::cout<<"File does not exist. Creating file"<<std::endl;
		op.open(fileName, std::ofstream::out);
		op<<"dataset,algo,dimension,window size,i/p,init,gpuio,D, X, S, M,SVD, Emb, total"<<std::endl;
	}else{
		op.open(fileName, std::ofstream::app);
	}


	op<<profile.dataset<<","
		<<profile.algo<<","
		<<profile.dimension<<","
		<<profile.window_size<<","
		<<profile.iptime.count()<<","
		<<profile.init.count()<<","
		<<profile.gpuio.count()<<","
		<<profile.compute_d.count()<<","
		<<profile.compute_x.count()<<","
		<<profile.compute_s.count()<<","
		<<profile.compute_m.count()<<","
		<<profile.svd.count()<<","
		<<profile.emb.count()<<","
		<<tot<<std::endl;	
	op.close();
}
Graph read_graph(std::string filename, std::string format, const char *mapping_filename){
	// TODO: Add check for unknown graph format


	if(!format.compare("dense"))
		return read_graph_from_dense(filename);

	if(!format.compare("mkl-dense"))
		return read_graph_from_mkl_dense(filename);
	
	if(!format.compare("csr"))
		return read_graph_from_csr(filename);

	if(!format.compare("edgelist"))
		return read_graph_from_edgelist(filename, mapping_filename);
}

Graph read_graph_from_csr(std::string filename){
	std::string line;

	std::ifstream metaFile;
	std::string metaFilename = filename + "meta.bin";
	metaFile.open(metaFilename);

	std::cout<<"Reading metadata";

	int num_nodes;
	int num_edges;
	int volume;

	metaFile >> line;

	std::string temp;
	std::stringstream stream(line);

	getline(stream,temp, ',');
	num_nodes = stoi(temp);
	
	getline(stream,temp, ',');
	num_edges = stoi(temp);
	
	getline(stream,temp, ',');
	volume = stoi(temp);

	Graph g(num_nodes, num_edges);
	g.volume = volume;
	
	std::ifstream adjFile;
	std::string adjFilename = filename + "adj_vals.bin";
	adjFile.open(adjFilename);

	while(adjFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.adj_csr.h_values[i] = stof(temp);
			i++;
		}

	}	
	adjFile.close();
	
	adjFilename = filename + "adj_cols.bin";
	adjFile.open(adjFilename);
	
	while(adjFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.adj_csr.h_colIndices[i] = stoi(temp);
			i++;
		}

	}	
	adjFile.close();

	adjFilename = filename + "adj_rows_start.bin";
	adjFile.open(adjFilename);
	
	while(adjFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.adj_csr.h_rowIndices[i] = stoi(temp);
			i++;
		}

	}	
	adjFile.close();

	g.adj_csr.nnz = num_edges;
	g.adj_csr.h_rowIndices[num_nodes] = num_edges;

	std::ifstream degFile;
	std::string degFilename = filename + "degree_vals.bin";
	degFile.open(degFilename);

	while(degFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.degree_csr.h_values[i] = stof(temp);
			i++;
		}

	}	
	degFile.close();
	
	degFilename = filename + "degree_cols.bin";
	degFile.open(degFilename);
	
	while(degFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.degree_csr.h_colIndices[i] = stoi(temp);
			i++;
		}

	}	
	degFile.close();

	degFilename = filename + "degree_rows_start.bin";
	degFile.open(degFilename);
	
	while(degFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.degree_csr.h_rowIndices[i] = stoi(temp);
			i++;
		}

	}	
	degFile.close();

	g.degree_csr.h_rowIndices[num_nodes] = num_nodes;
	g.degree_csr.nnz = num_nodes;

	return g;

}

Graph read_graph_from_mkl_dense(std::string filename){
	std::string line;

	std::ifstream metaFile;
	std::string metaFilename = filename + "meta.bin";
	metaFile.open(metaFilename);

	std::cout<<"Reading metadata";

	int num_nodes;
	int num_edges;
	int volume;

	metaFile >> line;

	std::string temp;
	std::stringstream stream(line);

	getline(stream,temp, ',');
	num_nodes = stoi(temp);
	
	getline(stream,temp, ',');
	num_edges = stoi(temp);
	
	getline(stream,temp, ',');
	volume = stoi(temp);
	
	std::cout<<"Reading metadata";
	std::ifstream adjFile;
	std::string adjFilename = filename + "adj.bin";
	adjFile.open(adjFilename);
	
	Graph g(num_nodes, num_edges, true);

	g.volume = volume;

	int source;
	int target;
	int weight;
	
	while(adjFile>>line){
		std::stringstream st(line);
		
		getline(st, temp, ',');
		source = stoi(temp);

		getline(st, temp, ',');
		target = stoi(temp);

		getline(st, temp, ',');
		weight = stof(temp);

		g.adj_mkl[source * num_nodes + target] = weight;
	}

	std::cout<<"Reading metadata";
	std::ifstream degFile;
	std::string degFilename = filename + "degree_diag.bin";
	degFile.open(degFilename);
	
	while(degFile>>line){
		std::stringstream st(line);
		
		getline(st, temp, ',');
		source = stoi(temp);

		getline(st, temp, ',');
		target = stoi(temp);

		getline(st, temp, ',');
		weight = stof(temp);

		g.degree_mkl[source * num_nodes + target] = weight;
	}

	return g;
}



Graph read_graph_from_dense(std::string filename){
	std::string line;

	std::ifstream metaFile;
	std::string metaFilename = filename + "meta.bin";
	metaFile.open(metaFilename);

	std::cout<<"Reading metadata";

	int num_nodes;
	int num_edges;
	int volume;

	metaFile >> line;

	std::string temp;
	std::stringstream stream(line);

	getline(stream,temp, ',');
	num_nodes = stoi(temp);
	
	getline(stream,temp, ',');
	num_edges = stoi(temp);
	
	getline(stream,temp, ',');
	volume = stoi(temp);
	
	std::cout<<"Reading metadata";
	std::ifstream adjFile;
	std::string adjFilename = filename + "adj.bin";
	adjFile.open(adjFilename);
	
	Graph g(num_nodes);

	g.volume = volume;

	int source;
	int target;
	int weight;
	
	while(adjFile>>line){
		std::stringstream st(line);
		
		getline(st, temp, ',');
		source = stoi(temp);

		getline(st, temp, ',');
		target = stoi(temp);

		getline(st, temp, ',');
		weight = stof(temp);

		g.adj[source * num_nodes + target] = weight;
	}

	std::cout<<"Reading metadata";
	std::ifstream degFile;
	std::string degFilename = filename + "degree.bin";
	degFile.open(degFilename);

	while(degFile >> line){
		std::stringstream st(line);
	
		int i=0;	
		while(getline(st, temp, ',')){
			g.degree[i] = stof(temp);
			i++;
		}

	}	

	return g;
}

Graph read_graph_from_edgelist(std::string filename, const char *mapping_filename){
	
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
			node_mapping.insert( std::pair<std::string, int>( nodes[0], node_num  ));
			node_num++;
		}
		if(node_mapping.find(nodes[1]) == node_mapping.end()){
			node_mapping.insert( std::pair<std::string, int>( nodes[1], node_num  ));
			node_num++;
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
	
	std::map<std::string, int>::iterator new_it;
	std::ofstream mapfile(mapping_filename);	
	for(new_it = node_mapping.begin(); new_it != node_mapping.end(); new_it++){
		mapfile<< new_it->first << ":" << new_it->second <<std::endl;
	}	
	mapfile.close();

	return G;
}
