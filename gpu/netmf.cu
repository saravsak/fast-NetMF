#include "../utils/graph.h"
#include "../utils/io.h"

#define DEBUG false
#define VERBOSE false

int main ( void ){
	/* Section 0: Preliminaries */

	/* Settings */
	int window_size = 3;
	int dimension = 128;
	int b = 1;

	/* Load graph */
        log("Reading data from file");
	
	//Graph g =  read_graph("../data/test/small_test.csv","edgelist");
	//Graph g =  read_graph("../data/ppi/ppi.edgelist","edgelist");
	Graph g =  read_graph("../data/blogcatalog/edges.csv","edgelist");


}
