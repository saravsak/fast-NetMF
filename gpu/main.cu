/* 
TODO: 
1. Change thread architecture
2. USe cuBlas for addition
*/

/* 
Question for prof
1. Copy stuff within GPU
2. Launch kernel without copy
3. Is it better to do more redundant work in one thread or one more kernle to do it once?
4. Results are wrong if I use same variable as result. Why?
*/
#include<stdlib.h>
#include<iostream>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>

#include "../utils/spgraph.h"
#include "../utils/graphio.h"

int main ( void ){

	/**************
 	* NetMF Sparse *
	**************/
	
	/* Load graph */
        std::cout<<"Reading data from file"<<std::endl;
	SpGraph g =  read_sparse_graph("../data/test/small_test.csv","edgelist");
	//g.print_degree();
	//g.print_graph();
}
