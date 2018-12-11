/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   main.cpp
 * Author: Mohit Kumar Jangid
 *
 * Created on October 27, 2018, 1:16 PM
 */

#include <iostream>
#include <string>
#include <fstream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "RedSVD-h"

#include <SymEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>


#define PRINT_MATRICES
// #define PRINT_ALGO_PARAMS
// #define PRINT_EXECUTION_TIME

namespace fs = boost::filesystem;
using namespace boost;
using namespace boost::timer;
using namespace Eigen;
using namespace std;
using namespace Spectra;
using namespace REDSVD;


int window_size = 10;
string print_prefix = "";

void print_timer(const string &message, cpu_timer &timer)
{
	// cout << std::right << std::setw(40) << message << " : " << std::right << std::setw(60) << timer.format()  << endl;
	// static variable
	static float last_wall;
	static float last_cpu;

	float wall = std::stof(timer.format(6, "%w"));
	float cpu = std::stof(timer.format(6, "%t"));

	// std::right << std::setw(15) <<
	// this is good for viewing
	// std::cout << std::right << std::setw(40) << message << std::right << std::setw(15) << (wall - last_wall) << " sec wall, " << std::right << std::setw(10) << (cpu - last_cpu) <<  " sec cpu,  Acc: " << std::right << std::setw(15) << wall << "sec wall, " <<  std::right << std::setw(10) <<  cpu << "sec cpu" << endl ;

	// this is good for storing into csv
	std::cout << print_prefix << ",, " << message << ",, " << (wall - last_wall) << ",, " << (cpu - last_cpu) << ",, " << wall << ",, " <<  cpu  << endl ;

	last_wall = wall;
	last_cpu = cpu;

}

struct EdgeProperties
{
	float weight;
};

// define function to be applied coefficient-wise
float summation_approximate(float x)
{

	// netmf logic
	//  evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window

	if (x >= 1)
		return 1 ;
	else
		//https://en.cppreference.com/w/c/numeric/math/pow
		return max((float)0, (x * (1 - powf(x, window_size))) / ((1 - x) * window_size ) );

	     // (.143883 * (1 - pow(.143883,10)))/ ((1-.143883)*10) = 0.01680646447306298
}
	




// define function to be applied coefficient-wise
float log_max_each_vs_1(float x)
{
	// https://en.cppreference.com/w/c/numeric/math/log
	// cout << "log 2 base e is :" << logf(2);
	// log 2 base e is :0.693147[

	// https://en.cppreference.com/w/cpp/algorithm/max

	return logf(max(x, (float)1));
}


MatrixXf netmf(Eigen::SparseMatrix<float> &A, const string &graph_size, unsigned long long int rank, int negative_sampling, unsigned long long int dim, cpu_timer &timer )
{

	double vol = (double) A.sum();
	unsigned long long int num_nodes = A.cols();
	unsigned long long int num_cols = num_nodes;
	unsigned long long int num_non_zeros = A.nonZeros(); 



	ArrayXf degree_vec(num_nodes, 1);


	// A: 
	// Nonzero entries:
	// (1,1) (2,2) (1,0) (1,2) (2,0) (1,1) (1,3) (1,2) 

	// Outer pointers:
	// 0 2 4 7  $

	// 0 1 2 0 
	// 1 0 1 0 
	// 2 1 0 1 
	// 0 0 1 0 
	#ifdef PRINT_MATRICES
	cout << "\nVol of matrix A is\n------------------------------\n" << vol << endl;
	#endif /* PRINT_MATRICES */
	
	// cout << "Third Col is " << A.innerVector(2).sum();


    // Filling sparse matrix when number of non zeros per column is known
	SparseMatrix<float> D_invsqrt(num_nodes, num_nodes);

	D_invsqrt.reserve(VectorXi::Constant(num_cols,1));

	for (unsigned long long int i = 0; i < num_cols ; i++)
		D_invsqrt.insert(i,i) = 1/sqrtf(A.innerVector(i).sum());

	D_invsqrt.makeCompressed(); 

	// cout << "D_invsqrt : " << D_invsqrt << endl; 


	// D_rt_inv.todense()
	// matrix([[0.57735027, 0.        , 0.        , 0.        ],
	//         [0.        , 0.70710678, 0.        , 0.        ],
	//         [0.        , 0.        , 0.5       , 0.        ],
	//         [0.        , 0.        , 0.        , 1.        ]])

	// 0.57735 0 0 0 
	// 0 0.707107 0 0 
	// 0 0 0.5 0 
	// 0 0 0 1 


	// normalized adjacency matrix
	SparseMatrix<float> norm_adj(num_nodes, num_nodes);
	norm_adj = D_invsqrt * A * D_invsqrt;

	#ifdef PRINT_MATRICES
	cout << "\nAdj norm\n------------------------\n" << endl << norm_adj  << endl;
	#endif /* PRINT_MATRICES */	

	// X.todense()
	// matrix([[0.        , 0.40824829, 0.57735027, 0.        ],
	//         [0.40824829, 0.        , 0.35355339, 0.        ],
	//         [0.57735027, 0.35355339, 0.        , 0.5       ],
	//         [0.        , 0.        , 0.5       , 0.        ]])

	// 0 0.408248 0.57735 0 
	// 0.408248 0 0.353553 0 
	// 0.57735 0.353553 0 0.5 
	// 0 0 0.5 0 


// 	print_timer("Normalized Adjacency Matrix", timer);


	MatrixXf M_cap(num_nodes, num_nodes);
	M_cap.setZero();
	// Approximate M cap

	if (graph_size == "large") {

		// Construct matrix operation object using the wrapper class SparseGenMatProd
		    SparseGenMatProd<float> op(norm_adj);

		     // https://spectralib.org/doc/classspectra_1_1symeigssolver
		    // look at above site for 3rd parameter
		    // Construct eigen solver object, requesting the largest three eigenvalues

		    unsigned long long int convergence_controller = (unsigned long long int) (2.1*rank); 

		    if ( (2.1 * rank) > num_nodes ){

		    	convergence_controller = num_nodes;
		    }

		    SymEigsSolver< float, LARGEST_ALGE, SparseGenMatProd<float>> eigs(&op, rank, convergence_controller);

		    // Initialize and compute
		    eigs.init();
		    int nconv = eigs.compute();


		    // diagonalized  eigen values
			DiagonalMatrix<float, Dynamic> evals_diag(rank);

			
		    if(eigs.info() == SUCCESSFUL){
		
		    	#ifdef PRINT_MATRICES
		    	cout << "\nEigen vals and vecs\n------------------------\n";
		    	std::cout << "\neval\n" <<eigs.eigenvalues().reverse().transpose() << std::endl;
		    	std::cout << "\nevec\n"<< ( eigs.eigenvectors().rowwise().reverse()) << std::endl;
		    	// cout << "summation_approximate :" << endl <<  ( ArrayXf) eigs.eigenvalues().reverse().unaryExpr(ptr_fun(summation_approximate)) << endl;
				#endif /* PRINT_MATRICES */	
				

			    #ifdef PRINT_EXECUTION_TIME	
				print_timer("Eigen Decomposition" , timer);
				 #endif /* PRINT_EXECUTION_TIME */


		        evals_diag.diagonal() << ( ( ArrayXf) eigs.eigenvalues().reverse().unaryExpr(ptr_fun(summation_approximate))).sqrt();
		    	// sqrt(.01680654) = 0.1296400401110706

		    	MatrixXf temp_m;
		    	temp_m = (evals_diag * (D_invsqrt * ( eigs.eigenvectors().rowwise().reverse())).transpose()).transpose();

		    	// this is euivalent to X in python approximate_deepwalk_matrix
		    	// cout << "temp_m is " << endl << temp_m << endl ;

		    	M_cap.triangularView<Upper>() = (temp_m * temp_m.transpose()); 

		    	M_cap.triangularView<Upper>() = (float)(vol / (float) negative_sampling) * M_cap;


		    }else{

				throw "Eigen Values and vector could not be solved.";
		    }

			// element wise log and maximum (each element, 1)
			// this is equivalent to Y in approximate_deepwalk_matrix


			M_cap.triangularView<Upper>() = M_cap.unaryExpr(ptr_fun(log_max_each_vs_1));

			M_cap.triangularView<StrictlyLower>() +=  M_cap.transpose();



	}
	else
		// graph size is small
	{
		// to be summed
		Eigen::SparseMatrix<float> S(num_nodes, num_nodes);
		S.setZero();

		Eigen::SparseMatrix<float>  X(num_nodes, num_nodes);
		X.setIdentity();

		for (unsigned long long int  i = 0; i < window_size; ++i)
		{
			X = X * norm_adj;
			S += X;

		// 	// cout << endl << "S " << i << " : " << endl << S <<  endl;
		}

		#ifdef PRINT_MATRICES
		cout << "\nSummed norm_adj\n---------------------------------\n" << S << endl ;
		#endif /* PRINT_MATRICES */	

		#ifdef PRINT_EXECUTION_TIME	
		print_timer("Summed Norm_adj" , timer);
		#endif /* PRINT_EXECUTION_TIME */

		// M_cap = D_invsqrt * (D_invsqrt * S).transpose()
		M_cap = (float)( vol / ((double)(window_size * negative_sampling)) ) * (D_invsqrt * (D_invsqrt * S).transpose());
		
		// element wise log and maximum (each element, 1)
		// this is equivalent to Y in approximate_deepwalk_matrix

		M_cap = M_cap.unaryExpr(ptr_fun(log_max_each_vs_1));

	}



	#ifdef PRINT_MATRICES
	cout << "\nMcap \n------------------------\n"  << M_cap << endl ;
	#endif /* PRINT_MATRICES */	

	#ifdef PRINT_EXECUTION_TIME	
	print_timer("Approximated M" , timer);
	#endif /* PRINT_EXECUTION_TIME */

	//----------------------------------------------------------------------------------------

	// SVD calculation
	//https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html
	//https://stats.stackexchange.com/questions/50663/what-is-a-thin-svd

// 

	REDSVD::RedSVD<Eigen::MatrixXf> svd(M_cap, dim);

	// sort singular values and matrix U by sorted order of singular values 
	 std::vector<unsigned long long int > sorted_indexes(dim);
	 std::size_t n(0);
	 std::generate(std::begin(sorted_indexes), std::end(sorted_indexes), [&]{ return n++; });

	 std::sort(  std::begin(sorted_indexes), 
	             std::end(sorted_indexes),
	             [&](unsigned long long int  i1, unsigned long long int  i2) { return svd.singularValues()[i1] < svd.singularValues()[i2]; } );

	// cout << "\nsort_indexes are \n----------------------------- \n";
	//  for (auto v : sorted_indexes)
	//          std::cout << v << ' ';


	 PermutationMatrix<Dynamic,Dynamic,int> perm(dim);

	 for (unsigned long long int  i = 0; i< dim; i++)
	     perm.indices()[i] = sorted_indexes[i];
	 #ifdef PRINT_MATRICES
	// the results are stored in particular members as shown https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
	cout << "\nSingular Values and U are\n------------------------\n";
	cout << "\nSingular Values are " << endl << svd.singularValues().transpose() * perm << endl ;
	cout << "\nComputed U  is " << endl << svd.matrixU() * perm << endl ;
	#endif /* PRINT_MATRICES */	

	#ifdef PRINT_EXECUTION_TIME	
	print_timer("SVD" , timer);
	#endif /* PRINT_EXECUTION_TIME */

	
	DiagonalMatrix<float, Dynamic> S_d(dim);
	S_d.diagonal() = ( perm * svd.singularValues() ).array().sqrt() ;

	#ifdef PRINT_MATRICES
	// cout << "\nSingular diagonal sqrt values\n---------------------------\n" << ( perm * svd.singularValues() ).array().sqrt() ;
	#endif /* PRINT_MATRICES */	

	return  (svd.matrixU() * perm) * S_d  ;

}



int main(int argc, char *argv[])
{


	try
	{

		fs::path dataset_name;
		fs::path dataset_folder;
		unsigned long long int rank;
		unsigned long long int dim;
		unsigned long long int window_size;
		float negative_sampling;
		std::string graph_size;


		program_options::options_description desc{"Options"};
		desc.add_options()
		("help,h", "Help screen")
		// requried options
		("dataset,d"    , program_options::value<fs::path>(&dataset_name)->required() , "dataset name e.g. PPI")
		("dataset_folder,f"        , program_options::value<fs::path>(&dataset_folder)->default_value("/home/jangid.6/work/datasets") , "path to dataset folder")
		// default options
		("rank,r"              , program_options::value<unsigned long long int>(&rank)->default_value(256) , "#eigenpairs used to approximate normalized graph laplacian.")
		("dim,d"               , program_options::value<unsigned long long int>(&dim)->default_value(128) , "dimension of embedding")
		("window,w"            , program_options::value<unsigned long long int>(&window_size)->default_value(10) , "context window size")
		("negative_sampling,n" , program_options::value<float>(&negative_sampling)->default_value(1.0) , "negative sampling")
		("graph_size,s"        , program_options::value<std::string>(&graph_size)->default_value("large") , "netmf large or small algorithm (e.g. small/large )");


		program_options::command_line_parser parser{argc, argv};
		parser.options(desc).allow_unregistered();
		program_options::parsed_options parsed_options = parser.run();

		program_options::variables_map vm;
		program_options::store(parsed_options, vm);

		// this overrise reuireded arg constraint for help
		if (vm.count("help") || vm.empty())
		{
			std::cout << desc << "\n";
			return false;
		}
		program_options::notify(vm); // throws on error, so do after help in case
		// there are any problems

		#ifdef PRINT_ALGO_PARAMS

		std::cout << "dataset_name : " << dataset_name << std::endl;
		std::cout << "rank : " << rank << std::endl;
		std::cout << "dim : " << dim << std::endl;
		std::cout << "window_size : " << window_size << std::endl;
		std::cout << "negative_sampling : " << negative_sampling << std::endl;
		std::cout << "graph_size : " << graph_size << std::endl;

		#endif /* PRINT_ALGO_PARAMS */

		typedef adjacency_list<vecS, vecS, undirectedS, no_property, EdgeProperties > Graph;
		boost::dynamic_properties dp;
		Graph g;


		// see how weight property can be retrieved in for loop below
		dp.property("weight", get(&EdgeProperties::weight, g));

		// connected_graph_200
		// connected_graph_500
		// connected_graph_1000
		// PPI
		// Blog
		// flickr


		fs::path dataset_file =  dataset_folder / dataset_name.replace_extension(".graphml") ;


		ifstream t(dataset_file.c_str());

		if (!t.is_open())
		{
			cout << "loading file (" << dataset_file << ") failed." << endl;
			throw "Could not load file.";
		}

		cpu_timer timer;

		read_graphml(t, g, dp);

		unsigned long long int num_nodes = boost::num_vertices(g);


		print_prefix = dataset_name.string() + ",, " + graph_size + ",, " + to_string(Eigen::nbThreads( )); 

		#ifdef PRINT_EXECUTION_TIME		
		print_timer("Graph Loaded from file", timer);
		#endif /* PRINT_EXECUTION_TIME */


		// Creating adjcency matrix A

		typedef Eigen::Triplet<float> T;
		std::vector<T> tripletList;
		tripletList.reserve(boost::num_edges(g));
		

		graph_traits<Graph>::edge_iterator ei, ei_end;

		// get the property map for vertex indices
		typedef property_map<Graph, vertex_index_t>::type IndexMap;
		IndexMap index = get(vertex_index, g);

		for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
		{

			// A(index[source(*ei, g)], index[target(*ei, g)]) = g[*ei].weight;
			// A( index[target(*ei, g)], index[source(*ei, g)]) = g[*ei].weight;

					

			tripletList.push_back(T(index[source(*ei, g)], index[target(*ei, g)], g[*ei].weight));
			tripletList.push_back(T( index[target(*ei, g)], index[source(*ei, g)], g[*ei].weight));
		}


		Eigen::SparseMatrix<float> A(num_nodes, num_nodes);
		A.setFromTriplets(tripletList.begin(), tripletList.end());
		// mat is ready to go!

		A.makeCompressed();

		#ifdef PRINT_ALGO_PARAMS

		cout << "num Edges:" << boost::num_edges(g) << endl;
		cout << "num Nodes:" << boost::num_vertices(g) << endl;
		cout << "Num Threads :" << Eigen::nbThreads( ) << endl; 
		#endif /* PRINT_ALGO_PARAMS */


		#ifdef PRINT_MATRICES
		cout << "\nA\n-----------------------\n"  << A << endl ;
		#endif /* PRINT_MATRICES */		

		// Nonzero entries:
		// (1,1) (2,2) (1,0) (1,2) (2,0) (1,1) (1,3) (1,2) 

		// Outer pointers:
		// 0 2 4 7  $

		// 0 1 2 0 
		// 1 0 1 0 
		// 2 1 0 1 
		// 0 0 1 0 

		

		MatrixXf network_embedding(num_nodes, dim);
		network_embedding = netmf(A, graph_size, rank, negative_sampling, dim, timer);

		#ifdef PRINT_MATRICES
		cout << "\nFinal Embeddings\n------------------------------\n" <<  network_embedding << endl ;
		#endif /* PRINT_MATRICES */	
		
		#ifdef PRINT_EXECUTION_TIME	
		print_timer("Embedding Ready" , timer);
		#endif /* PRINT_EXECUTION_TIME */

		fs::path embedding_file_path =  dataset_folder / dataset_name.replace_extension(".embedding") ;
		std::ofstream file(embedding_file_path.string());

		if (file.is_open())
		{
			file << network_embedding ;
		} else {

			throw "Opening file (" + embedding_file_path.string() + ") failed. Could not write embedding file.";

		}

		#ifdef PRINT_EXECUTION_TIME	
		print_timer("Embedding Written to file", timer);
		#endif /* PRINT_EXECUTION_TIME */

		// // small dim 2 

		// deepwalk_embedding
		// array([[-0.08001216,  0.1137426 ],
		//        [ 0.01545656,  0.090421  ],
		//        [ 0.211132  ,  0.20745989],
		//        [-0.19778848,  0.18250924]])

		// Final Embeddings are :
		//   0.113743 -0.0800115
		//    0.09042   0.015457
		//   0.207459   0.211133
		//    0.18251  -0.197788

		// // small dim 3 --signs changes , not fixed yet

		// deepwalk_embedding
		// array([[-0.15033474, -0.08001216,  0.1137426 ],
		//        [ 0.16173531,  0.01545656,  0.090421  ],
		//        [-0.02717237,  0.211132  ,  0.20745989],
		//        [ 0.04444911, -0.19778848,  0.18250924]])

		// Final Embeddings are :
		//   0.113743 -0.0800115   0.150335
		//    0.09042   0.015457  -0.161734
		//   0.207459   0.211133  0.0271718
		//    0.18251  -0.197788 -0.0444497


	 //   // large dim 2
		//    deepwalk_embedding
		//    array([[ 8.36876054e-02, -1.32933990e-17],
		//           [ 1.28825221e-01, -2.10478817e-17],
		//           [ 1.70554082e-17,  4.73809489e-02],
		//           [ 0.00000000e+00,  3.15761747e-01]])

		//    Final Embeddings are :
		//            0 0.0836838
		//            0   0.12882
		//    0.0473787         0
		//     0.315761         0



	 //   // large dim 3	
		// deepwalk_embedding
		// array([[ 1.61144400e-13,  8.36876054e-02,  2.29865024e-17],
		//        [-1.04680599e-13,  1.28825221e-01, -1.91092610e-17],
		//        [-8.76271124e-03,  3.41108164e-17,  4.73809489e-02],
		//        [ 1.31486977e-03,  3.41108164e-17,  3.15761747e-01]])



		// Final Embeddings are :
		//           0   0.0836838          -0
		//           0     0.12882          -0
		//   0.0473787           0 -0.00880246
		//    0.315761           0  0.00132077

	}
	catch (const program_options::error &ex)
	{
		std::cerr << ex.what() << '\n';
	}


}
