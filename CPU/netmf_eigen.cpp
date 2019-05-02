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
#include <math.h>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>


// #define PRINT_MATRICES
// #define PRINT_ALGO_PARAMS
#define PRINT_EXECUTION_TIME

namespace fs = boost::filesystem;
using namespace boost;
using namespace boost::timer;
using namespace Eigen;
using namespace std;


unsigned long long int window_size;
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
	std::cout << print_prefix << "," << message << "," << (wall - last_wall) << "," << (cpu - last_cpu) << "," << wall << "," <<  cpu  << endl ;

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
		return max((float)0, (x * (1 - powf(x, ::window_size))) / ((1 - x) * ::window_size ) );

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


MatrixXf netmf(MatrixXf &A, const string &graph_size, unsigned long long int rank, int negative_sampling, unsigned long long int dim, unsigned long long int num_nodes, cpu_timer &timer  )
{

	double vol = (double) A.sum();

	ArrayXf degree_vec(num_nodes, 1);

	// cout << "A : " << A << endl;

	// cout << "Vol of matrix A is " << vol << endl;

	// https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
	// std::cout << A.rowwise().sum() << std::endl;
	// 2
	// 2
	// 3
	// 1



	degree_vec << A.rowwise().sum();
	// degree_vec << A.rowwise().count().cast<float>();


	// https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html#a14235b62c90f93fe910070b4743782d0
	// D << MatrixXf(degree_vec.asDiagonal());
	// std::cout << "Diagonalization"  << std::endl << D << std::endl;
	// Diagonalization
	// 2 0 0 0
	// 0 2 0 0
	// 0 0 3 0
	// 0 0 0 1

	// https://eigen.tuxfamily.org/dox/classEigen_1_1ArrayBase.html#adbe7fe792be28d5293ae6a586b36c028
	// https://eigen.tuxfamily.org/dox/classEigen_1_1ArrayBase.html#a7da82f3eb206b756cf88248229699c6b

	// element wise vs whole opertion
	// https://stackoverflow.com/questions/44651097/error-with-eigen-vector-logarithm-invalid-use-of-incomplete-type
	// https://eigen.tuxfamily.org/dox/group__CoeffwiseMathFunctions.html

	// cout << "degree vec sqrt"  << endl << degree_vec.sqrt().inverse() << endl;

	
	#ifdef PRINT_MATRICES
	cout << "\nVol of matrix A is\n------------------------------\n" << vol << endl;
	#endif /* PRINT_MATRICES */

	// There are two ways to create Diagonal matrix

	// Type 1
	// the special Diagonal matrix class. stores only diagonals



	DiagonalMatrix<float, Dynamic> D_invsqrt(num_nodes);
	D_invsqrt.diagonal() << degree_vec.sqrt().inverse();

	// We can not print full matrix with DiagonaMatrix, only digonals can be printed
	// cout << endl << "D_invsqrt is " << D_invsqrt.diagonal().transpose() << endl;


	// Type 2

	// normal matrix , filled with diagonal elements
	// D = VectorXf(degree_vec.sqrt().inverse()).asDiagonal();
	// cout << D << endl;

	// not using this for now

	// normalized adjacency matrix
	MatrixXf norm_adj;
	norm_adj.noalias() = D_invsqrt * A * D_invsqrt;
	// cout << "normalized adjacency matrix " << endl << norm_adj  << endl;

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

	#ifdef PRINT_EXECUTION_TIME	
	print_timer("Normalized Adjacency Matrix", timer);
	#endif /* PRINT_EXECUTION_TIME */



	MatrixXf M_cap;

	// Approximate M cap

	if (graph_size == "large") {

		// http://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
		SelfAdjointEigenSolver<MatrixXf> es(norm_adj);

		// The eigenvalues are repeated according to their algebraic multiplicity, so there are as many eigenvalues as rows in the matrix. The eigenvalues are sorted in increasing order.
		// http://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html#a3df8721abcc71132f7f02bf9dfe78e41
		// cout << "The eigenvalues of adj_norm are:" << endl << es.eigenvalues() << endl;
		// cout << "The matrix of eigenvectors, adj_norm, is:" << endl << es.eigenvectors() << endl << endl;

	    #ifdef PRINT_EXECUTION_TIME
		print_timer("Eigen Decomposition" , timer);
    	#endif /* PRINT_EXECUTION_TIME */

		// reduction to top rank values
		// see quick reference http://eigen.tuxfamily.org/dox/group__QuickRefPage.html#QuickRef_DiagTriSymm


		// cout << "The top rank eigenvalues of adj_norm are:" << endl << es.eigenvalues().tail(rank) << endl;

    	#ifdef PRINT_MATRICES
    	cout << "\nEigen vals and vecs\n------------------------\n";
    	std::cout << "\neval\n" <<es.eigenvalues().reverse().transpose() << std::endl;
    	std::cout << "\nevec\n"<< ( es.eigenvectors().rowwise().reverse()) << std::endl;
    	// cout << "summation_approximate :" << endl <<  ( ArrayXf) eigs.eigenvalues().reverse().unaryExpr(ptr_fun(summation_approximate)) << endl;
		#endif /* PRINT_MATRICES */	

		// slicing top rank
		MatrixXf e_vec_M_h = es.eigenvectors().rightCols(rank) ;
		// Filtering eigen values as approximation done in python code
		ArrayXf e_val_h = es.eigenvalues().tail(rank).unaryExpr(ptr_fun(summation_approximate));

		// cout << "The top rank sum approximated eigenvalues of adj_norm are:" << endl << e_val_h  << endl;
		// cout << "The top rank matrix of eigenvectors, adj_norm, is:" << endl << e_vec_M_h << endl ;

		// diagonalized  eigen values
		DiagonalMatrix<float, Dynamic> ev_diag_M_h(rank);
		ev_diag_M_h.diagonal() << e_val_h.sqrt();


		MatrixXf temp_m;
		temp_m.noalias() = (ev_diag_M_h * (D_invsqrt * e_vec_M_h).transpose()).transpose();

		// cout << "temp_m is " << endl << temp_m << endl ;


		M_cap.noalias() = (float)(vol / (double) negative_sampling) * (temp_m * temp_m.transpose());




	}
	else
		// graph size is small
	{
		// to be summed
		MatrixXf S = MatrixXf::Zero(num_nodes, num_nodes);
		MatrixXf X = MatrixXf::Identity(num_nodes, num_nodes);

		for (int i = 0; i < ::window_size; ++i)
		{
			X *= norm_adj;
			S += X;

			// cout << endl << "S " << i << " : " << endl << S <<  endl;
		}

		// cout << "Summed norm_adj " << endl << S << endl ;
		#ifdef PRINT_MATRICES
		cout << "\nSummed norm_adj\n---------------------------------\n" << S << endl ;
		#endif /* PRINT_MATRICES */	

		#ifdef PRINT_EXECUTION_TIME	
		print_timer("Summed Norm_adj" , timer);
		#endif /* PRINT_EXECUTION_TIME */


		M_cap.noalias() = (float)( vol / ((double)(::window_size * negative_sampling)) ) * (D_invsqrt * (D_invsqrt * S).transpose());
		// cout << "approximated M is " << endl << M_cap << endl ;
		
		
	}
	// element wise log and maximum (each element, 1)

	M_cap = M_cap.unaryExpr(ptr_fun(log_max_each_vs_1));

	
	#ifdef PRINT_MATRICES
	cout << "\nMcap \n------------------------\n"  << M_cap << endl ;
	#endif /* PRINT_MATRICES */	

	#ifdef PRINT_EXECUTION_TIME	
	print_timer("Approximated M" , timer);
	#endif /* PRINT_EXECUTION_TIME */

	// SVD calculation
	//https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html
	//https://stats.stackexchange.com/questions/50663/what-is-a-thin-svd

	// 

	BDCSVD<MatrixXf> svd(M_cap, ComputeFullU);

	// the results are stored in particular members as shown https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html

	// cout << "Singular Values are " << endl << svd.singularValues() << endl ;
	// cout << "Computed U  is " << endl << svd.matrixU() << endl ;
	
	#ifdef PRINT_MATRICES
	// the results are stored in particular members as shown https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
	cout << "\nSingular Values and U are\n------------------------\n";
	cout << "\nSingular Values are " << endl << svd.singularValues().transpose() * perm << endl ;
	cout << "\nComputed U  is " << endl << svd.matrixU() * perm << endl ;
	#endif /* PRINT_MATRICES */	

	#ifdef PRINT_EXECUTION_TIME	
	print_timer("SVD" , timer);
	#endif /* PRINT_EXECUTION_TIME */


	MatrixXf U_d = svd.matrixU().leftCols(dim);
	DiagonalMatrix<float, Dynamic> S_d(dim);
	S_d.diagonal() = svd.singularValues().head(dim).array().sqrt();

	return U_d * S_d;

}



int main(int argc, char *argv[])
{


	try
	{

		fs::path dataset_name;
		fs::path dataset_folder;
		unsigned long long int rank;
		unsigned long long int dim;
		float negative_sampling;
		std::string graph_size;
		std::string embedding_file_name_suffix;


		program_options::options_description desc{"Options"};
		desc.add_options()
		("help,h", "Help screen")
		// requried options
		("dataset,d"    , program_options::value<fs::path>(&dataset_name)->required() , "dataset name e.g. PPI")
		("dataset_folder,f"        , program_options::value<fs::path>(&dataset_folder)->default_value("/home/jangid.6/work/datasets") , "path to dataset folder")
		// default options
		("rank,r"              , program_options::value<unsigned long long int>(&rank)->default_value(256) , "#eigenpairs used to approximate normalized graph laplacian.")
		("dim,d"               , program_options::value<unsigned long long int>(&dim)->default_value(128) , "dimension of embedding")
		("window,w"            , program_options::value<unsigned long long int>(&::window_size)->default_value(10) , "context window size")
		("negative_sampling,n" , program_options::value<float>(&negative_sampling)->default_value(1.0) , "negative sampling")
		("graph_size,s"        , program_options::value<std::string>(&graph_size)->default_value("large") , "netmf large or small algorithm (e.g. small/large )")
		("embedding_file_name_suffix,es"        , program_options::value<std::string>(&embedding_file_name_suffix)->default_value("") , "Optional suffix to add in embedding_file_name for miscellaneous purpose");
        

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
		std::cout << "window_size : " << ::window_size << std::endl;
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
		// cout << timer.format() << '\n';

		unsigned long long int num_nodes = boost::num_vertices(g);

		// cout << "Dataset : " << dataset_file << endl;
		// cout << "Num_nodes : " << num_nodes << endl;
		// cout << "Num Threads :" << Eigen::nbThreads( ) << endl; 


		print_prefix = boost::regex_replace(dataset_name.string(), boost::regex(".graphml"), "") + "," + graph_size + "," + to_string(Eigen::nbThreads( )) + "," + to_string(::window_size)  + "," + to_string(dim) + "," + to_string(embedding_file_name_suffix) ;

		#ifdef PRINT_EXECUTION_TIME		
		print_timer("Graph Loaded from file", timer);
        #endif /* PRINT_EXECUTION_TIME */

		MatrixXf A =  MatrixXf::Zero(num_nodes, num_nodes);
		graph_traits<Graph>::edge_iterator ei, ei_end;

		// get the property map for vertex indices
		typedef property_map<Graph, vertex_index_t>::type IndexMap;
		IndexMap index = get(vertex_index, g);

		for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
		{

			A(index[source(*ei, g)], index[target(*ei, g)]) = g[*ei].weight;
			A( index[target(*ei, g)], index[source(*ei, g)]) = g[*ei].weight;

		}


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

		

		MatrixXf network_embedding(num_nodes, dim+1);
		network_embedding << VectorXf::LinSpaced(num_nodes,0,num_nodes-1), netmf(A, graph_size, rank, negative_sampling, dim, num_nodes, timer);



		#ifdef PRINT_MATRICES
		cout << "\nFinal Embeddings\n------------------------------\n" <<  network_embedding.rightCols(dim) << endl ;
		#endif /* PRINT_MATRICES */	
		
		#ifdef PRINT_EXECUTION_TIME	
		print_timer("Embedding Ready" , timer);
		#endif /* PRINT_EXECUTION_TIME */

		if(embedding_file_name_suffix != "") // use has provided a value
			embedding_file_name_suffix = "_" + embedding_file_name_suffix;

			
		string embedding_file_name = boost::regex_replace(dataset_name.string(), boost::regex(".graphml"), "") + "_dense_eigen_algo_" + graph_size + "_win" + to_string(::window_size) + "_emdDim" + to_string(dim) + "_numThreads" + to_string(Eigen::nbThreads( )) +  embedding_file_name_suffix + ".embedding"; 


		fs::path embedding_file_path =  dataset_folder / fs::path(embedding_file_name) ;
		std::ofstream file(embedding_file_path.string());

		if (file.is_open())
		{
			
			
			IOFormat space_separated(FullPrecision, DontAlignCols, " ", "\n", "", "", "", "");
			file << num_nodes << " " << dim << "\n";
			file << network_embedding.format(space_separated) ;
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
