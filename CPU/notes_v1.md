
## ilp64 and lp64
	The Intel MKL ILP64 libraries use the 64-bit integer type (necessary for indexing large arrays, with more than 231-1 elements), whereas the LP64 libraries index arrays with the 32-bit integer type.

	Edge list format
		http://rosalind.info/glossary/algo-edge-list-format/

## Both of following is different command:

	Works
	g++ sparse_spmm.c -DMKL_ILP64 -I${MKLROOT}/include  -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

	Does not work
	g++ -DMKL_ILP64 -I${MKLROOT}/include  -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl sparse_spmm.c


## What is alignment in malloc
	https://stackoverflow.com/questions/3994035/what-is-aligned-memory-allocation

## learn about CSC and CSR handles
	http://amath.colorado.edu/sites/default/files/2015/01/195762631/SparseDataStructs.pdf


## When I create a matrix handle using routine mkl_sparse_?_create_csr, the handle will copy all data to new memory blocks? Or it only save the pointers in handle?  

	It save pointer in handle only.


## How pointerB and pointerE are saved:
	
	http://amath.colorado.edu/sites/default/files/2015/01/195762631/SparseDataStructs.pdf
    
    Both of them are in one single array. 
    pointerB is like col_ptr or row_ptr in CSC or CSR format, but have one extra item. This extra item makes pointerE by pointing it as (pointerB + 1)


## Popular C++ libraries

	https://en.cppreference.com/w/cpp/links/libs
	https://stackoverflow.com/questions/777764/what-modern-c-libraries-should-be-in-my-toolbox
	https://github.com/fffaraz/awesome-cpp

	Quora says : Boost and STL are must learn


# properties of normalized adjacency matrix
	
	https://people.orie.cornell.edu/dpw/orie6334/lecture7.pdf
	it's eigne values lies between 1 to -1


# eigen decomposition solution
		https://code.google.com/archive/p/redsvd/wikis/English.wiki

	Mixed programming intel mkl : 
		https://software.intel.com/en-us/articles/introduction-to-the-intel-mkl-extended-eigensolver
		https://software.intel.com/en-us/articles/solve-svd-problem-for-sparse-matrix-with-intel-math-kernel-library

	Spectra 
		https://spectralib.org/doc/index.html : Sparse Eigenvalue Computation Toolkit as a Redesigned ARPACK
		https://github.com/yixuan/spectra

# Eigen 3.5

	Symmetric eigen solver http://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html

	Self-adj is same as symmetric for real data values

# Inline Functions C++
	
	https://www.geeksforgeeks.org/inline-functions-cpp/

# Classes and objects C++ 
	https://www.geeksforgeeks.org/c-classes-and-objects/

# :: operator C++
	Defining a class function outside 	https://www.geeksforgeeks.org/c-classes-and-objects/

# Templates C++ 
	https://www.geeksforgeeks.org/templates-cpp/

# Eigen SVD thin vs full U
	https://stats.stackexchange.com/questions/50663/what-is-a-thin-svd
