
# Compiler Optimization

    https://stackoverflow.com/questions/28866601/optimize-large-matrices-multiplication-in-eigen

    -O3 -DEIGEN_NO_DEBUG -fopenmp
        O3

            -O1
            Optimize. Optimizing compilation takes somewhat more time, and a lot more memory for a large function.

            With -O, the compiler tries to reduce code size and execution time, without performing any optimizations that take a great deal of compilation time.

            -O turns on the following optimization flags:

            -fauto-inc-dec 
            -fbranch-count-reg 
            -fcombine-stack-adjustments 
            -fcompare-elim 
            -fcprop-registers 
            -fdce 
            -fdefer-pop 
            -fdelayed-branch 
            -fdse 
            -fforward-propagate 
            -fguess-branch-probability 
            -fif-conversion2 
            -fif-conversion 
            -finline-functions-called-once 
            -fipa-pure-const 
            -fipa-profile 
            -fipa-reference 
            -fmerge-constants 
            -fmove-loop-invariants 
            -fomit-frame-pointer 
            -freorder-blocks 
            -fshrink-wrap 
            -fshrink-wrap-separate 
            -fsplit-wide-types 
            -fssa-backprop 
            -fssa-phiopt 
            -ftree-bit-ccp 
            -ftree-ccp 
            -ftree-ch 
            -ftree-coalesce-vars 
            -ftree-copy-prop 
            -ftree-dce 
            -ftree-dominator-opts 
            -ftree-dse 
            -ftree-forwprop 
            -ftree-fre 
            -ftree-phiprop 
            -ftree-scev-cprop 
            -ftree-sink 
            -ftree-slsr 
            -ftree-sra 
            -ftree-pta 
            -ftree-ter 
            -funit-at-a-time
            -O2
            Optimize even more. GCC performs nearly all supported optimizations that do not involve a space-speed tradeoff. As compared to -O, this option increases both compilation time and the performance of the generated code.

            -O2 turns on all optimization flags specified by -O. It also turns on the following optimization flags:

            -fthread-jumps 
            -falign-functions  -falign-jumps 
            -falign-loops  -falign-labels 
            -fcaller-saves 
            -fcrossjumping 
            -fcse-follow-jumps  -fcse-skip-blocks 
            -fdelete-null-pointer-checks 
            -fdevirtualize -fdevirtualize-speculatively 
            -fexpensive-optimizations 
            -fgcse  -fgcse-lm  
            -fhoist-adjacent-loads 
            -finline-small-functions 
            -findirect-inlining 
            -fipa-cp 
            -fipa-bit-cp 
            -fipa-vrp 
            -fipa-sra 
            -fipa-icf 
            -fisolate-erroneous-paths-dereference 
            -flra-remat 
            -foptimize-sibling-calls 
            -foptimize-strlen 
            -fpartial-inlining 
            -fpeephole2 
            -freorder-blocks-algorithm=stc 
            -freorder-blocks-and-partition -freorder-functions 
            -frerun-cse-after-loop  
            -fsched-interblock  -fsched-spec 
            -fschedule-insns  -fschedule-insns2 
            -fstore-merging 
            -fstrict-aliasing 
            -ftree-builtin-call-dce 
            -ftree-switch-conversion -ftree-tail-merge 
            -fcode-hoisting 
            -ftree-pre 
            -ftree-vrp 
            -fipa-ra
            Please note the warning under -fgcse about invoking -O2 on programs that use computed gotos.

            -O3
            Optimize yet more. -O3 turns on all optimizations specified by -O2 and also turns on the following optimization flags:

            -finline-functions 
            -funswitch-loops 
            -fpredictive-commoning 
            -fgcse-after-reload 
            -ftree-loop-vectorize 
            -ftree-loop-distribution 
            -ftree-loop-distribute-patterns 
            -floop-interchange 
            -floop-unroll-and-jam 
            -fsplit-paths 
            -ftree-slp-vectorize 
            -fvect-cost-model 
            -ftree-partial-pre 
            -fpeel-loops 
            -fipa-cp-clone

        second disables eigen debugging

# control threads in eigen 

    OMP_NUM_THREADS=16 ./netmf_eigen

# Eigen general optimization

    http://eigen.tuxfamily.org/index.php?title=FAQ#Optimization

    intel optimization
       use icc (intels) compilercheck http://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html 

# enable vectorization

    http://eigen.tuxfamily.org/index.php?title=FAQ#How_can_I_enable_vectorization.3F

# optimizing expressions
    
    http://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html

# take care of which Gcc version G++ uses
     
     use module avail

     # blog and flickr job
     Submitted batch job 149861

     2 hr Submitted batch job 149903

     2 hr flickr Submitted batch job 149911

# ri2 build command

    g++ \
    -std=c++11 -march=native \
    # \
    netmf_eigen.cpp -o netmf_eigen
    # \
    -I ~/boost_1_68_0 \
    -I ~/eigen335 \
    -L ~/boost_1_68_0/lib \
    -L ~/expat-2.0.1/lib \
    -L ~/boost_1_68_0/stage/lib \
    # \
    -O3 \
    -DEIGEN_NO_DEBUG \
    -fopenmp    \
    -lboost_graph \
    -lboost_regex \
    -lexpat \
    -lboost_timer \
    -lboost_program_options \
    -lboost_program_options \
    -lboost_filesystem \
    -lboost_system \
    # \
    -Wl, -rpath=~/boost_1_68_0/lib  \
    -Wl, -rpath=~/expat-2.0.1/lib  \
    -Wl, -rpath=~/boost_1_68_0/stage/lib


    g++ -std=c++11 -march=native netmf_eigen.cpp -o netmf_eigen -I ~/boost_1_68_0 -I ~/eigen335 -L ~/boost_1_68_0/lib -L ~/expat-2.0.1/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_system -lboost_filesystem -Wl,-rpath=/home/jangid.6/boost_1_68_0/lib -Wl,-rpath=/home/jangid.6/expat-2.0.1/lib

    g++ -std=c++11 -march=native netmf_sparse.cpp -o netmf_sparse -I ~/boost_1_68_0 -I ~/eigen335 -I ~/spectra/include/Spectra -L ~/boost_1_68_0/lib -L ~/expat-2.0.1/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_system -lboost_filesystem -Wl,-rpath=/home/jangid.6/boost_1_68_0/lib -Wl,-rpath=/home/jangid.6/expat-2.0.1/lib

# my PC build command

    g++ -std=c++11 -march=native netmf_eigen.cpp -o netmf_eigen -I /home/mohit/Documents/boost_1_61_0 -I /home/mohit/Documents/eigen_cpp_installation -I /home/mohit/Documents/spectra_installation/include/Spectra -L /home/mohit/Documents/boost_1_61_0/lib -L /home/mohit/Documents/expat-2.0.1/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -Wl,-rpath=/home/mohit/Documents/boost_1_61_0/lib -Wl,-rpath=/home/mohit/Documents/expat-2.0.1/lib

# How command line work
    
    g++
                                           -std=c++11    :  for writing better format of c++ code
                                        -march=native    :  enable vectorization
                       netmf_eigen.cpp -o netmf_eigen    :  input output

                -I /home/mohit/Documents/boost_1_61_0    :  boost installation
      -I /home/mohit/Documents/eigen_cpp_installation    :  eigen installtion 

            -L /home/mohit/Documents/boost_1_61_0/lib    :  boost libraries
             -L /home/mohit/Documents/expat-2.0.1/lib    :  expact libraries for graphml XML parsing
                                                         :  eigen works without reference 

                                                  -O3    :  compiler optimiation  see above
                                     -DEIGEN_NO_DEBUG    :  eigen won't debug 

                                             -fopenmp    :  multipe threading
                                        -lboost_graph    :  boost graph io
                                        -lboost_regex    :  xml parse time regex 
                                              -lexpat    :  XML parsing 
                                        -lboost_timer    :  execution time measurement
                              -lboost_program_options    :  setting command line arguments
                                   -lboost_filesystem    :  joing dataset file and folder 
                                       -lboost_system    :  joing dataset file and folder 
    -Wl,-rpath=/home/mohit/Documents/boost_1_61_0/lib    :  load time library refrence for boost 
     -Wl,-rpath=/home/mohit/Documents/expat-2.0.1/lib    :  load time library refrence for eigen 
    ;

# time complexity of eigen package algorithm
    1. Self Adjoint eigen solver 
        http://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html

        --calls compute() function:
            http://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html#adf397f6bce9f93c4b0139a47e261fc24
            This implementation uses a symmetric QR algorithm. The matrix is first reduced to tridiagonal form using the Tridiagonalization class. The tridiagonal matrix is then brought to diagonal form with implicit symmetric QR steps with Wilkinson shift. Details can be found in Section 8.3 of Golub & Van Loan, Matrix Computations.
            The cost of the computation is about 9n^3 if the eigenvectors are required and 4n^3/3 if they are not required. 

    2. BDCSDV algorithm: 
        http://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html 

        This class first reduces the input matrix to bi-diagonal form using class UpperBidiagonalization, and then performs a divide-and-conquer diagonalization.


        // We used the "A Divide-And-Conquer Algorithm for the Bidiagonal SVD"
        // research report written by Ming Gu and Stanley C.Eisenstat
        https://epubs-siam-org.proxy.lib.ohio-state.edu/doi/pdf/10.1137/S0895479892242232

        BDC computes all the singular values in O(N^2) time and all the singular values and singular vectors in O(N^3) time.

# Symmetric Matrix
    All symmetric properties can be proved using following three properties
    1.  


# netmf Sparse code optimization
    
    1. insert into adjacency_list in two ways, see Filling a sparse matrix section https://eigen.tuxfamily.org/dox/group__TutorialSparse.html

    2. for each of above 
        insert only half trigular and create A by adding transpose   VS  add non zeros at once
    3. convert std math function sqrt max log to boost multiprecision
    4. Directly feed normalized adj matrix to eigen solver
    5. Mcap is still a dense matrix. Can be utilized for space efficiency until Mcap log max function

    6. combining m cap multiplication
    7. remove reverse operation and -ve sign in eigen solved values 
    8. Mcap log max uniary function can be done in place. Check if other fall in this category as well. 
    9. In RedSVD-h disable V calculation. 

#
jangid.6@head ~/work ❯❯❯ sbatch js_24_threads.sh
Submitted batch job 151662
jangid.6@head ~/work ❯❯❯ sbatch js_24_threads.sh
Submitted batch job 151663
jangid.6@head ~/work ❯❯❯ sbatch js_24_threads.sh
Submitted batch job 151664
jangid.6@head ~/work ❯❯❯ sbatch js_24_threads.sh
Submitted batch job 151665
jangid.6@head ~/work ❯❯❯ sbatch js_24_threads.sh
Submitted batch job 151666
jangid.6@head ~/work ❯❯❯ sbatch js_24_threads.sh
Submitted batch job 151667
jangid.6@head ~/work ❯❯❯ squeue -u jangid.6
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            151662     devel ppi_blog jangid.6  R       0:12      1 gpu01
            151663     devel ppi_blog jangid.6  R       0:09      1 storage03
            151664     devel ppi_blog jangid.6  R       0:09      1 storage04
            151665     devel ppi_blog jangid.6  R       0:06      1 storage11
            151666     devel ppi_blog jangid.6  R       0:06      1 storage12
            151667     devel ppi_blog jangid.6  R       0:03      1 storage13


jangid.6@head ~/work ❯❯❯ squeue -u jangid.6
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            151679     batch ppi_blog jangid.6  R       0:59      1 storage04
            151680     batch ppi_blog jangid.6  R       0:22      1 storage11
            151681     batch ppi_blog jangid.6  R       0:19      1 storage12
            151682     batch ppi_blog jangid.6  R       0:14      1 storage13
            151683     batch ppi_blog jangid.6  R       0:08      1 storage14
            151684     batch ppi_blog jangid.6  R       0:01      1 storage16


     