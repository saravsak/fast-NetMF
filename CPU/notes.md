
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

    g++ -std=c++11 -march=native netmf_sparse.cpp -o netmf_sparse -I /home/mohit/Documents/boost_1_61_0 -I /home/mohit/Documents/eigen_cpp_installation -I /home/mohit/Documents/spectra_installation/include/Spectra -L /home/mohit/Documents/boost_1_61_0/lib -L /home/mohit/Documents/expat-2.0.1/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -Wl,-rpath=/home/mohit/Documents/boost_1_61_0/lib -Wl,-rpath=/home/mohit/Documents/expat-2.0.1/lib
    
    g++ -std=gnu++11 arpack_ng_sample.cpp -o arpack_ng_sample -I /home/mohit/Documents/arpack-ng/install/include/arpack -L /home/mohit/Documents/arpack-ng/install/lib -L /home/mohit/Documents/arpack-ng/SRC/.libs =L /home/mohit/Documents/arpack-ng/UTIL/.libs -Wl,-rpath=/home/mohit/Documents/arpack-ng/install/lib -Wl,-rpath=/home/mohit/Documents/arpack-ng/SRC/.libs -Wl,-rpath=/home/mohit/Documents/arpack-ng/UTIL/.libs 


    --- add lapck and openblas links 

    export LD_LIBRARY_PATH=/usr/local/lib:/home/mohit/Documents/boost_1_69_0/link/lib
    
    g++ -g -std=c++11 -march=native netmf_sparse.cpp -o netmf_sparse -I /home/mohit/Documents/boost_1_69_0 -I /home/mohit/Documents/eigen_3 -I /home/mohit/Documents/spectra-0.7.0/include/Spectra -L /home/mohit/Documents/boost_1_69_0/link/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -Wl,-rpath=/home/mohit/Documents/boost_1_69_0/link/lib



    arpack c++

    g++ -std=gnu++11 arpack_ng_sample.cpp -o arpack_ng_sample -I/home/mohit/Documents/arpack-ng/link/include/arpack -L/home/mohit/Documents/arpack-ng/link/lib -I/home/mohit/Documents/openmpi-4.0.0/link/include -I/home/mohit/Documents/OpenBLAS/link/include -L/home/mohit/Documents/openmpi-4.0.0/link/lib -L/home/mohit/Documents/OpenBLAS/link/lib -L/home/mohit/Documents/lapack/link -larpack -llapack -lblas -Wl,-rpath=/home/mohit/Documents/openmpi-4.0.0/link/lib -Wl,-rpath=/home/mohit/Documents/OpenBLAS/link/lib -Wl,-rpath=/home/mohit/Documents/lapack/link -Wl,-rpath=/home/mohit/Documents/arpack-ng/link/lib



    ------------ Updated ----------------------

    Dense:
        
        g++ -std=c++11 -march=native netmf_eigen.cpp -o netmf_eigen -I /home/mohit/Documents/boost_1_69_0 -I /home/mohit/Documents/eigen_3 -L /home/mohit/Documents/boost_1_69_0/link/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -Wl,-rpath=/home/mohit/Documents/boost_1_69_0/link/lib



    Sparse - Spectra :
        g++ -std=c++11 -march=native netmf_sparse.cpp -o netmf_sparse -I /home/mohit/Documents/boost_1_69_0 -I /home/mohit/Documents/eigen_3 -I /home/mohit/Documents/spectra-0.7.0/include/Spectra -L /home/mohit/Documents/boost_1_69_0/link/lib -O3 -DEIGEN_NO_DEBUG -fopenmp  -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -Wl,-rpath=/home/mohit/Documents/boost_1_69_0/link/lib
    

    Sparse - Arpack :

        export LD_LIBRARY_PATH=$HOME/Documents/openmpi-4.0.0/link/lib:$HOME/Documents/OpenBLAS/link/lib:$HOME/Documents/lapack/link:$HOME/Documents/superlu/link/lib:$HOME/Documents/arpack-ng/link/lib:$HOME/Documents/boost_1_69_0/link/lib:$HOME/Documents/armadillo-9.300.2/link/lib


        g++ -std=c++11 netmf_sparse_arma.cpp -o netmf_sparse_arma -I$HOME/Documents/boost_1_69_0 -I$HOME/Documents/eigen_3 -I$HOME/Documents/spectra-0.7.0/include/Spectra -I$HOME/Documents/openmpi-4.0.0/link/include -I$HOME/Documents/OpenBLAS/link/include -I$HOME/Documents/superlu/link/include -I$HOME/Documents/lapack/link/include -I$HOME/Documents/arpack-ng/link/include -I$HOME/Documents/armadillo-9.300.2/link/include -L$HOME/Documents/openmpi-4.0.0/link/lib -L$HOME/Documents/boost_1_69_0/link/lib -L$HOME/Documents/OpenBLAS/link/lib -L$HOME/Documents/lapack/link -L$HOME/Documents/superlu/link/lib -L$HOME/Documents/arpack-ng/link/lib -L$HOME/Documents/armadillo-9.300.2/link/lib -O3 -DEIGEN_NO_DEBUG -fopenmp -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -larmadillo -lopenblas -larpack -lsuperlu -lboost_chrono -Wl,-rpath=$HOME/Documents/openmpi-4.0.0/link/lib -Wl,-rpath=$HOME/Documents/OpenBLAS/link/lib -Wl,-rpath=$HOME/Documents/lapack/link -Wl,-rpath=$HOME/Documents/superlu/link/lib -Wl,-rpath=$HOME/Documents/arpack-ng/link/lib -Wl,-rpath=$HOME/Documents/armadillo-9.300.2/link/lib -Wl,-rpath=$HOME/Documents/boost_1_69_0/link/lib


        *** DO NOT USE $HOME in above command ***

    Sparse - Arpack -- Ri2 -- version;

     
        export LD_LIBRARY_PATH=$HOME/openmpi-4.0.0/link/lib:$HOME/OpenBLAS/link/lib:$HOME/lapack/link:$HOME/superlu/link/lib:$HOME/arpack-ng/link/lib:$HOME/boost_1_70_0/link/lib:$HOME/armadillo-9.300.2/link/lib


        g++ -std=c++11 -march=native netmf_sparse_arma.cpp  -o netmf_sparse_arma -I$HOME/boost_1_70_0 -I$HOME/boost_1_70_0/link/include -I$HOME/eigen335 -I$HOME/spectra/include/Spectra -I$HOME/openmpi-4.0.0/link/include -I$HOME/OpenBLAS/link/include -I$HOME/superlu/link/include -I$HOME/lapack/link/include -I$HOME/arpack-ng/link/include -L$HOME/openmpi-4.0.0/link/lib -L$HOME/boost_1_70_0/link/lib -L$HOME/OpenBLAS/link/lib -L$HOME/lapack/link -L$HOME/superlu/link/lib -L$HOME/arpack-ng/link/lib -I$HOME/armadillo-9.300.2/link/include/ -L$HOME/armadillo-9.300.2/link/lib -O3 -DEIGEN_NO_DEBUG -fopenmp -lboost_graph -lboost_regex -lexpat -lboost_timer -lboost_program_options -lboost_filesystem -lboost_system -larmadillo -lopenblas -larpack -lsuperlu -Wl,-rpath=$HOME/openmpi-4.0.0/link/lib -Wl,-rpath=$HOME/OpenBLAS/link/lib -Wl,-rpath=$HOME/lapack/link -Wl,-rpath=$HOME/superlu/link/lib -Wl,-rpath=$HOME/arpack-ng/link/lib -Wl,-rpath=$HOME/armadillo-9.300.2/link/lib -Wl,-rpath=$HOME/boost_1_70_0/link/lib

# check Embedding quality 
    
    My PC:

        from CPU folder 

        export DS=blog

        python ./datasets/predict.py --label ./datasets/$DS/$DS.mat --embedding ./datasets/$DS/embe --matfile-variable-name group --seed 10 --start-train-ratio 10  --stop-train-ratio 90  --num-train-ratio 3  --num-split 5


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
    10. re compile arpack with optimization enabled mpi..  

# Wall vs System vs Usr time 

    https://stackoverflow.com/questions/7335920/what-specifically-are-wall-clock-time-user-cpu-time-and-system-cpu-time-in-uni

    https://stackoverflow.com/questions/26398106/how-to-interpret-the-output-of-boosttimercpu-timer-on-multicore-machine

    Wall clock time: time elapsed according to the computer's internal clock, which should match time in the outside world. This has nothing to do with CPU usage; it's given for reference.

    User CPU time and system time: exactly what you think. System calls, which include I/O calls such as read, write, etc. are executed by jumping into kernel code and executing that.

    If wall clock time < CPU time, then you're executing a program in parallel.
    if two core runs for 5 seconds each then CPU time will be 10 sec. 

    If wall clock time > CPU time, you're waiting for disk, network or other devices.

    https://unix.stackexchange.com/questions/40694/why-real-time-can-be-lower-than-user-time
    The rule of thumb is:

        real < user: The process is CPU bound and takes advantage of parallel execution on multiple cores/CPUs.
        real ≈ user: The process is CPU bound and takes no advantage of parallel exeuction.
        real > user: The process is I/O bound. Execution on multiple cores would be of little to no advantage.

# Installing LAPAC and openblas
    https://stackoverflow.com/questions/36676449/lapack-blas-openblas-proper-installation-from-source-replace-system-libraries

    lapack http://www.netlib.org/lapack/
        > 
        > OepnBlas is already included in it version 
        > 
        > install gfortran (sudo apt-get install gfortran)
        > 
        > clone git repo https://github.com/Reference-LAPACK/lapack-release
        > 
        > type make -- this should include testing as well
        > 
        > fixed testing error by https://unix.stackexchange.com/questions/428394/lapack-make-fails-recipe-for-target-znep-out-failed-error 
        > 
        > C Wrapper documentation http://www.netlib.org/lapack/lapacke.html 
        > 
        > function syntax documentation http://www.netlib.org/lapack/lug/node19.html 
        > 
        >  
    


    openblas https://github.com/xianyi/OpenBLAS
        make
        make PREFIX=your_installation_directory install 


# Take care of these things 
    
    In graphml file the order of nodes should be from 0 to max (in sorted order). Otherwise the embedding quality will fall.

    /home/mohit/Dropbox/spring_2019_MILE_project/netbeans/fast-NetMF/CPU/datasets/edgelist_to_mat_and_graphml.py file will keep the order of graphml nodes in sorted order only when compiling with python2.7

# first success compilation of ARPACK-NG
    1. Install OpenBLAS https://github.com/xianyi/OpenBLAS
        make
        make PREFIX=/home/mohit/Documents/OpenBLAS/link install
    
    2. Install openmpi https://www.open-mpi.org/software/ompi/v4.0/

        ./configure --prefix=/home/mohit/Documents/openmpi-4.0.1/link
        make
        make PREFIX=$HOME/openmpi-4.0.1/link install

        . add openmpi bin to PATH 
        ~ ❯❯❯ echo $PATH
        /usr/local/bin:/usr/local/sbin:/home/mohit/bin:/home/mohit/.local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:$HOME/Documents/openmpi-4.0.0/link/bin

        
        export PATH=$PATH:$HOME/


    3. Install lapack https://github.com/Reference-LAPACK/lapack 
        change cmake to 
            
            mkdir build
            cd build
            cmake -DCMAKE_INSTALL_LIBDIR=/home/mohit/Documents/lapack/link -DCMAKE_INSTALL_PREFIX=/home/mohit/Documents/lapack/link -DBUILD_SHARED_LIBS=ON -DLAPACKE=ON .. 
            cmake --build . --target install


            for -- ri2
            
            mkdir build
            cd build
            cmake -DCMAKE_INSTALL_LIBDIR=$HOME/lapack/link -DCMAKE_INSTALL_PREFIX=$HOME/lapack/link -DBUILD_SHARED_LIBS=ON -DLAPACKE=ON .. 
            cmake --build . --target install
    
    4. Arpack-ng

        https://github.com/opencollab/arpack-ng
        
        -- terminal.log will show following data last history
        1. sh bootstrap

        2. Setup enviroment variables 

        export LDFLAGS="-I/home/mohit/Documents/openmpi-4.0.0/link/include -I/home/mohit/Documents/OpenBLAS/link/include -L/home/mohit/Documents/openmpi-4.0.0/link/lib -L/home/mohit/Documents/OpenBLAS/link/lib -L/home/mohit/Documents/lapack/link -Wl,-rpath=/home/mohit/Documents/openmpi-4.0.0/link/lib -Wl,-rpath=/home/mohit/Documents/OpenBLAS/link/lib -Wl,-rpath=/home/mohit/Documents/lapack/link"


        export LOADD="-Wl,-rpath=/home/mohit/Documents/openmpi-4.0.0/link/lib -Wl,-rpath=/home/mohit/Documents/OpenBLAS/link/lib -Wl,-rpath=/home/mohit/Documents/lapack/link"

        export LIBS="-llapack -lblas"

        export LT_SYS_LIBRARY_PATH="/home/mohit/Documents/openmpi-4.0.0/link/lib /home/mohit/Documents/OpenBLAS/link/lib /home/mohit/Documents/lapack/link"

        export PATH="$PATH:/home/mohit/Documents/openmpi-4.0.0/link/bin"

        ./configure --enable-icb --enable-mpi  --enable-shared --with-blas=/home/mohit/Documents/OpenBLAS/link/lib/libopenblas.so -with-lapack=/home/mohit/Documents/lapack/link/liblapack.so --prefix=/home/mohit/Documents/arpack-ng/link



        For ri2 --


        export LDFLAGS="-I$HOME/openmpi-4.0.0/link/include -I$HOME/OpenBLAS/link/include -L$HOME/openmpi-4.0.0/link/lib -L$HOME/OpenBLAS/link/lib -L$HOME/lapack/link -Wl,-rpath=$HOME/openmpi-4.0.0/link/lib -Wl,-rpath=$HOME/OpenBLAS/link/lib -Wl,-rpath=$HOME/lapack/link"


        export LOADD="-Wl,-rpath=$HOME/openmpi-4.0.0/link/lib -Wl,-rpath=$HOME/OpenBLAS/link/lib -Wl,-rpath=$HOME/lapack/link"

        export LIBS="-llapack -lblas"

        export LT_SYS_LIBRARY_PATH="$HOME/openmpi-4.0.0/link/lib $HOME/OpenBLAS/link/lib $HOME/lapack/link"

        export PATH="$PATH:$HOME/openmpi-4.0.0/link/bin"

        ./configure --enable-icb --enable-mpi  --enable-shared --with-blas=$HOME/OpenBLAS/link/lib/libopenblas.so -with-lapack=$HOME/lapack/link/liblapack.so --prefix=$HOME/arpack-ng/link


        export LD_LIBRARY_PATH="$HOME/openmpi-4.0.0/link/lib:$HOME/OpenBLAS/link/lib:$HOME/lapack/link"


        3. make
        4. make check
        5. make install

    5. SuperLU (Required for Armagadiilo)

        -- 




        from superlu directory

        1. cp MAKE_INC/make.linux make.inc
        2. edit make.inc
            uncomment to following
            BLASDEF = -DUSE_VENDOR_BLAS
            BLASLIB = -L/home/mohit/Documents/OpenBLAS/link/lib -lblas

        3. mkdir build; cd build;
        4. cmake .. -DCMAKE_INSTALL_LIBDIR=/home/mohit/Documents/superlu/link/lib -DCMAKE_INSTALL_PREFIX=/home/mohit/Documents/superlu/link -DCMAKE_INSTALL_INCLUDEDIR=/home/mohit/Documents/superlu/link/include -DBUILD_SHARED_LIBS=ON


        for ri2 --
        module avail
        module load cmake/3.10.2

        cmake .. -DCMAKE_INSTALL_LIBDIR=$HOME/superlu/link/lib -DCMAKE_INSTALL_PREFIX=$HOME/superlu/link -DCMAKE_INSTALL_INCLUDEDIR=$HOME/superlu/link/include -DBUILD_SHARED_LIBS=ON
        
        5. make 
        6. make install
        7. make test

    7. Installing boost again on ri2


        export LD_LIBRARY_PATH=$HOME/openmpi-4.0.0/link/lib:$HOME/OpenBLAS/link/lib:$HOME/lapack/link:$HOME/superlu/link/lib:$HOME/arpack-ng/link/lib
        export PATH="$PATH:$HOME/openmpi-4.0.0/link/bin"

        ./bootstrap.sh --prefix=$HOME/boost_1_70_0/link
        ./b2 install --prefix=$HOME/boost_1_70_0/link

    6. Installing armadillio

        1. 

        export LDFLAGS="-I/home/mohit/Documents/openmpi-4.0.0/link/include -I/home/mohit/Documents/OpenBLAS/link/include -I/home/mohit/Documents/superlu/link/include -I/home/mohit/Documents/lapack/link/include -I/home/mohit/Documents/arpack-ng/link/include -L/home/mohit/Documents/openmpi-4.0.0/link/lib -L/home/mohit/Documents/OpenBLAS/link/lib -L/home/mohit/Documents/lapack/link -L/home/mohit/Documents/superlu/link/lib -L/home/mohit/Documents/arpack-ng/link/lib -Wl,-rpath=/home/mohit/Documents/openmpi-4.0.0/link/lib -Wl,-rpath=/home/mohit/Documents/OpenBLAS/link/lib -Wl,-rpath=/home/mohit/Documents/lapack/link -Wl,-rpath=/home/mohit/Documents/superlu/link/lib -Wl,-rpath=/home/mohit/Documents/arpack-ng/link/lib"


        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/mohit/Documents/openmpi-4.0.0/link/lib:/home/mohit/Documents/OpenBLAS/link/lib:/home/mohit/Documents/lapack/link:/home/mohit/Documents/superlu/link/lib:/home/mohit/Documents/arpack-ng/link/lib


        rm -f CMakeCache.txt;  cmake -DLAPACK_LIBRARY=/home/mohit/Documents/lapack/link/liblapack.so -DBLAS_LIBRARY=/home/mohit/Documents/lapack/link/libblas.so  -DOpenBLAS_LIBRARY=/home/mohit/Documents/OpenBLAS/link/lib/libopenblas.so -DARPACK_LIBRARY=/home/mohit/Documents/arpack-ng/link/lib/libarpack.so -DSuperLU_LIBRARY=/home/mohit/Documents/superlu/link/lib/libsuperlu.so -DOpenBLAS_INCLUDE_DIR=/home/mohit/Documents/OpenBLAS/link/include -DLAPACK_INCLUDE_DIR=/home/mohit/Documents/lapack/link/include -DARPACK_INCLUDE_DIR=/home/mohit/Documents/arpack-ng/link/include -DSuperLU_INCLUDE_DIR=/home/mohit/Documents/superlu/link/include -DCMAKE_INSTALL_LIBDIR=/home/mohit/Documents/armadillo-9.300.2/link/lib -DCMAKE_INSTALL_PREFIX=/home/mohit/Documents/armadillo-9.300.2/link -DCMAKE_INSTALL_INCLUDEDIR=/home/mohit/Documents/armadillo-9.300.2/link/include -DCMAKE_INSTALL_BINDIR=/home/mohit/Documents/armadillo-9.300.2/link/bin -DBUILD_SHARED_LIBS=ON .


        g++ -std=c++11 armadillo_first_simple.cpp -I/home/mohit/Documents/eigen_3 -I/home/mohit/Documents/openmpi-4.0.0/link/include -I/home/mohit/Documents/OpenBLAS/link/include -I/home/mohit/Documents/superlu/link/include -I/home/mohit/Documents/lapack/link/include -I/home/mohit/Documents/arpack-ng/link/include -L/home/mohit/Documents/openmpi-4.0.0/link/lib -L/home/mohit/Documents/OpenBLAS/link/lib -L/home/mohit/Documents/lapack/link -L/home/mohit/Documents/superlu/link/lib -L/home/mohit/Documents/arpack-ng/link/lib -I/home/mohit/Documents/armadillo-9.300.2/link/include/ -L/home/mohit/Documents/armadillo-9.300.2/link/lib -larmadillo -lopenblas -larpack -lsuperlu -Wl,-rpath=/home/mohit/Documents/openmpi-4.0.0/link/lib -Wl,-rpath=/home/mohit/Documents/OpenBLAS/link/lib -Wl,-rpath=/home/mohit/Documents/lapack/link -Wl,-rpath=/home/mohit/Documents/superlu/link/lib -Wl,-rpath=/home/mohit/Documents/arpack-ng/link/lib -Wl,-rpath=/home/mohit/Documents/armadillo-9.300.2/link/lib 


        For ri2:

        export LDFLAGS="-I$HOME/openmpi-4.0.0/link/include -I$HOME/OpenBLAS/link/include -I$HOME/superlu/link/include -I$HOME/lapack/link/include -I$HOME/arpack-ng/link/include -L$HOME/openmpi-4.0.0/link/lib -L$HOME/OpenBLAS/link/lib -L$HOME/lapack/link -L$HOME/superlu/link/lib -L$HOME/arpack-ng/link/lib -Wl,-rpath=$HOME/openmpi-4.0.0/link/lib -Wl,-rpath=$HOME/OpenBLAS/link/lib -Wl,-rpath=$HOME/lapack/link -Wl,-rpath=$HOME/superlu/link/lib -Wl,-rpath=$HOME/arpack-ng/link/lib"


        export LD_LIBRARY_PATH=$HOME/openmpi-4.0.0/link/lib:$HOME/OpenBLAS/link/lib:$HOME/lapack/link:$HOME/superlu/link/lib:$HOME/arpack-ng/link/lib

        rm -f CMakeCache.txt;  cmake -DLAPACK_LIBRARY=$HOME/lapack/link/liblapack.so -DBLAS_LIBRARY=$HOME/lapack/link/libblas.so  -Dopenblas_LIBRARY=$HOME/OpenBLAS/link/lib/libopenblas.so -DARPACK_LIBRARY=$HOME/arpack-ng/link/lib/libarpack.so -DSuperLU_LIBRARY=$HOME/superlu/link/lib/libsuperlu.so -DOpenBLAS_INCLUDE_DIR=$HOME/OpenBLAS/link/include -DLAPACK_INCLUDE_DIR=$HOME/lapack/link/include -DARPACK_INCLUDE_DIR=$HOME/arpack-ng/link/include -DSuperLU_INCLUDE_DIR=$HOME/superlu/link/include -DCMAKE_INSTALL_LIBDIR=$HOME/armadillo-9.300.2/link/lib -DCMAKE_INSTALL_PREFIX=$HOME/armadillo-9.300.2/link -DCMAKE_INSTALL_INCLUDEDIR=$HOME/armadillo-9.300.2/link/include -DCMAKE_INSTALL_BINDIR=$HOME/armadillo-9.300.2/link/bin -DBUILD_SHARED_LIBS=ON .






# Speeding up from Arpack-ng
        
    -- use parallel superlu version https://crd-legacy.lbl.gov/~xiaoye/SuperLU/
    -- user parallel arpack-ng version -lparpack
        

# sparse matrix eigen values using arpack
    Links
    -----

    https://scicomp.stackexchange.com/questions/26786/eigen-max-and-minimum-eigenvalues-of-a-sparse-matrix


# armagallio eigen decomposition
    sparse matrix creation http://arma.sourceforge.net/docs.html#SpMat

    dense matrix initilialization http://arma.sourceforge.net/docs.html#element_initialisation




    todos:
       > check floating point precision..only 4 decimals are being printed http://arma.sourceforge.net/docs.html#raw_print


# paper talk
    1. What will my number speak for april 10th
       conversion cost

    2. other deadlines
    3. 


--> get single node from ri2, let only your code run
--> email jon smith for deadline
 




