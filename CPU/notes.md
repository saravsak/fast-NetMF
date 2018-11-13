

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


