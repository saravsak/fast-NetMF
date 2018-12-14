# fast-NetMF
A fast implementation of NetMF for multi-core CPUs and GPUs

cd gpu
./exec small for NetMF Small
./exec large for NetMF large

These will both run on the blogcatalog dataset and produce two files:

blogcatalog.emb: The embedding file in gensim format
profile.txt: Performance report

