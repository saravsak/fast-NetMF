#!/bin/bash

mkdir ../data/
cd ../data/

# Download BlogCatalog dataset
wget -O blogcatalog.mat http://leitang.net/code/social-dimension/data/blogcatalog.mat

# Download PPI dataset
wget -O ppi.mat http://snap.stanford.edu/node2vec/Homo_sapiens.mat

# Download Wiki dataset
wget -O wikipedia.mat http://snap.stanford.edu/node2vec/POS.mat

# Download Flickr dataser
wget -O flickr.mat  http://leitang.net/code/social-dimension/data/flickr.mat

cd ../src/
