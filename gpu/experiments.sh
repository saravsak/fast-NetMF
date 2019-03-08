#!/bin/bash

source ~/.profile

declare -a WINDOW=( "3" "10" )
declare -a DATASET=( "blogcatalog" "ppi" "pubmed" "microsoft" "flickr" )

for D in "${DATASET[@]}"
do
	for W in "${WINDOW[@]}"
	do
		./netmf_small_dense_hybrid $D $W 128 1 ../../nrl-data/${D}.edgelist ../embeddings/${D}_${W}_128_1_small_dense_hybrid.emb ../mapping/${D}_${W}_128_1_small_dense_hybrid.map
	done
done

for D in "${DATASET[@]}"
do
	for W in "${WINDOW[@]}"
	do
		./netmf_large_sparse_hybrid $D $W 128 1 ../../nrl-data/${D}.edgelist ../embeddings/${D}_${W}_128_1_large_dense_hybrid.emb ../mapping/${D}_${W}_128_1_large_dense_hybrid.map 256
	done
done



