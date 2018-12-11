./netmf_sparse --dataset small_4_weight --dataset_folder ./datasets --rank 2 --dim 3
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n6_e10_weighted --dataset_folder ./datasets --rank 3 --dim 4
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n7_e12_weighted --dataset_folder ./datasets --rank 3 --dim 5
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n8_e15_weighted --dataset_folder ./datasets --rank 4 --dim 6 
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n9_e15_weighted --dataset_folder ./datasets --rank 4 --dim  7
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n10_e19_weighted --dataset_folder ./datasets --rank 5 --dim 8 


echo "\n\n=================================================  Small Graph Size =========================================================="
echo "==================================================================================================================================\n\n"


./netmf_sparse --dataset small_4_weight --dataset_folder ./datasets --rank 2 --dim 3 --graph_size small
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n6_e10_weighted --dataset_folder ./datasets --rank 3 --dim 4 --graph_size small
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n7_e12_weighted --dataset_folder ./datasets --rank 3 --dim 5 --graph_size small
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n8_e15_weighted --dataset_folder ./datasets --rank 4 --dim 6  --graph_size small
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n9_e15_weighted --dataset_folder ./datasets --rank 4 --dim  7 --graph_size small
echo "----------------------------------------------------------------------------------------------------------"
./netmf_sparse --dataset small_n10_e19_weighted --dataset_folder ./datasets --rank 5 --dim 8  --graph_size small