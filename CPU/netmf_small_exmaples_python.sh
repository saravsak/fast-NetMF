python netmf.py --input ./datasets/small_4_weight.mat --matfile-variable-name network --output ./datasets/small_4_weight.embedding --rank 2 --dim 3
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n6_e10_weighted.mat --matfile-variable-name network --output ./datasets/small_n6_e10_weighted.embedding --rank 3 --dim 4
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n7_e12_weighted.mat --matfile-variable-name network --output ./datasets/small_n7_e12_weighted.embedding --rank 3 --dim 5
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n8_e15_weighted.mat --matfile-variable-name network --output ./datasets/small_n8_e15_weighted.embedding --rank 4 --dim 6 
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n9_e15_weighted.mat --matfile-variable-name network --output ./datasets/small_n9_e15_weighted.embedding --rank 4 --dim  7
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n10_e19_weighted.mat --matfile-variable-name network --output ./datasets/small_n10_e19_weighted.embedding --rank 5 --dim 8 


echo "\n\n=================================================  Small Graph Size =========================================================="
echo "==================================================================================================================================\n\n"

python netmf.py --input ./datasets/small_4_weight.mat --matfile-variable-name network --output ./datasets/small_4_weight.embedding --rank 2 --dim 3 --small
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n6_e10_weighted.mat --matfile-variable-name network --output ./datasets/small_n6_e10_weighted.embedding --rank 3 --dim 4 --small
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n7_e12_weighted.mat --matfile-variable-name network --output ./datasets/small_n7_e12_weighted.embedding --rank 3 --dim 5 --small
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n8_e15_weighted.mat --matfile-variable-name network --output ./datasets/small_n8_e15_weighted.embedding --rank 4 --dim 6  --small
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n9_e15_weighted.mat --matfile-variable-name network --output ./datasets/small_n9_e15_weighted.embedding --rank 4 --dim  7 --small
echo "----------------------------------------------------------------------------------------------------------"
python netmf.py --input ./datasets/small_n10_e19_weighted.mat --matfile-variable-name network --output ./datasets/small_n10_e19_weighted.embedding --rank 5 --dim 8  --small

