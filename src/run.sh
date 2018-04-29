rm out 2> /dev/null
nvcc -o out -g pagerank.cu p_arnoldi.cu filereader.cpp matrix.cpp -lcusolver -lm  -lpthread -Xcompiler -fopenmp -arch=sm_35
./out
#python Datasets/dataset.py small_temp.txt ranking_small.txt
