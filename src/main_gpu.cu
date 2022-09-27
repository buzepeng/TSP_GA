#include "TSP_GA.cuh"

int main(){
    int s = 100, times = 10000;
    float c = 0.2, pc = 0.9, pm = 0.2;
    cudaFree(0);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);
    TSP_GA tsp;
    tsp.InitCityAndPop("../TSPlib/kroA100.tsp", s);
    thrust::host_vector<int> opt = tsp.find_shortest(s, c, pc, pm, times);
    tsp.SaveResult("../TSPlib/bestleader.txt", opt);
}