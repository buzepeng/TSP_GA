#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__device__ float d_sum;
static int N;

class TSP_GA{
    thrust::device_vector<float>    city_dis, distance, old_fit, new_fit;
    thrust::device_vector<int>      old_pop, new_pop, pop_min; 
    
    public:

    void InitCityAndPop(std::string filename, int s, float vx = 14.5, float vy = 42.0, float tx = 0, float ty = 0);

    thrust::host_vector<int> find_shortest(int s, float c, float pc, float pm, int times);

    void SaveResult(std::string filepath, thrust::host_vector<int>& res);
};