// #include <curand_kernel.h>
// #include <stdio.h>

// #define RANGE 9

// __global__ void generateRandom(long rand)
// {
//     curandState state;
//     int id = threadIdx.x;
//     long seed = rand;
//     // int seed = id;
//     curand_init(seed, id, 0, &state);

//     printf("rand: %ld seed:%ld id:%d \n",rand, seed, threadIdx.x);
//     // printf("id:%ld \n", seed);

//     printf("random double: %f \n",curand_uniform(&state));

//     printf("random int: %d \n", (int)((curand_uniform(&state)*9.9)));
// }

// int main()

// {

//        srand((unsigned int)time(NULL));

//        cudaSetDevice(0);

//        generateRandom<<<1,16>>>(rand());
//        cudaDeviceReset();

//        return 0;

// }

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstdlib>
int main(){
    thrust::host_vector<float> h_vec(100);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    thrust::device_vector<float> d_vec = h_vec;

    thrust::device_vector<float>::iterator iter =
    thrust::max_element(d_vec.begin(), d_vec.end());

    unsigned int position = iter - d_vec.begin();
    float max_val = *iter;

    std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;
    // thrust::for_each(thrust::host, v.begin, v.end(), [])
    // h_v = v;
    // for(int i = 0;i<10;i++){
    //     std::cout<<h_v[i]/RAND_MAX<<" ";
    // }
}