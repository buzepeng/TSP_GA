#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <curand_kernel.h>
#include <tuple>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string.h>

__device__ float d_sum;
static int N;

