__device__ void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

__host__ static __inline__ float rand_01()
{
    return ((float)rand()/RAND_MAX);
}

__host__ static __inline__ int rand_N()
{
    return ((int)rand()%N);
}


__global__ void individual_fit(int *pop, float* city_distance, float* fitness, int N, int s){
    int ix = threadIdx.x;
    int iy = blockIdx.x;
    extern __shared__ float s_distance[];
    s_distance[ix] = ix<N?city_distance[pop[iy*N+ix]*N+pop[iy*N+(ix+1)%N]]:0;
    __syncthreads();

    for(int strid = blockDim.x>>1;strid>0;strid = strid>>1){
        if(ix<strid){
            s_distance[ix] += s_distance[ix+strid];
        }
        __syncthreads();
    }
    if(ix==0){
        fitness[iy] = 1/s_distance[0];
    }
    __syncthreads();
}

__global__ void mutation(int *pop, float* pm1, int* mutation_pos, float* prob_b, int N, int s){
    int ix = threadIdx.x;
    int iy = blockIdx.x;

    __shared__ float b;
    __shared__ int start, end;

    b = prob_b[iy];
    start = min(mutation_pos[iy], mutation_pos[iy+s]), end = max(mutation_pos[iy], mutation_pos[iy+s]);

    extern __shared__ int s_pop[];
    s_pop[ix] = pop[iy*N+ix];
    __syncthreads();
    if(ix>=start && ix<=end && iy<s && b<pm1[iy]){
        pop[iy*N+ix] = s_pop[end-(ix-start)];
    }
}

__global__ void choose_parents(int *old_pop, int *new_pop, int *old_ind, int *new_ind, float *old_fit, float *new_fit, int old_rows, int new_rows, int N, int s){
    int ix = threadIdx.x;
    int iy = blockIdx.x;

    if(ix<N && iy<s){
        new_pop[iy*N+ix] = iy<new_rows?new_pop[new_ind[iy]*N+ix]:old_pop[old_ind[iy-new_rows]*N+ix];
        if(ix == 0){
            new_fit[iy] = iy<new_rows?new_fit[iy]:old_fit[iy-new_rows];
        }
    }
}

__global__ void crossover(int* pop, float* pc1, int* cross_pos, float* prob_a, int N, int s){
    int ix = threadIdx.x;
    int iy = blockIdx.x;

    __shared__ float a;
    __shared__ int start, end;

    a = prob_a[iy];
    start = min(cross_pos[iy], cross_pos[iy+s]), end = max(cross_pos[iy], cross_pos[iy+s]);
    
    extern __shared__ int s_buffer[];
    int *baby1 = (int*)&s_buffer[0], *baby2 = (int*)&s_buffer[N], *lookup1 = (int*)&s_buffer[2*N], *lookup2 = (int*)&s_buffer[3*N], *mum = &pop[iy*N], *dad = &pop[(iy+s/2)*N];

    baby1[ix] = mum[ix];
    baby2[ix] = dad[ix];
    lookup1[mum[ix]] = ix;
    lookup2[dad[ix]] = ix;
    __syncthreads();

    if(ix==0 && iy < s/2 && a<pc1[iy]){
        for(int j = start;j<=end;j++){
            int gene1 = mum[j], gene2 = dad[j], posgene1, posgene2;
            posgene1 = lookup1[gene1];
            posgene2 = lookup1[gene2];
            swap(&baby1[posgene1], &baby1[posgene2]);
            swap(&lookup1[baby1[posgene1]], &lookup1[baby1[posgene2]]);

            posgene1 = lookup2[gene1];
            posgene2 = lookup2[gene2];
            swap(&baby2[posgene1], &baby2[posgene2]);
            swap(&lookup2[baby2[posgene1]], &lookup2[baby2[posgene2]]);
        }
    }
    __syncthreads();
    mum[ix] = baby1[ix];
    dad[ix] = baby2[ix];

}
