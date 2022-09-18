#include "TSP_GA.cuh"
#include "utils.h"

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

class TSP_GA{
    thrust::device_vector<float>    city_dis, distance, old_fit, new_fit;
    thrust::device_vector<int>      old_pop, new_pop, pop_min; 
    
    public:
    TSP_GA(){}

    void InitCityAndPop(std::string filename, int s, float vx = 14.5, float vy = 42.0, float tx = 0, float ty = 0){
        //read city coordinates
        std::ifstream fp(filename, std::ios::in);
        std::string line;
        thrust::host_vector<int2>   h_city_coors;
        if(!fp.is_open()){
            std::cout<<"Error: opening file fail"<<std::endl;
            std::exit(1);
        }
        int cnt = 0;
        while(std::getline(fp, line)){
            std::istringstream sin(line);
            std::string temp;

            sin>>temp;
            if(!isStringNumber(temp)){
                sin>>temp;
                if(isStringNumber(temp)){
                    N = std::stoi(temp);
                    h_city_coors.resize(N);
                }
                continue;
            }
          
            sin>>h_city_coors[cnt].x>>h_city_coors[cnt].y;
            cnt++;
        }
        fp.close();

        //compute city distance
        thrust::host_vector<float> city_distance(N*N), coor_x_tmp1(N*N), coor_x_tmp2(N*N), coor_y_tmp1(N*N), coor_y_tmp2(N*N), lx(N*N), ly(N*N);
        for(int i = 0;i<N;i++){
            for(int j = 0;j<N;j++){
                coor_x_tmp1[i*N+j] = h_city_coors[i].x;
                coor_x_tmp2[j*N+i] = h_city_coors[i].x;
                coor_y_tmp1[i*N+j] = h_city_coors[i].y;
                coor_y_tmp2[j*N+i] = h_city_coors[i].y;
            }
        }
        thrust::transform(thrust::host, coor_x_tmp1.begin(), coor_x_tmp1.end(), coor_x_tmp2.begin(), lx.begin(), [=]__host__(float tmp1, float tmp2){
            return (double)(abs(tmp1-tmp2)/vx+tx);
        });
        thrust::transform(thrust::host, coor_y_tmp1.begin(), coor_y_tmp1.end(), coor_y_tmp2.begin(), ly.begin(), [=]__host__(float tmp1, float tmp2){
            return (double)(abs(tmp1-tmp2)/vy+ty);
        });
        thrust::transform(thrust::host, lx.begin(), lx.end(), ly.begin(), city_distance.begin(), [=]__host__(float x, float y){
            return max(x, y);
        });
        city_dis = city_distance;

        //generate population
        thrust::host_vector<int> h_pop(s*N, 0);
        for(int i = 0;i<s;i++){
            int sb = rand() % N;
            std::vector<bool> visited(N, false);
            visited[sb] = true;
            int *path = h_pop.data()+i*N;
            path[0] = sb;
            for(int j = 0;j<N-1;j++){
                thrust::host_vector<float> q(city_distance.begin()+sb*N, city_distance.begin()+(sb+1)*N);
                for(int k = 0;k<N;k++)  if(visited[k])  q[k] = FLT_MAX;
                int location = thrust::min_element(thrust::host, q.begin(), q.end())-q.begin();
                path[j+1] = location;
                visited[location] = true;
                sb = location;
            }
        }
        
        new_pop = h_pop;
        old_pop.resize(s*N);
        old_fit.resize(s);
        new_fit.resize(s);
        distance.resize(s);
        pop_min.resize(N);
    }
    
    thrust::host_vector<int> find_shortest(int s, float c, float pc, float pm, int times){
        thrust::host_vector<float> h_prob_a(s), h_prob_b(s);
        thrust::host_vector<int> h_cross_pos(2*s), h_mutation_pos(2*s);
        thrust::device_vector<float> pc1(int(s/2)),pm1(s), d_prob_a(s), d_prob_b(s);
        thrust::device_vector<int> old_ind(s), new_ind(s), d_cross_pos(2*s), d_mutation_pos(2*s);
        thrust::device_vector<int> ind(s);
        thrust::sequence(thrust::device, ind.begin(), ind.end());
        int *old_pop_ptr = thrust::raw_pointer_cast(old_pop.data());
        int *new_pop_ptr = thrust::raw_pointer_cast(new_pop.data());
        int *old_ind_ptr = thrust::raw_pointer_cast(old_ind.data());
        int *new_ind_ptr = thrust::raw_pointer_cast(new_ind.data());
        int *pop_min_ptr = thrust::raw_pointer_cast(pop_min.data());
        int *cross_pos_ptr = thrust::raw_pointer_cast(d_cross_pos.data());
        int *mutation_pos_ptr = thrust::raw_pointer_cast(d_mutation_pos.data());
        float *pc1_ptr = thrust::raw_pointer_cast(pc1.data());
        float *pm1_ptr = thrust::raw_pointer_cast(pm1.data());
        float *old_fit_ptr = thrust::raw_pointer_cast(old_fit.data());
        float *new_fit_ptr = thrust::raw_pointer_cast(new_fit.data());
        float *city_dis_ptr = thrust::raw_pointer_cast(city_dis.data());
        float *prob_a_ptr = thrust::raw_pointer_cast(d_prob_a.data());
        float *prob_b_ptr = thrust::raw_pointer_cast(d_prob_b.data());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), new_fit.begin(), old_fit.begin()));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(s), new_fit.end(), old_fit.end()));

        int next_pow2N = getCeilPowerOfTwo(N), next_pow2s = getCeilPowerOfTwo(s), new_rows = s*(1-c), old_rows = s - new_rows, max_fit_offset = 0;

        float min11 = 0, max_fit = 0, sum = 0;
        individual_fit<<<s,next_pow2N,next_pow2N*sizeof(float)>>>(new_pop_ptr,city_dis_ptr, new_fit_ptr, N, s);

        sum = thrust::reduce(thrust::device, new_fit.begin(),new_fit.end());

        time_t t;
        srand((unsigned int)time(&t));

        for(int time = 0;time<times;time++){
            old_pop = new_pop;
            old_fit = new_fit;
            min11 = *thrust::min_element(thrust::device, new_fit.begin(), new_fit.end());
            thrust::generate(thrust::host, h_prob_a.begin(), h_prob_a.end(), rand_01);
            thrust::generate(thrust::host, h_prob_b.begin(), h_prob_b.end(), rand_01);
            thrust::generate(thrust::host, h_cross_pos.begin(), h_cross_pos.end(), rand_N);
            thrust::generate(thrust::host, h_mutation_pos.begin(), h_mutation_pos.end(), rand_N);
            d_prob_a = h_prob_a;
            d_prob_b = h_prob_b;
            d_cross_pos = h_cross_pos;
            d_mutation_pos = h_mutation_pos;

            if(time%100==0) std::cout<<"iter: "<<time<<", max fit:"<<max_fit<<std::endl;
            thrust::transform(thrust::device, new_fit.begin(), new_fit.begin()+int(s/2), new_fit.begin()+int(s/2), pc1.begin(), [=]__device__(float fit1, float fit2)->float{
            float bj = fit1<fit2?fit1:fit2;
            if(bj<=sum/s)   return pc*(bj-min11)/(sum/s-min11);
            else            return pc;
            });
            thrust::transform(thrust::device, new_fit.begin(), new_fit.end(), pm1.begin(), [=]__device__(float fit){
            if(fit<=sum/s)  return pm*(fit-min11)/(sum/s-min11);
            else            return pm;
            });
            // CrossVariation
            crossover<<<s,N,5*N*sizeof(int)>>>(new_pop_ptr, pc1_ptr, cross_pos_ptr, prob_a_ptr, N, s);

            // Mutation
            mutation<<<s,N,N*sizeof(int)>>>(new_pop_ptr, pm1_ptr, mutation_pos_ptr, prob_b_ptr, N,s);

            // GroupFit
            individual_fit<<<s,next_pow2N,next_pow2N*sizeof(float)>>>(new_pop_ptr,city_dis_ptr, new_fit_ptr, N, s);

            // ChooseParents
            old_ind = ind;
            new_ind = ind;
            thrust::sort_by_key(thrust::device, old_fit.begin(), old_fit.end(), old_ind.begin(), thrust::greater<float>());
            thrust::sort_by_key(thrust::device, new_fit.begin(), new_fit.end(), new_ind.begin(), thrust::greater<float>());
            choose_parents<<<s,N>>>(old_pop_ptr, new_pop_ptr, old_ind_ptr, new_ind_ptr, old_fit_ptr, new_fit_ptr, old_rows, new_rows, N, s);

            thrust::device_vector<float>::iterator max_ptr = thrust::max_element(thrust::device, new_fit.begin(), new_fit.end());
            float max_val = *max_ptr;

            if(max_val>max_fit){
                std::cout<<"save new pop_min"<<std::endl;
                max_fit_offset = max_ptr - new_fit.begin();
                max_fit = max_val;
            }
        }
        thrust::copy(thrust::device, new_pop.begin()+N*max_fit_offset, new_pop.begin()+N*(max_fit_offset+1), pop_min.begin());
        thrust::host_vector<int> fittest_pop = pop_min;
        return std::move(fittest_pop);
    }
    void SaveResult(std::string filepath, thrust::host_vector<int>& res){
        std::ofstream fp(filepath, std::ios::out);
        if(!fp.is_open()){
            std::cout<<"Error: saving result fail"<<std::endl;
            std::exit(1);
        }
        for(int i = 0;i<res.size();i++){
            // std::cout<<res[i]<<std::endl;
            fp<<res[i]<<" ";
        }
        fp.close();
        std::cout<<"solution saved!"<<std::endl;
    }   
};

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