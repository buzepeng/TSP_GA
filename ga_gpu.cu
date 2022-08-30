#include <iostream>
#include <algorithm>
#include <functional>
#include <fstream>
#include <vector>
#include <sstream>
#include <random>
#include <chrono>

// Program includes
// #include "ga_gpu.h"
// #include "common.h"

using namespace std;

struct City
{
//Store location - city
	int x, y;
};

typedef struct World
{
	// 2D world for the TSP
	int num_cities;    
	City* cities;      
	float fitness;     // The current fitness
	float fit_prob;    // The fitness probability

	inline __host__ void calc_fitness()
	{
		float distance = 0.0;
		for (int i=0; i<num_cities-1; i++)
			distance += (cities[i].x - cities[i + 1].x) * (cities[i].x -      \
				cities[i +1 ].x) + (cities[i].y - cities[i + 1].y)     *      \
				(cities[i].y - cities[i + 1].y);
		fitness = 1.0 / distance;
	}

	inline __host__ float calc_distance()
	{
		float distance = 0.0;
		for (int i=0; i<num_cities-1; i++)
			distance += (float)sqrt((float)((cities[i].x - cities[i + 1].x) * \
				(cities[i].x - cities[i + 1].x) + (cities[i].y              - \
				cities[i +1 ].y) * (cities[i].y - cities[i + 1].y)));
		return distance;
	}
} World;

bool checkForError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		cout << cudaGetErrorString(error) << endl;
		return true;
	}
	else
	{
		return false;
	}
}

bool checkForKernelError(const char *err_msg)
{
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		cout << err_msg << cudaGetErrorString(status) << endl;
		return true;
	}
	else
	{
		return false;
	}
}

//Read data
vector<pair<int, int>> get_coor(string filename){
    ifstream dataFile(filename, ios::in);
    if(!dataFile.is_open()){
        cout<<"file does not exit!";
    }
    vector<pair<int, int>> coor;
    string data_line;
    while(getline(dataFile, data_line)){
        if(data_line == "EOF")   break;
        istringstream ss(data_line);
        string data;
        ss >> data;
        if(isdigit(data[0])){
            string x, y;
            ss >> x;
            ss >> y;
            coor.push_back(make_pair(stoi(x), stoi(y)));
        }
    }
    return coor;
}

void init_world(World* world, int num_cities)
{
	world->num_cities = num_cities;
	world->fitness    = (float)0.0;
	world->fit_prob   = (float)0.0;
	world->cities     = new City[num_cities * sizeof(City)];
}

void clone_city(City* src, City* dst, int num_cities)
{
memcpy(dst, src, num_cities * sizeof(City));
}

void clone_world(World* src, World* dst)
{
	dst->num_cities = src->num_cities;
	dst->fitness    = src->fitness;
	dst->fit_prob   = src->fit_prob;
	clone_city(src->cities, dst->cities, src->num_cities);
}

void free_world(World* world)
{
	delete[] world->cities;
	delete[] world;
}

//Functions for both CPU and GPU
void make_world(World* world, string filename, int seed)
{
	// Random number generation
	mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));
	
	// Create a set to deal with uniqueness
	vector<pair<int, int>> coordinates = get_coor(filename);
	vector<pair<int, int>>::iterator it;
	
	init_world(world,coordinates.size());
	// Add those cities to the world
	{
		int i = 0;
		for (it=coordinates.begin(); it!=coordinates.end(); it++)
		{
			world->cities[i].x = (*it).first;
			world->cities[i].y = (*it).second;
			i++;
		}
	}
}

bool g_soft_clone_world(World* d_world, World* h_world)
{
	// Error checking
	bool error;
	
	// error = checkForError(cudaMemcpy(&d_world->width, &h_world->width,        \
	// 	sizeof(int), cudaMemcpyHostToDevice));
	// if (error)
	// return true;
	// error = checkForError(cudaMemcpy(&d_world->height, &h_world->height,      \
	// 	sizeof(int), cudaMemcpyHostToDevice));
	// if (error)
	// return true;
	error = checkForError(cudaMemcpy(&d_world->num_cities,                    \
		&h_world->num_cities, sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	return true;

	return false;
}

bool g_init_world(World* d_world, World* h_world)
{
	// Error checking
	bool error;
	
	// Soft clone world
	error = g_soft_clone_world(d_world, h_world);
	if (error)
		return true;
	
	// Allocate space for cities on device
	City *d_city;
	error = checkForError(cudaMalloc((void**)&d_city, h_world->num_cities * sizeof(City)));
	if (error)
	return true;
	
	// Update pointer on device
	error = checkForError(cudaMemcpy(&d_world->cities, &d_city, sizeof(City*), cudaMemcpyHostToDevice));
	if (error)
	return true;
	
	return false;
}

void print_status(World* generation_leader, World* best_leader, int generation)
{
	cout << "Generation " << generation << ":" << endl;
	cout << "  Current Leader's Fitness: "  << generation_leader->fitness << endl;
	cout << "  Best Leader's Fitness: "  << best_leader->fitness << endl;
}

void storeBestLeader(World* bestleader, string path){
	ofstream resFile(path, ios::out);
	for(int i = 0;i<bestleader->num_cities;i++){
		resFile<<bestleader->cities[i].x<<" "<<bestleader->cities[i].y<<endl;
	}
	resFile.close();
}

__device__ int getGlobalIdx_2D_1D()
{
	int blockId  = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	return threadId;
}

__device__ void crossover(World* old_pop, World* new_pop, int* sel_ix,  int* cross_loc, int tid)
{

	// Copy elements from first parent up through crossover point
	memcpy(new_pop[tid].cities, old_pop[sel_ix[2*tid]].cities,(cross_loc[tid] + 1) * sizeof(City));

	// Add remaining elements from second parent to child, in order
	int remaining = old_pop[tid].num_cities - cross_loc[tid] - 1;
	int count     = 0;
	for (int i=0; i<old_pop[tid].num_cities; i++)
	{
		bool in_child = false;
		for (int j=0; j<=cross_loc[tid]; j++)     
		{
			// If the city is in the child, exit
			if ((new_pop[tid].cities[j].x == old_pop[sel_ix[2 * tid + 1]].cities[i].x) &               
				(new_pop[tid].cities[j].y == old_pop[sel_ix[2 * tid + 1]].cities[i].y))
			{
				in_child = true;
				break;
			}
		}
		if (!in_child)
		{
			count++;
			memcpy(&new_pop[tid].cities[cross_loc[tid] + count],&old_pop[sel_ix[2 * tid + 1]].cities[i], sizeof(City));
		}
	
		// Stop once all of the cities have been added
		if (count == remaining) break;
	}
}

__device__ void mutate(World* new_pop, int* mutate_loc, int tid)
{
	// Swap the elements
	City temp = *(new_pop[tid].cities + mutate_loc[2*tid]);
	*(new_pop[tid].cities + mutate_loc[2*tid])   = *(new_pop[tid].cities + mutate_loc[2*tid+1]);
	*(new_pop[tid].cities + mutate_loc[2*tid+1]) = temp;
}

__global__ void fitness_kernel(World* pop, int pop_size)
{
	// Get the thread id
	int tid = getGlobalIdx_2D_1D();
	
	// Evaluate if the thread is valid
	if (tid < pop_size)
	{
		float distance = (float)0.0; // Total "normalized" "distance"
		// Calculate fitnesses using the fitness formula that will be explained in the reports
		int N = pop[tid].num_cities;
		for (int i=0; i<pop[tid].num_cities-1; i++){
			distance += (pop[tid].cities[i].x - pop[tid].cities[(i + 1)%N].x) * (pop[tid].cities[i].x - pop[tid].cities[(i + 1)%N].x)         
				+ (pop[tid].cities[i].y - pop[tid].cities[(i + 1)%N].y)*(pop[tid].cities[i].y - pop[tid].cities[(i + 1)%N].y);
		}
		pop[tid].fitness = 3950.0 * 1969 / distance;
	}
}

__global__ void fit_sum_kernel(World* pop, int pop_size, float* fit_sum)
{
	int tid = getGlobalIdx_2D_1D();
	
	// Evaluate if the thread is valid
	if (tid < pop_size)
	{
		// Sum of all fitness
		float sum = (float)0.0;
		
		// Calculate the partial sum
		for (int i=0; i<=tid; i++)
			sum += pop[i].fitness;
		// printf("sum:%.10f\n", sum);
		pop[tid].fit_prob = sum;

		if (tid == (pop_size - 1))	*fit_sum = sum;
	}
}

__device__ float max_fitsum_kernel(int pop_size, float* fit_sum){
	float max = fit_sum[0];
	for(int i = 0;i<pop_size;i++){
		if(fit_sum[i]>max)	max = fit_sum[i];
	}
	return max;
}

__device__ float min_fitsum_kernel(int pop_size, float* fit_sum){
	float min = fit_sum[0];
	for(int i = 0;i<pop_size;i++){
		if(fit_sum[i]<min)	min = fit_sum[i];
	}
	return min;
}

__global__ void fit_prob_kernel(World* pop, int pop_size, float* fit_sum)
{
	// Get the thread id
	int tid = getGlobalIdx_2D_1D();
	// float fit_sum_max = max_fitsum_kernel(pop_size, fit_sum);
	// float fit_sum_min = min_fitsum_kernel(pop_size, fit_sum);
	float factor = max_fitsum_kernel(pop_size, fit_sum) - min_fitsum_kernel(pop_size, fit_sum);
	// printf("factor:%.10f\n", factor);
	// Evaluate if the thread is valid
	if (tid < pop_size)
		pop[tid].fit_prob /= *fit_sum;
		// printf("fit_sum:%.10f, prob:%.10f\n", *fit_sum, pop[tid].fit_prob);
}

__global__ void max_fit_kernel(World* pop, int pop_size, World* gen_leader)
{
	// Get the thread id
	int tid = getGlobalIdx_2D_1D();

	// Evaluate if the thread is valid
	if (tid < pop_size)
	{
		if (tid == 0)
		{
			float max = (float)0.0;
			int ix  = 0;
			for (int i=1; i<pop_size; i++)
			{
				if (pop[i].fitness > max)
				{
					max = pop[i].fitness;
					ix  = i;
				}
			}
			gen_leader->cities  = pop[ix].cities;
			gen_leader->fitness = max;
		}
		else if (tid == 1)
		{
			gen_leader->num_cities = pop[0].num_cities;
		}
	}
}

__global__ void selection_kernel(World* pop, int pop_size, float* rand_nums,  \
	int* sel_ix)
{
	// Get the thread id
	int tid = getGlobalIdx_2D_1D();

	// Evaluate if the thread is valid
	if (tid < (2 * pop_size))
	{
		// Select the parents
		for (int j=0; j<pop_size; j++)
		{
			if (rand_nums[tid] <= pop[j].fit_prob)
			{
				sel_ix[tid] = j;
				break;
			}
		}
	}
}

__global__ void child_kernel(World* old_pop, World* new_pop, int pop_size,    \
	int* sel_ix, float prob_crossover, float* prob_cross, int* cross_loc,     \
	float prob_mutation, float* prob_mutate, int* mutate_loc)
{
	// Get the thread id
	int tid = getGlobalIdx_2D_1D();

	// Evaluate if the thread is valid
	if (tid < pop_size)
	{
		// Determine how many children are born
		if (prob_cross[tid] <= prob_crossover)
		{
			crossover(old_pop, new_pop, sel_ix, cross_loc, tid);
			if(prob_mutate[tid] <= prob_mutation)
				mutate(new_pop, mutate_loc, tid);
			
		}
		else // Select the first parent
		{
			// Add child to new population
			memcpy(new_pop[tid].cities, old_pop[sel_ix[2*tid]].cities, old_pop[tid].num_cities * sizeof(City));
			if(prob_mutate[tid] <= prob_mutation)
				mutate(new_pop, mutate_loc, tid);
		}
	}
}

bool g_initialize(World* world, World* pop, int pop_size, int seed)
{
	bool error;
	World h_world;
	h_world.cities = new City[world->num_cities * sizeof(City)];
	srand(seed);
	for (int i=0; i<pop_size; i++)
	{
		clone_world(world, &h_world);
		// Randomly adjust the path between cities
		random_shuffle(&h_world.cities[0], &h_world.cities[world->num_cities]);
		// Copy world to device
		error = g_soft_clone_world(&pop[i], &h_world);
		if (error)
		return true;

		// Allocate space for cities on device
		City *d_city;
		error = checkForError(cudaMalloc((void**)&d_city, world->num_cities * sizeof(City)));
		if (error)
		return true;

		// Copy cities to device
		error = checkForError(cudaMemcpy(d_city, h_world.cities,world->num_cities * sizeof(City), cudaMemcpyHostToDevice));
		if (error)
		return true;

		// Update pointer on device
		error = checkForError(cudaMemcpy(&pop[i].cities, &d_city, sizeof(City*), cudaMemcpyHostToDevice));
		if (error)
		return true;
	}

	// Success
	delete[] h_world.cities; return false;
}

bool g_evaluate(World *pop, int pop_size, dim3 Block, dim3 Grid)
{
	bool error;
	
	// Allocate fitness sum on the GPU
	float *fit_sum_d;
	error = checkForError(cudaMalloc((void**)&fit_sum_d, sizeof(float)));
	if (error)
	return true;
	
	// Calculate the fitnesses
	fitness_kernel <<< Grid, Block >>> (pop, pop_size);
	cudaDeviceSynchronize();
	if (checkForKernelError("fitness_kernel is failing "))
		return true;
	// Calculate the total sum and compute the partial probabilities
	fit_sum_kernel <<< Grid, Block >>> (pop, pop_size, fit_sum_d);
	cudaDeviceSynchronize();
	if (checkForKernelError("fit_sum_kernel failing "))
		return true;

	// Compute the full probabilities
	fit_prob_kernel <<< Grid, Block >>> (pop, pop_size, fit_sum_d);
	cudaDeviceSynchronize();
	if (checkForKernelError("fit_prob_kernel failing "))
		return true;

	cudaFree(fit_sum_d); 
	return false;
}

int g_select_leader(World* pop, int pop_size, World* generation_leader,
	World* best_leader, dim3 Block, dim3 Grid)
{
	// Error handling
	bool error;

	// Initialize world for device generation leader
	World *gen_leader_d;
	error = checkForError(cudaMalloc((void**)&gen_leader_d, sizeof(World)));
	if (error)
	return -1;
	// Purposely don't allocate space for the cities, simply use a pointer.
	// Since we only need to copy data to cpu 
	error = g_soft_clone_world(gen_leader_d, generation_leader);
	if (error)
	return -1;

	// Calculate the max fitness
	max_fit_kernel <<< Grid, Block >>> (pop, pop_size, gen_leader_d);
	cudaDeviceSynchronize();
	if (checkForKernelError("*** Max fitness kernel failed: "))
	return -1;

	// Copy results from device
	City *h_ptr = generation_leader->cities;
	City *d_ptr;
	error = checkForError(cudaMemcpy(generation_leader, gen_leader_d, sizeof(World), cudaMemcpyDeviceToHost));
	if (error)
	{
		cout<<"memcpy generation leader error!"<<endl;
		return -1;
	}
	d_ptr = generation_leader->cities;
	generation_leader->cities = h_ptr;
	error = checkForError(cudaMemcpy(generation_leader->cities,  d_ptr, generation_leader->num_cities * sizeof(City), cudaMemcpyDeviceToHost));
	if (error){
		cout<<"memcpy cities error!"<<endl;
		return -1;
	}

	// Update best leader
	if (generation_leader->fitness > best_leader->fitness)
	{
		clone_world(generation_leader, best_leader);
		cudaFree(gen_leader_d); return 1;
	}

	// Success
	cudaFree(gen_leader_d);	return 0;
}

bool g_execute(float prob_mutation, float prob_crossover, int pop_size,
	int max_gen, World* world, int seed)
{
	cudaSetDevice(2);
	// Error checking variables
	bool error;
	
	// Random number generation
	mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));
	
	// Tile and grid variables
	int tile_size  = 1024;
	int grid_size  = (int)ceil((float)pop_size / tile_size);
	int grid_size2 = (int)ceil((float)(2 * pop_size) / tile_size);
	dim3 Block(tile_size);
	dim3 Grid(grid_size, grid_size);
	dim3 Grid2(grid_size2, grid_size2);
	
	int pop_bytes  = pop_size * sizeof(World);
	World *old_pop_d, *new_pop_d;

	// Random numbers
	float *prob_select = new float[2 * pop_size * sizeof(float)];
	float *prob_cross  = new float[pop_size * sizeof(float)];
	float *prob_mutate = new float[pop_size * sizeof(float)];
	int   *cross_loc   = new int[pop_size * sizeof(int)];
	int   *mutate_loc  = new int[2 * pop_size * sizeof(int)];
	float *prob_select_d, *prob_cross_d, *prob_mutate_d;
	int   *cross_loc_d, *mutate_loc_d;
	
	int   sel;
	int   best_generation    = 0;
	World *best_leader       = new World[sizeof(World)];
	World *generation_leader = new World[sizeof(World)];

	// Other "temporary" parameters
	int *sel_ix_d;

	//Initializations
	init_world(best_leader, world->num_cities);
	init_world(generation_leader, world->num_cities);
	if (checkForError(cudaMalloc((void**) &old_pop_d, pop_bytes)))
	return true;
	if (checkForError(cudaMalloc((void**) &new_pop_d, pop_bytes)))
	return true;
	if (checkForError(cudaMalloc((void**) &prob_select_d, sizeof(float)* 2 * pop_size)))
	return true;
	if (checkForError(cudaMalloc((void**) &prob_cross_d, sizeof(float) * pop_size)))
	return true;
	if (checkForError(cudaMalloc((void**) &prob_mutate_d, sizeof(float) * pop_size)))
	return true;
	if (checkForError(cudaMalloc((void**) &cross_loc_d, sizeof(int) * pop_size)))
	return true;
	if (checkForError(cudaMalloc((void**) &mutate_loc_d, sizeof(int) * 2 * pop_size)))
	return true;
	// Other parameters
	if (checkForError(cudaMalloc((void**) &sel_ix_d, sizeof(int) * 2 * pop_size)))
	return true;

	///////// GPU Initializations
	// Populations
	error = g_initialize(world, old_pop_d, pop_size, seed);
	if (error)
	return true;
	for (int i=0; i<pop_size; i++)
	{
		error = g_init_world(&new_pop_d[i], world);
		if (error)
		return true;
	}

	// Calculate the fitnesses
	error = g_evaluate(old_pop_d, pop_size, Block, Grid);
	if (error)
	return true;
	
	// Initialize the best leader
	sel = g_select_leader(old_pop_d, pop_size, generation_leader,             \
		best_leader, Block, Grid);	
	if (-1 == sel)
	return true;	
	print_status(generation_leader, best_leader, 0);
		
	// Continue through all generations
	for (int i=0; i<max_gen; i++)
	{

		// Generate all probabilities for each step
		// Note : The order the random numbers are generated must be consistent to
		// ensure the results will match the CPU.
		for (int j=0; j<pop_size; j++)
		{
			prob_select[2*j]     = (float)rgen();
			prob_select[2*j + 1] = (float)rgen();
			prob_cross[j]        = (float)rgen();
			cross_loc[j]         = (int)(rgen() * (world->num_cities - 1));
			prob_mutate[j]       = (float)rgen();
			mutate_loc[2*j]      = (int)(rgen() * (world->num_cities));
			mutate_loc[2*j + 1]  = (int)(rgen() * (world->num_cities));
			while (mutate_loc[2*j + 1] == mutate_loc[2*j])
			{
				mutate_loc[2*j + 1] = (int)(rgen() * world->num_cities);
			}
		}
		
		// Copy random numbers to device
		if (checkForError(cudaMemcpy(prob_select_d, prob_select, 2 * pop_size * sizeof(float), cudaMemcpyHostToDevice)))
		return true;
		if (checkForError(cudaMemcpy(prob_cross_d, prob_cross, pop_size * sizeof(float), cudaMemcpyHostToDevice)))
		return true;
		if (checkForError(cudaMemcpy(prob_mutate_d, prob_mutate,pop_size * sizeof(float), cudaMemcpyHostToDevice)))
		return true;
		if (checkForError(cudaMemcpy(cross_loc_d, cross_loc,pop_size * sizeof(int), cudaMemcpyHostToDevice)))
		return true;
		if (checkForError(cudaMemcpy(mutate_loc_d, mutate_loc, 2 * pop_size * sizeof(int), cudaMemcpyHostToDevice)))
		return true;

		// Select the parents
		selection_kernel <<< Grid2, Block >>> (old_pop_d, pop_size, prob_select_d, sel_ix_d);
		cudaDeviceSynchronize();
		if (checkForKernelError("slection_kernel is failing "))
		return true;
		
		// Create the children (form the new population entirely on the GPU!)
		child_kernel <<< Grid, Block >>> (old_pop_d, new_pop_d, pop_size, sel_ix_d, prob_crossover, prob_cross_d, cross_loc_d, prob_mutation, prob_mutate_d, mutate_loc_d);
		cudaDeviceSynchronize();
		if (checkForKernelError("child_kernel is failing"))
		return true;
		
		// Calculate the fitnesses on the new population
		error = g_evaluate(new_pop_d, pop_size, Block, Grid);
		if (error)
		return true;	

		// Swap the populations
		World* temp_d = old_pop_d;
		old_pop_d     = new_pop_d;
		new_pop_d     = temp_d;

		// Select the new leaders
		sel = g_select_leader(old_pop_d, pop_size, generation_leader,best_leader, Block, Grid);
		if (-1 == sel)
		return true;
		else if (1 == sel) best_generation = i + 1;
		print_status(generation_leader, best_leader, i + 1);
	} 
	
	cout << endl << "Best generation found at " << best_generation << " generations" << endl;

	storeBestLeader(best_leader, "/home/TSP/res/bestleader.txt");

	//freeing memory
	delete[] prob_select; delete[] prob_cross; delete[] prob_mutate;
	delete[] cross_loc; delete[] mutate_loc; free_world(best_leader);
	free_world(generation_leader); cudaFree(old_pop_d); cudaFree(cross_loc_d); 
	cudaFree(new_pop_d); cudaFree(prob_select_d); cudaFree(prob_cross_d);
	cudaFree(prob_mutate_d); cudaFree(mutate_loc_d); cudaFree(sel_ix_d);	
	return false;
}


int main()
{
	// GA parameters
	float prob_mutation  = (float)0.15; // The probability of a mutation
	float prob_crossover = (float)0.8;  // The probability of a crossover
	int world_seed       = 12438955;    // Seed for initial city selection
	int ga_seed          = 87651111;    // Seed for all other random numbers
	
	// The test cases
	int iterations          = 1;  // Number of full runs
	const int num_cases     = 1; // How many trials to test
	int cases[num_cases][2] =     // pop_size, max_gen
	{
		{100, 10000}
	};
	for (int i=0; i<num_cases; i++)
	{
		int pop_size   = cases[i][0];
		int max_gen    = cases[i][1];
		string filename = "/home/TSP/TSPlib/kroA100.tsp";
		World* world = new World[sizeof(World)];
		make_world(world, filename, world_seed);
		cout << "GPU Version - START" << endl;
		auto start = std::chrono::steady_clock::now();
		for (int j=0; j<iterations; j++)
		{
			cudaDeviceReset(); 
			if(g_execute(prob_mutation, prob_crossover, pop_size, max_gen,world, ga_seed))
				{
					cout<<"GPU Related error - Could be an issue if GPU is being used by others"<<endl
						<<"Please try running again when there is memory free + GPU is free"	<<endl;
				}

		}
		auto end = std::chrono::steady_clock::now();
		auto time_used1 = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
		cout<<"use"<<time_used1.count()<<"s"<<endl;
		cudaDeviceReset();

		cout << "GPU Version - END" << endl;

		free_world(world);
	}
	
	return 0;
}
