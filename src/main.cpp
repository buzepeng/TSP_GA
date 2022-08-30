#include <ctime>
#include <iostream>
#include <string.h>
#include <sstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <chrono>
#include "common.h"
#include "ga_gpu.h"

using namespace std;

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
