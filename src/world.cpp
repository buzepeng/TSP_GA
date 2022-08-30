// Native Includes
// #include <set>
// #include <tuple>
#include <random>
#include <functional>
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include <ctype.h>

// Program Includes
#include "world.h"
#include "common.h"
#include "ga_gpu.h"

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

//CPU

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

void free_population(World* pop, int pop_size)
{
	for (int i=0; i<pop_size; i++)
		delete[] pop[i].cities;
	delete[] pop;
}

//GPU

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
