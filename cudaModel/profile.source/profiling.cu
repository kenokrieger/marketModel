#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include<conio.h>
#include <windows.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <cuda_fp16.h>
#include <curand.h>

#include "cudamacro.h"
#include <cuda_runtime.h>

#define timer std::chrono::high_resolution_clock
#define THREADS 256


__global__ void init_agents(signed char* agents,
                            const float* __restrict__ random_values,
                            const long long grid_height,
                            const long long grid_width) {
    // iterate over all agents in parallel and assign each of them
    // a strategy of either +1 or -1
    const long long  thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    // check for out of bound access
    if (thread_id >= grid_width * grid_height) return;

    // use random number between 0.0 and 1.0 generated beforehand
    float random = random_values[thread_id];
    agents[thread_id] = (random < 0.5f) ? -1 : 1;
}


template<bool is_black>
__global__ void update_agents(signed char* agents,
                              const signed char* __restrict__ checkerboard_agents,
                              const float* __restrict__ random_values,
                              const long long grid_height,
                              const long long grid_width)
{
    const long long thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int row = thread_id / grid_width;
    const int col = thread_id % grid_width;

    // check for out of bound access
    if (row >= grid_height || col >= grid_width) return;

    // determine nearest neighbors on the opposite grid
    int lower_neighbor_row = (row + 1 < grid_height) ? row + 1 : 0;
    int upper_neighbor_row = (row - 1 >= 0) ? row - 1: grid_height - 1;
    int right_neighbor_col = (col + 1 < grid_width) ? col + 1 : 0;
    int left_neighbor_col = (col - 1 >= 0) ? col - 1: grid_width - 1;

    // Select off-column index based on color and row index parity:
    // One of the neighbors will always have the exact same index
    // as the agents where as the remaining one will either have an
    // index differing by +1 or -1 depending on the position of the
    // agent on the grid
    int horizontal_neighbor_col;
    if (is_black) {
        horizontal_neighbor_col = (row % 2) ? right_neighbor_col : left_neighbor_col;
    } else {
        horizontal_neighbor_col = (row % 2) ? left_neighbor_col : right_neighbor_col;
    }
    // Compute sum of nearest neighbor spins:
    // Multiply the row with the grid-width to receive
    // the actual index in the array
    signed char neighbor_coupling =
            checkerboard_agents[upper_neighbor_row * grid_width + col]
          + checkerboard_agents[lower_neighbor_row * grid_width + col]
          + checkerboard_agents[row * grid_width + col]
          + checkerboard_agents[row * grid_width + horizontal_neighbor_col];

    // Determine whether to flip spin
    float probability = 1 / (1 + exp(-2.0 * 0.666f * neighbor_coupling));
    signed char new_strategy = random_values[row * grid_width + col] < probability ? 1 : -1;
    agents[row * grid_width + col] = new_strategy;
}


void update(signed char *d_black_tiles, signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            long long grid_height, long long grid_width)
{
    // Setup CUDA launch configuration
    int blocks = (grid_height * grid_width / 2 + THREADS - 1) / THREADS;

    // Update black tiles on "checkerboard"
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<true><<<blocks, THREADS>>>(d_black_tiles, d_white_tiles, random_values, grid_height, grid_width/2);

    // Update white tiles on "checkerboard"
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<false><<<blocks, THREADS>>>(d_white_tiles, d_black_tiles, random_values, grid_height, grid_width/2);
}


int main(int argc, char** argv)
{
    // Default parameters
    const long long grid_height = 2048;
    const long long grid_width = 2048;
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    float alpha = 1.0f;
    float j = 1.0f;
    float beta = 1 / 1.5f;
    // The global market represents the sum over the strategies of each
    // agent. Agents will choose a strategy contrary to the sign of the
    // global market.
    int *d_global_market;
    signed char *d_black_tiles, *d_white_tiles;
    float *random_values;
    curandGenerator_t rng;

    // searches for available cuda devices
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("Found %d cuda device(s)\n", device_count);

    // Finds and prints the devices name and computing power
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    // Set up cuRAND generator
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&d_white_tiles, grid_height * grid_width/2 * sizeof(*d_white_tiles)))
    CHECK_CUDA(cudaMalloc(&d_black_tiles, grid_height * grid_width/2 * sizeof(*d_black_tiles)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_height * grid_width / 2 * sizeof(*random_values)));
    CHECK_CUDA(cudaMalloc(&d_global_market, sizeof(*d_global_market)));

    int blocks = (grid_height * grid_width/2 + THREADS - 1) / THREADS;
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, THREADS>>>(d_black_tiles, random_values, grid_height, grid_width / 2);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, THREADS>>>(d_white_tiles, random_values, grid_height, grid_width / 2);

    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());

    // test for grid size up to maximum memory
    for (int iteration = 0; iteration < 10000; iteration++)
    {
        cudaSetDevice(0);
        update(d_black_tiles, d_white_tiles, random_values, rng,
               grid_height, grid_width);
        if (iteration % 1000 == 0)
          std::cout << '#';
    }
    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
