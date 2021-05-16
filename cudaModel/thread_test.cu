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
#define TRIAL_ITERATIONS 100
#define MAX_THREADS 1024


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
__global__ void update_agents(signed char* agents, int *d_global_market,
                              const signed char* __restrict__ checkerboard_agents,
                              const float* __restrict__ random_values,
                              const float alpha,
                              const float beta,
                              const float j,
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
    float neighbor_coupling = j * (
            checkerboard_agents[upper_neighbor_row * grid_width + col]
          + checkerboard_agents[lower_neighbor_row * grid_width + col]
          + checkerboard_agents[row * grid_width + col]
          + checkerboard_agents[row * grid_width + horizontal_neighbor_col]
          );

    signed char old_strategy = agents[row * grid_width + col];
    double market_coupling = -alpha / (grid_width * grid_height) * abs(d_global_market[0]);
    double field = neighbor_coupling + market_coupling * old_strategy;
    // Determine whether to flip spin
    float probability = 1 / (1 + exp(-2.0 * beta * field));
    signed char new_strategy = random_values[row * grid_width + col] < probability ? 1 : -1;
    agents[row * grid_width + col] = new_strategy;
    __syncthreads();
    // If the strategy was changed remove the old value from the sum and add the new value.
    if (new_strategy != old_strategy)
        d_global_market[0] -= 2 * old_strategy;
}

void update(int threads, signed char *d_black_tiles, signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            int *d_global_market,
            float alpha, float beta, float j,
            long long grid_height, long long grid_width)
{
    // Setup CUDA launch configuration
    int blocks = (grid_height * grid_width / 2 + threads - 1) / threads;

    // Update black tiles on "checkerboard"
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<true><<<blocks, threads>>>(d_black_tiles, d_global_market, d_white_tiles, random_values, alpha, beta, j, grid_height, grid_width/2);

    // Update white tiles on "checkerboard"
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<false><<<blocks, threads>>>(d_white_tiles, d_global_market, d_black_tiles, random_values, alpha, beta, j, grid_height, grid_width/2);
}

double time_updates_per_nano_second(int threads, unsigned int seed, float alpha, float beta, float j, long long grid_height, long long grid_width, int* d_global_market)
{
    signed char *d_black_tiles, *d_white_tiles;
    float *random_values;
    curandGenerator_t rng;
    // Set up cuRAND generator
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&d_white_tiles, grid_height * grid_width / 2 * sizeof(*d_white_tiles)))
    CHECK_CUDA(cudaMalloc(&d_black_tiles, grid_height * grid_width / 2 * sizeof(*d_black_tiles)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_height * grid_width / 2 * sizeof(*random_values)));
    CHECK_CUDA(cudaMalloc(&d_global_market, sizeof(*d_global_market)));

    // initialise agents
    int blocks = (grid_height * grid_width/2 + threads - 1) / threads;
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, threads>>>(d_black_tiles, random_values, grid_height, grid_width / 2);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, threads>>>(d_white_tiles, random_values, grid_height, grid_width / 2);

    timer::time_point start = timer::now();
    for (int iteration = 0; iteration < TRIAL_ITERATIONS; iteration++)
    {
        update(threads, d_black_tiles, d_white_tiles, random_values, rng,
               d_global_market, alpha, beta, j,
               grid_height, grid_width);
    }
    timer::time_point stop = timer::now();

    curandDestroyGenerator(rng);
    cudaFree(d_black_tiles);
    cudaFree(d_white_tiles);
    cudaFree(random_values);

    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    double spin_updates_per_nanosecond = TRIAL_ITERATIONS * grid_width * grid_height / duration * 1e-3;
    return spin_updates_per_nanosecond;
}

int main(int argc, char** argv)
{
    // Default parameters
    long long grid_height = 128;
    long long grid_width = 128;
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    float alpha = 1.0f;
    float j = 1.0f;
    float beta = 1 / 1.5f;
    // The global market represents the sum over the strategies of each
    // agent. Agents will choose a strategy contrary to the sign of the
    // global market.
    int *d_global_market;
    CHECK_CUDA(cudaMalloc(&d_global_market, sizeof(*d_global_market)));
    // create directory for saves if not already exists
    struct stat st = {0};

    if (stat("logs", &st) == -1) {
        CreateDirectoryA("logs", NULL);
    }

    double spin_updates_per_nanosecond;
    int blocks;
    std::ofstream file;
    std::string filename;

    //warm up
    time_updates_per_nano_second(1024, 1, 2.0, 2.0, 2.0, 2000, 2000, d_global_market);


    for (int trial = 1; trial < 20; trial++)
    {
        CHECK_CUDA(cudaDeviceSynchronize());
        grid_width *= trial;
        grid_height *= trial;
        printf("\nSpeed test with grid = %lld x %lld \n", grid_width, grid_height);
        filename = "logs/grid = " + std::to_string(grid_height) + 'x' + std::to_string(grid_width) + ".dat";
        file.open(filename);
        if (!file.is_open())
        {
            printf("Could not write to file!\n");
            return -1;
        }

        file << '#' << "grid = " << grid_width << 'x' << grid_height << std::endl;
        file << '#' << "beta = " << beta << std::endl;
        file << '#' << "alpha = " << alpha << std::endl;
        file << '#' << "j = " << j << std::endl;
        file << '#' << "number_of_threads | spin_updates_per_nanosecond" << std::endl;

        for (int number_of_threads = 1; number_of_threads < MAX_THREADS + 1; number_of_threads++)
        {
            blocks = (grid_height * grid_width / 2 + number_of_threads - 1) / number_of_threads;
            spin_updates_per_nanosecond = time_updates_per_nano_second(number_of_threads, seed, alpha, beta, j, grid_height, grid_width, d_global_market);
            file << number_of_threads << ' ' << spin_updates_per_nanosecond << std::endl;
            std::cout << '\r' << "blocks: " << blocks << " | threads: " << number_of_threads << " | updates/ns: " << spin_updates_per_nanosecond;
            std::cout << std::string(10, ' ') << std::string(10, '\b') << std::flush;
            CHECK_CUDA(cudaDeviceSynchronize());

        }
        file.close();
    }
    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
