/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#include "cudamacro.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#define TCRIT 1.0f
#define THREADS  128

__device__ int GLOBAL_MARKET = 0;

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
                              const float alpha,
                              const float beta,
                              const float j,
                              const long long grid_height,
                              const long long grid_width) {
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
    // Multiply the row with the grid-width to contain
    // the actual index in the array
    float neighbor_coupling = j * (
            checkerboard_agents[upper_neighbor_row * grid_width + col]
          + checkerboard_agents[lower_neighbor_row * grid_width + col]
          + checkerboard_agents[row * grid_width + col]
          + checkerboard_agents[row * grid_width + horizontal_neighbor_col]);

    signed char old_strategy = agents[row * grid_width + col];
    double market_coupling = -alpha / pow(grid_width, 2) * abs(GLOBAL_MARKET);
    double field = neighbor_coupling + market_coupling * old_strategy;
    // Determine whether to flip spin
    float probability = 1 / (1 + exp(-2.0 * beta * field));
    signed char new_strategy = random_values[row * grid_width + col] < probability ? -1 : 1;
    if (new_strategy != old_strategy)
        GLOBAL_MARKET -= 2 * old_strategy;
    agents[row * grid_width + col] = new_strategy;
}

// Write lattice configuration to file
void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
  printf("Writing lattice to %s...\n", filename.c_str());
  signed char *lattice_h, *lattice_b_h, *lattice_w_h;
  lattice_h = (signed char*) malloc(nx * ny * sizeof(*lattice_h));
  lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
  lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));

  CHECK_CUDA(cudaMemcpy(lattice_b_h, lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(lattice_w_h, lattice_b, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost));

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny/2; j++) {
      if (i % 2) {
        lattice_h[i*ny + 2*j+1] = lattice_b_h[i*ny/2 + j];
        lattice_h[i*ny + 2*j] = lattice_w_h[i*ny/2 + j];
      } else {
        lattice_h[i*ny + 2*j] = lattice_b_h[i*ny/2 + j];
        lattice_h[i*ny + 2*j+1] = lattice_w_h[i*ny/2 + j];
      }
    }
  }

  std::ofstream f;
  f.open(filename);
  if (f.is_open()) {
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
         f << (int)lattice_h[i * ny + j] << " ";
      }
      f << std::endl;
    }
  }
  f.close();

  free(lattice_h);
  free(lattice_b_h);
  free(lattice_w_h);
}

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, float alpha,
            float inv_temp, float j, long long nx, long long ny) {

  // Setup CUDA launch configuration
  int blocks = (nx * ny/2 + THREADS - 1) / THREADS;

  // Update black
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
  update_agents<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals, alpha, inv_temp, j, nx, ny/2);

  // Update white
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
  update_agents<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals, alpha, inv_temp, j, nx, ny/2);
}

int main() {
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    // command-line input is given as 0, 0
    int dev = findCudaDevice(0, 0);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    // Defaults
    long long nx = 2048;
    long long ny = 2048;
    int nwarmup = 100;
    int niters = 1000;
    bool write = false;
    unsigned long long seed = 1234ULL;
    write = false;

  float inv_temp = 1 / 1.5f;

  // Setup cuRAND generator
  curandGenerator_t rng;
  CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
  float *randvals;
  CHECK_CUDA(cudaMalloc(&randvals, nx * ny/2 * sizeof(*randvals)));

  // Setup black and white lattice arrays on device
  signed char *lattice_b, *lattice_w;
  CHECK_CUDA(cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b)));
  CHECK_CUDA(cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w)));

  int blocks = (nx * ny/2 + THREADS - 1) / THREADS;
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
  init_agents<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/2);
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
  init_agents<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/2);

  float alpha = 3.0f;
  float j = 1.0f;
  // Warmup iterations
  printf("Starting warmup...\n");
  for (int i = 0; i < nwarmup; i++) {
    update(lattice_b, lattice_w, randvals, rng, alpha, inv_temp, j, nx, ny);
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    update(lattice_b, lattice_w, randvals, rng, alpha, inv_temp, j, nx, ny);
    if (i % 1000 == 0) printf("Completed %d/%d iterations...\n", i+1, niters);
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();

  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  printf("REPORT:\n");
  printf("\tnGPUs: %d\n", 1);
  printf("\ttemperature: %f * %f\n", alpha, TCRIT);
  printf("\tseed: %llu\n", seed);
  printf("\twarmup iterations: %d\n", nwarmup);
  printf("\ttrial iterations: %d\n", niters);
  printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
  printf("\telapsed time: %f sec\n", duration * 1e-6);
  printf("\tupdates per ns: %f\n", (double) (nx * ny) * niters / duration * 1e-3);

  // Reduce
  double* devsum;
  int nchunks = (nx * ny/2 + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMalloc(&devsum, 2 * nchunks * sizeof(*devsum)));
  size_t cub_workspace_bytes = 0;
  void* workspace = NULL;
  CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, lattice_b, devsum, CUB_CHUNK_SIZE));
  CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));
  for (int i = 0; i < nchunks; i++) {
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_b[i*CUB_CHUNK_SIZE], devsum + 2*i,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE)));
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_w[i*CUB_CHUNK_SIZE], devsum + 2*i + 1,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE)));
  }

  double* hostsum;
  hostsum = (double*)malloc(2 * nchunks * sizeof(*hostsum));
  CHECK_CUDA(cudaMemcpy(hostsum, devsum, 2 * nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost));
  double fullsum = 0.0;
  for (int i = 0; i < 2 * nchunks; i++) {
    fullsum += hostsum[i];
  }
  std::cout << "\taverage magnetism (absolute): " << abs(fullsum / (nx * ny)) << std::endl;

  if (write) write_lattice(lattice_b, lattice_w, "final.txt", nx, ny);

  return 0;
}
