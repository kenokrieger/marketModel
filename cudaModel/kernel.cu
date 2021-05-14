#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include<conio.h>

#include <GL/glut.h>
#include <GL/freeglut_std.h>

#include "ProgressBar.h"

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#include "cudamacro.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#define timer std::chrono::high_resolution_clock
#define THREADS 128

// Default parameters
int device_id = 0;
const long long grid_height = 2048;
const long long grid_width = 2048;
int total_updates = 0;
unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
float alpha = 0.0f;
float j = 1.0f;
float beta = 10 / 1.5f;

signed char *black_tiles, *white_tiles;
float *random_values;
curandGenerator_t rng;
signed char *h_black_tiles, *h_white_tiles;

bool VISUALISE = true;

// The global market represents the sum over the strategies of each
// agent. Agents will choose a strategy contrary to the sign of the
// global market.
int *d_global_market;
int *h_global_market;


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
                              int *d_global_market,
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

// Write lattice configuration to file
void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
  signed char *lattice_h, *lattice_b_h, *lattice_w_h;
  ProgressBar progress_bar = ProgressBar(nx);
  lattice_h = (signed char*) malloc(nx * ny * sizeof(*lattice_h));
  lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
  lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));

  CHECK_CUDA(cudaMemcpy(lattice_b_h, lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(lattice_w_h, lattice_b, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost));
  progress_bar.start();
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
  progress_bar.start();
  f.open(filename);
  if (f.is_open()) {
    for (int i = 0; i < nx; i++) {
      progress_bar.next();
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
  progress_bar.end();
}

void update(signed char *black_tiles, signed char *white_tiles,
            float* random_values,
            curandGenerator_t rng,
            int *d_global_market,
            float alpha, float beta, float j,
            long long grid_height, long long grid_width) {
  // Setup CUDA launch configuration
  int blocks = (grid_height * grid_width/2 + THREADS - 1) / THREADS;

  // Update black tiles on "checkerboard"
  CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
  update_agents<true><<<blocks, THREADS>>>(black_tiles, white_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width/2);

  // Update white tiles on "checkerboard"
  CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
  update_agents<false><<<blocks, THREADS>>>(white_tiles, black_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width/2);
}

void reshape(int width, int height)
{
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, 1000, 0, 1000);
	glMatrixMode(GL_MODELVIEW);
}

void render()
{
  update(black_tiles, white_tiles, random_values, rng, d_global_market, alpha, beta, j, grid_height, grid_width);
  total_updates += 1;
  CHECK_CUDA(cudaMemcpy(h_global_market, d_global_market, sizeof(*d_global_market), cudaMemcpyDeviceToHost));
  std::cout << "MARKET = " << h_global_market[0] << std::endl;
  if (kbhit()) {
    char pressed_key = getch();
    // if the pressed key is "esc"
    if (pressed_key == 27) {
      std::string exit_confirmation;
      std::cout << "Exit? ";
      std::cin >> exit_confirmation;
      if (exit_confirmation == "y" || exit_confirmation == "Y")
      {
        write_lattice(black_tiles, white_tiles, "final_configuration.dat", grid_height, grid_width);
        exit(0);
      }
    }

    if (pressed_key == 's')
    {
      write_lattice(black_tiles, white_tiles, "snapshot.dat", grid_height, grid_width);
    }

    // if the pressed key is "spacebar"
    if (pressed_key == 32)
    {
    const char* new_title = VISUALISE ? "Live View (paused)" : "Live View";
    glutSetWindowTitle(new_title);
    VISUALISE = !VISUALISE;
    }
  }
  if (!VISUALISE) return;

  CHECK_CUDA(cudaMemcpy(h_black_tiles, black_tiles, grid_height * grid_width / 2 * sizeof(*black_tiles), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_white_tiles, white_tiles, grid_height * grid_width / 2 * sizeof(*white_tiles), cudaMemcpyDeviceToHost));

  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();

  float size = 1 / (double)grid_width * 1000;

  for (int row = 0; row < grid_width; row++)
  {
    for (int col = 0; col < grid_height; col++)
    {
      if (row % 2 == col % 2)
      {
        if (h_black_tiles[row * grid_width / 2 + col / 2] == -1) continue;
      }
      else
      {
        if (h_white_tiles[row * grid_width / 2 + col / 2] == -1) continue;
      }
      float xpos = col / (double)grid_width * 1000;
      float ypos = row / (double)grid_width * 1000;

      glBegin(GL_POLYGON);

      glVertex2f(xpos, ypos);
      glVertex2f(xpos, ypos + size);
      glVertex2f(xpos + size, ypos + size);
      glVertex2f(xpos + size, ypos);

      glEnd();
    }
  }
  glutSwapBuffers();
}

int main(int argc, char** argv) {
    // searches for available cuda devices
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    printf("Found %d cuda device(s)\n", device_count);

    // finds and sets the specified cuda device
    findCudaDevice(device_id);

    // Finds and prints the devices name and computing power
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_id));
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    // Set up cuRAND generator
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&white_tiles, grid_height * grid_width/2 * sizeof(*white_tiles)))
    CHECK_CUDA(cudaMalloc(&black_tiles, grid_height * grid_width/2 * sizeof(*black_tiles)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_height * grid_width / 2 * sizeof(*random_values)));
    CHECK_CUDA(cudaMalloc(&d_global_market, sizeof(*d_global_market)));
    h_black_tiles = (signed char*)malloc(grid_height * grid_width / 2 * sizeof(*h_black_tiles));
    h_white_tiles = (signed char*)malloc(grid_height * grid_width / 2 * sizeof(*h_white_tiles));
    h_global_market = (int*)malloc(sizeof(*h_global_market));

    int blocks = (grid_height * grid_width/2 + THREADS - 1) / THREADS;
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, THREADS>>>(black_tiles, random_values, grid_height, grid_width / 2);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, THREADS>>>(white_tiles, random_values, grid_height, grid_width / 2);

    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());

    glutInit(&argc, argv);
  	glutInitDisplayMode(GLUT_RGB);

  	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - 1000) / 2, 0);
  	glutInitWindowSize(1000, 1000);

  	glutCreateWindow("Live View");

    glClearColor(0.1f, 0.35f, 0.71f, 1.0f);
  	glutDisplayFunc(render);
    glutIdleFunc(render);
  	glutReshapeFunc(reshape);

  	glutMainLoop();

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
