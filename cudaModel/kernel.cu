#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include<conio.h>
#include <windows.h>

#include <sys/types.h>
#include <sys/stat.h>

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


// Default parameters
int device_id = 0;
int threads = 1024;
const long long grid_height = 1024;
const long long grid_width = 1024;
int total_updates = 0;
unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
// the rng offset can be used to revert the random number generator to a specific
// state of a simulation. It is equal to the total number of random numbers
// generated. Meaning the following equation holds for this specific case:
// rng_offset = total_updates * grid_width * grid_height
long long rng_offset = 0;
float alpha = 0.0f;
float j = 1.0f;
float beta = 1 / 1.5f;

signed char *d_black_tiles, *d_white_tiles;
float *random_values;
curandGenerator_t rng;
signed char *h_black_tiles, *h_white_tiles;

bool VISUALISE = false;
bool SHOW_RENDER_PROCESS = false;

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

// Write lattice configuration to file
void write_lattice(signed char *h_black_tiles, signed char *h_white_tiles, std::string filename, long long grid_width, long long grid_height)
{
    CHECK_CUDA(cudaMemcpy(h_global_market, d_global_market, sizeof(*d_global_market), cudaMemcpyDeviceToHost));

    ProgressBar progress_bar = ProgressBar(grid_width);
    progress_bar.start();
    std::ofstream f;
    progress_bar.start();

    f.open(filename);
    if (!f.is_open())
    {
        printf("Could not write to file!\n");
        return;
    }

    f << '#' << "grid = " << grid_width << 'x' << grid_height << std::endl;
    f << '#' << "beta = " << beta << std::endl;
    f << '#' << "alpha = " << alpha << std::endl;
    f << '#' << "j = " << j << std::endl;
    f << '#' << "market = " << h_global_market[0] << std::endl;
    f << '#' << "seed = " << seed << std::endl;
    f << '#' << "total updates = " << total_updates << std::endl;

    for (int row = 0; row < grid_width; row++)
    {
        progress_bar.next();
        for (int col = 0; col < grid_height; col++)
        {
            if (row % 2 == col % 2)
            {
                f << (int)h_black_tiles[row * grid_width / 2 + col / 2] << " ";
            }
            else
            {
                f << (int)h_white_tiles[row * grid_width / 2 + col / 2] << " ";
            }
          }
          f << std::endl;
    }
    f.close();
    progress_bar.end();
}

void update(signed char *d_black_tiles, signed char *d_white_tiles,
            float* random_values,
            curandGenerator_t rng,
            int *d_global_market,
            float alpha, float beta, float j,
            long long grid_height, long long grid_width)
{
    int blocks = (grid_height * grid_width/2 + threads - 1) / threads;

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<true><<<blocks, threads>>>(d_black_tiles, d_white_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width/2);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<false><<<blocks, threads>>>(d_white_tiles, d_black_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width/2);
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
    CHECK_CUDA(cudaDeviceSynchronize());
    timer::time_point start = timer::now();
    update(d_black_tiles, d_white_tiles, random_values, rng, d_global_market, alpha, beta, j, grid_height, grid_width);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer::time_point stop = timer::now();
    total_updates += 1;

    if (kbhit())
    {
        char pressed_key = getch();
        // if the pressed key is "esc"
        if (pressed_key == 27)
        {
            std::string exit_confirmation;
            std::cout << "Exit? ";
            std::cin >> exit_confirmation;
            if (exit_confirmation == "y" || exit_confirmation == "Y")
            {
                std::string filename = "snapshots/snapshot_" + std::to_string(total_updates) + ".dat";
                CHECK_CUDA(cudaMemcpy(h_black_tiles, d_black_tiles, grid_width * grid_height / 2 * sizeof(*d_black_tiles), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_white_tiles, d_white_tiles, grid_width * grid_height / 2 * sizeof(*d_white_tiles), cudaMemcpyDeviceToHost));
                write_lattice(h_black_tiles, h_white_tiles, filename, grid_height, grid_width);
                exit(0);
            }
        }

        if (pressed_key == 's')
        {
            std::string filename = "snapshots/snapshot_" + std::to_string(total_updates) + ".dat";
            CHECK_CUDA(cudaMemcpy(h_black_tiles, d_black_tiles, grid_width * grid_height / 2 * sizeof(*d_black_tiles), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_white_tiles, d_white_tiles, grid_width * grid_height / 2 * sizeof(*d_white_tiles), cudaMemcpyDeviceToHost));
            write_lattice(h_black_tiles, h_white_tiles, filename, grid_height, grid_width);
        }

        // if the pressed key is "spacebar"
        if (pressed_key == 32)
        {
            const char* new_title = VISUALISE ? "Live View (paused)" : "Live View";
            glutSetWindowTitle(new_title);
            VISUALISE = !VISUALISE;
            SHOW_RENDER_PROCESS = VISUALISE;
        }

        if (pressed_key == 'c')
        {
            std::string val;

            std::cout << "New value for alpha?: " << std::flush;
            std::getline(std::cin, val);
            alpha = val != "" ? std::stof(val): alpha;

            std::cout << "New value for beta?: " << std::flush;
            std::getline(std::cin, val);
            beta = val != "" ? std::stof(val): beta;

            std::cout << "New value for j?: " << std::flush;
            std::getline(std::cin, val);
            j = val != "" ? std::stof(val): j;
            printf("alpha = %f\n", alpha);
            printf("beta = %f\n", beta);
            printf("j = %f\n", j);
        }

        if (pressed_key == 'i')
        {
            CHECK_CUDA(cudaMemcpy(h_global_market, d_global_market, sizeof(*d_global_market), cudaMemcpyDeviceToHost));
            double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            double spin_updates_per_nanosecond = grid_width * grid_height / duration * 1e-3;
            printf("----------------------------------------------\n");
            printf("Current iteration: %d\n", total_updates);
            printf("grid = %lld x %lld\n", grid_width, grid_height);
            printf("alpha = %f\n", alpha);
            printf("beta = %f\n", beta);
            printf("j = %f\n", j);
            printf("MARKET = %d\n", h_global_market[0]);
            printf("Updates/ns = %f\n", spin_updates_per_nanosecond);
        }

        if (pressed_key == 'p')
        {
            std::string place_holder;
            std::cout << "Resume? ";
            std::getline(std::cin, place_holder);
            printf("Resuming\n");
        }
    }
    if (!VISUALISE) return;

    CHECK_CUDA(cudaMemcpy(h_black_tiles, d_black_tiles, grid_height * grid_width / 2 * sizeof(*d_black_tiles), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_white_tiles, d_white_tiles, grid_height * grid_width / 2 * sizeof(*d_white_tiles), cudaMemcpyDeviceToHost));

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
            float ypos = 1000 - row / (double)grid_width * 1000;

            glBegin(GL_POLYGON);

            glVertex2f(xpos, ypos);
            glVertex2f(xpos, ypos + size);
            glVertex2f(xpos + size, ypos + size);
            glVertex2f(xpos + size, ypos);

            glEnd();
        }
        if (SHOW_RENDER_PROCESS) glutSwapBuffers();
    }
    glutSwapBuffers();
    SHOW_RENDER_PROCESS = false;
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
    CHECK_CURAND(curandSetGeneratorOffset(rng, rng_offset));

    // allocate memory for the arrays
    CHECK_CUDA(cudaMalloc(&d_white_tiles, grid_height * grid_width/2 * sizeof(*d_white_tiles)))
    CHECK_CUDA(cudaMalloc(&d_black_tiles, grid_height * grid_width/2 * sizeof(*d_black_tiles)));
    CHECK_CUDA(cudaMalloc(&random_values, grid_height * grid_width / 2 * sizeof(*random_values)));
    CHECK_CUDA(cudaMalloc(&d_global_market, sizeof(*d_global_market)));
    h_black_tiles = (signed char*)malloc(grid_height * grid_width / 2 * sizeof(*h_black_tiles));
    h_white_tiles = (signed char*)malloc(grid_height * grid_width / 2 * sizeof(*h_white_tiles));
    h_global_market = (int*)malloc(sizeof(*h_global_market));

    int blocks = (grid_height * grid_width/2 + threads - 1) / threads;
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, threads>>>(d_black_tiles, random_values, grid_height, grid_width / 2);
    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    init_agents<<<blocks, threads>>>(d_white_tiles, random_values, grid_height, grid_width / 2);

    // Synchronize operations on the GPU with CPU
    CHECK_CUDA(cudaDeviceSynchronize());

    // create directory for saves if not already exists
    struct stat st = {0};

    // create directory snapshots if it does not exist already
    if (stat("snapshots", &st) == -1) {
        CreateDirectoryA("snapshots", NULL);
    }

    glutInit(&argc, argv);
  	glutInitDisplayMode(GLUT_RGB);

  	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - 1000) / 2, 0);
  	glutInitWindowSize(1000, 1000);

  	glutCreateWindow("Live View (paused)");

    glClearColor(0.1f, 0.35f, 0.71f, 1.0f);
  	glutDisplayFunc(render);
    glutIdleFunc(render);
  	glutReshapeFunc(reshape);

  	glutMainLoop();

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
