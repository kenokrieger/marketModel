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

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "cudamacro.h"
#include "device_launch_parameters.h"

#include "ProgressBar.h"
#include "traders.cuh"

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
    int blocks = (grid_height * grid_width / 2 + threads - 1) / threads;

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<<<blocks, threads>>>(true, d_black_tiles, d_white_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width/2);

    CHECK_CURAND(curandGenerateUniform(rng, random_values, grid_height * grid_width / 2));
    update_agents<<<blocks, threads>>>(false, d_white_tiles, d_black_tiles, random_values, d_global_market, alpha, beta, j, grid_height, grid_width/2);
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
