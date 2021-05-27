__global__ void init_agents(signed char* agents,
                            const float* __restrict__ random_values,
                            const long long grid_height,
                            const long long grid_width);

__global__ void update_agents(bool is_black,
                              signed char* agents,
                              const signed char* __restrict__ checkerboard_agents,
                              const float* __restrict__ random_values,
                              int *d_global_market,
                              const float alpha,
                              const float beta,
                              const float j,
                              const long long grid_height,
                              const long long grid_width);
