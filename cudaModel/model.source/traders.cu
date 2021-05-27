/*
This is a comment
*/
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


__global__ void update_agents(bool is_black,
                              signed char* agents,
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

    // determine nearest neighbours on the opposite grid
    int lower_neighbour_row = (row + 1 < grid_height) ? row + 1 : 0;
    int upper_neighbour_row = (row - 1 >= 0) ? row - 1: grid_height - 1;
    int right_neighbour_col = (col + 1 < grid_width) ? col + 1 : 0;
    int left_neighbour_col = (col - 1 >= 0) ? col - 1: grid_width - 1;

    // Select off-column index based on color and row index parity:
    // One of the neighbours will always have the exact same index
    // as the agents where as the remaining one will either have an
    // index differing by +1 or -1 depending on the position of the
    // agent on the grid
    int horizontal_neighbour_col;
    if (is_black) {
        horizontal_neighbour_col = (row % 2) ? right_neighbour_col : left_neighbour_col;
    } else {
        horizontal_neighbour_col = (row % 2) ? left_neighbour_col : right_neighbour_col;
    }
    // Compute sum of nearest neighbour spins:
    // Multiply the row with the grid-width to receive
    // the actual index in the array
    float neighbour_coupling = j * (
            checkerboard_agents[upper_neighbour_row * grid_width + col]
          + checkerboard_agents[lower_neighbour_row * grid_width + col]
          + checkerboard_agents[row * grid_width + col]
          + checkerboard_agents[row * grid_width + horizontal_neighbour_col]
          );

    signed char old_strategy = agents[row * grid_width + col];
    double market_coupling = -alpha / (grid_width * grid_height) * abs(d_global_market[0]);
    double field = neighbour_coupling + market_coupling * old_strategy;
    // Determine whether to flip spin
    float probability = 1 / (1 + exp(-2.0 * beta * field));
    signed char new_strategy = random_values[row * grid_width + col] < probability ? 1 : -1;
    agents[row * grid_width + col] = new_strategy;
    __syncthreads();
    // If the strategy was changed remove the old value from the sum and add the new value.
    if (new_strategy != old_strategy)
        d_global_market[0] -= 2 * old_strategy;
}
