class IsingModel {
    __global__ void init_agents(signed char* agents,
                                const float* __restrict__ random_values,
                                const long long grid_height,
                                const long long grid_width) {
        // iterate over all agents in parallel and assign each of them
        // a strategy of either +1 or -1
        const long long  thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

        // check for out of bound access
        if (thread_id >= pow(grid_width, grid_heigth)) return;

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
            horizontal_neighbor_col = (row % 2) ? right_neighbor : left_neighbor;
        } else {
            horizontal_neighbor_col = (row % 2) ? left_neighbor : right_neighbor;
        }
        // Compute sum of nearest neighbor spins:
        // Multiply the row with the grid-width to contain
        // the actual index in the array
        float neighbor_coupling = j * (
                checkerboard_agents[upper_neighbor_row * grid_width + col]
                + checkerboard_agents[lower_neighbor_row * grid_width + col]
                + checkerboard_agents[row * grid_width + col]
                + checkerboard_agents[row * grid_width + horizontal_neighbor_col]
        );

        signed char old_strategy = agents[row * grid_width + col];
        double market_coupling = -alpha / pow(grid_width, 2) * abs(global_market);
        double hamiltonian = neighbor_coupling + market_coupling * old_strategy;
        // Determine whether to flip spin
        float probability = 1 / (1 + exp(-2.0 * beta * field));
        signed int new_strategy = random_values[row * grid_length + col] < probability ? -1 : 1;
        agents[row * grid_length + j] = new_strategy
    }

};
