from os.path import join

import numpy as np

from timeit import default_timer as timer

from progressbar import ProgressBar

# parameter
total_updates: int = 10
grid_size: int = 2048
j: float = 1.0  # neighbor coupling
alpha: float = 4.0  # market coupling
beta: float = 0.6  # inverse randomness


def init_agents(grid_height, grid_width):
    """
    Initialise an array of agents with strategies +1 or -1.

    """
    agents: np.ndarray = np.random.randint(2, size=(grid_height * grid_width))
    agents[np.where(agents == 0)] = -1
    return agents


def update_agents(is_black, agents, checkerboard_agents, grid_width,
                  grid_height,random_values, alpha, beta, j):
    """
    Update the agents with metropolis dynamics and Heatbath updates.

    :param agents: The agents to update.
    :type agents: np.ndarray
    :param checkerboard_agents: The neighbors of the agents to update.
    :type checkerboard_agents: np.ndarray
    :param random_values: Random values between 0 and 1 generated beforehand.
    :type random_values: np.ndarray
    :param alpha: The market coupling.
    :type alpha: float
    :param beta: The inverse temperature
    :type beta: float
    :param j: The neighbor coupling.
    :type j: float
    """
    global GLOBAL_MARKET
    for row in range(grid_height):
        for col in range(grid_width):
            lower_neighbor_row: int = row + 1 if row + 1 < grid_height else 0
            upper_neighbor_row: int = row - 1 if row - 1 >= 0 else grid_height - 1
            right_neighbor_col: int = col + 1 if col + 1 < grid_width else 0
            left_neighbor_col: int = col - 1 if col - 1 >= 0 else grid_width - 1

            if is_black:
                horizontal_neighbor_col = right_neighbor_col if row % 2 else left_neighbor_col
            else:
                horizontal_neighbor_col = left_neighbor_col if row % 2 else right_neighbor_col

            neighbor_coupling: float = 0
            neighbor_coupling += checkerboard_agents[upper_neighbor_row * grid_width + col]
            neighbor_coupling += checkerboard_agents[lower_neighbor_row * grid_width + col]
            neighbor_coupling += checkerboard_agents[row * grid_width + col]
            neighbor_coupling += checkerboard_agents[row * grid_width + horizontal_neighbor_col]
            neighbor_coupling *= j
            market_coupling: float = - alpha / (grid_width * grid_height) * abs(GLOBAL_MARKET)
            old_strategy: int = agents[row * grid_width + col]
            field: float = neighbor_coupling + market_coupling * old_strategy
            probability: float = 1 / (1 + np.exp(-2 * beta * field))
            new_strategy = 1 if random_values[row * grid_width + col] < probability else -1
            agents[row * grid_width + col] = new_strategy

            if new_strategy != old_strategy:
                GLOBAL_MARKET -= 2 * old_strategy


def update(black_tiles, white_tiles, grid_width, grid_height, alpha, beta, j):
    """
    Update the whole grid by updating white and black tiles.

    """
    random_values = np.random.rand(black_tiles.shape[0])
    update_agents(True, black_tiles, white_tiles, grid_width, grid_height,
                  random_values, alpha, beta, j)
    random_values = np.random.rand(white_tiles.shape[0])
    update_agents(False, white_tiles, black_tiles, grid_width, grid_height,
                  random_values, alpha, beta, j)


if __name__ == "__main__":
    GLOBAL_MARKET = 0
    black_tiles = init_agents(grid_size, grid_size // 2)
    white_tiles = init_agents(grid_size, grid_size // 2)
    bar = ProgressBar(total_updates)
    start = timer()
    bar.start()
    for iteration in range(total_updates):
        update(black_tiles, white_tiles, grid_size, grid_size // 2, alpha, beta, j)
        bar.next()
    bar.end()
    end = timer()

    total_duration = end - start
    print("Total duration: ", total_duration)
    updates_per_nanosecond = total_updates * grid_size ** 2/ total_duration * 1e-9
    print("Updates/ns", updates_per_nanosecond)
