#include <chrono>
#include <string>

#define timer std::chrono::high_resolution_clock

class ProgressBar {
    int total_iterations;
    int current_iteration = 0;
    unsigned long long number_of_agents;
    timer::time_point progress_start;
    timer::time_point last_call;

    float calculate_eta();
    double calculate_flips_per_ns();
    static std::string convert_to_time(float time_duration);
    void show();

public:
    explicit ProgressBar(int max_iterations, int grid_height, int grid_width);

    void set_start(timer::time_point now = timer::now()) {progress_start = now;};
    void start();
    void next();
    void end();
};
