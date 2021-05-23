#include <chrono>
#include <string>

#define timer std::chrono::high_resolution_clock

class ProgressBar {
    int total_iterations;
    int current_iteration = 0;
    timer::time_point progress_start;
    timer::time_point pause_start;
    float pause_duration = 0.0f;

    
    float calculate_eta();
    static std::string convert_to_time(float time_duration);
    void show();

public:
    explicit ProgressBar(int max_iterations);

    void set_start(timer::time_point now = timer::now()) {progress_start = now;};
    void start();
    void next();
    void pause();
    void resume();
    void end();
};
