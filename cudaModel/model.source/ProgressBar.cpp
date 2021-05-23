#include <iostream>
#include <iomanip>
#include <cmath>

#include "ProgressBar.h"

using namespace std;

#define timer std::chrono::high_resolution_clock


ProgressBar::ProgressBar(int max_iterations) {
    total_iterations = max_iterations;
    progress_start = timer::now();
}

void ProgressBar::show() {
    const int barlength = 30;
    float progress = (float) current_iteration / (float) total_iterations;
    int progress_count = progress * barlength;
    const string out = "[" + string(progress_count, '#') + string(barlength - progress_count, '.') + "]";

    cout << '\r';
    cout << out << '[' << setw(log10(total_iterations) + 2) << current_iteration << '/' << total_iterations;
    cout << " eta " << convert_to_time(calculate_eta()) << "]";
    // write some spaces at the end to overwrite old brackets
    // and return cursor to the end of the output
    cout << string(10, ' ') << string(10, '\b') << flush;
}

float ProgressBar::calculate_eta() {
    float eta;
    timer::time_point now = timer::now();
    auto time_since_start = chrono::duration_cast<chrono::microseconds>(now - progress_start);
    if (current_iteration == 0)
      eta = -1;
    else
      // The estimated time is calculated as remaining_computations * mean_time_per_computation
      eta = (total_iterations / (float)current_iteration - 1) * (time_since_start.count() - pause_duration) * 1e-6;
    return eta;
}

string ProgressBar::convert_to_time(float time_duration) {
    string duration;
    int hours = time_duration / 3600;
    int minutes = time_duration / 60 - hours * 60;
    float seconds = time_duration - hours * 3600 - minutes * 60;

    if (hours != 0)
        duration += to_string(hours) + "h ";
    if (minutes != 0)
        duration += to_string(minutes) + "m ";
    if (seconds != 0.0f)
        duration += to_string(seconds) + "s ";
    else
        duration += "done";
    return duration;
}

void ProgressBar::start() {
    progress_start = std::chrono::high_resolution_clock::now();
    show();
}

void ProgressBar::next() {
    current_iteration += 1;
    show();
}

void ProgressBar::pause() {
    pause_start = std::chrono::high_resolution_clock::now();
}

void ProgressBar::resume() {
    timer::time_point now = timer::now();
    auto time_since_pause = chrono::duration_cast<chrono::microseconds>(now - pause_start);
    pause_duration += time_since_pause.count();
}

void ProgressBar::end() {
    timer::time_point now = timer::now();
    auto time_since_start = chrono::duration_cast<chrono::microseconds>(now - progress_start);
    float total_duration = time_since_start.count() * 1e-6;
    cout << "\nProcess finished in " << total_duration << " s\n";
    cout << "at an average of " << total_iterations / total_duration << " per second" << endl;
}
