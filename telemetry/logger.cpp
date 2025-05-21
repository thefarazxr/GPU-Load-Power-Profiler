#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

// Returns a timestamped CSV path like: data/run_20260511_153042.csv
std::string make_run_path(const std::string& output_dir) {
    std::filesystem::create_directories(output_dir);
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream ss;
    ss << output_dir << "/run_"
       << std::put_time(&tm, "%Y%m%d_%H%M%S")
       << ".csv";
    return ss.str();
}
