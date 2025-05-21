#pragma once
#include <cstdint>
#include <string>

struct Config {
    uint32_t block_size    = 256;
    uint32_t thread_count  = 1048576;  // total threads; grid = thread_count / block_size
    uint32_t memory_stride = 1;
    uint32_t duration_sec  = 0;   // 0 = run until Ctrl+C
    uint32_t poll_interval_ms = 200;
    bool     enable_cuda   = true;
    bool     enable_vulkan = true;
    std::string output_dir = "data";
};

Config parse_args(int argc, char** argv);
void   print_config(const Config& cfg);
