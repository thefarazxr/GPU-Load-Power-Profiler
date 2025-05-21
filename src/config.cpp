#include "config.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " [options]\n"
        "  --block-size    N    Threads per CUDA block (default 256)\n"
        "  --threads       N    Total thread count; grid = N / block-size (default 1048576)\n"
        "  --stride        N    Memory access stride in elements (default 1)\n"
        "  --duration      N    Run for N seconds; omit to run until Ctrl+C\n"
        "  --poll          N    Telemetry poll interval ms (default 200)\n"
        "  --no-cuda            Disable CUDA engine\n"
        "  --no-vulkan          Disable Vulkan engine\n"
        "  --output-dir    DIR  Output directory for CSVs (default data/)\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> uint32_t {
            if (i + 1 >= argc) throw std::invalid_argument("missing value");
            return static_cast<uint32_t>(std::stoul(argv[++i]));
        };
        if      (!strcmp(argv[i], "--block-size"))  cfg.block_size       = next();
        else if (!strcmp(argv[i], "--threads"))     cfg.thread_count     = next();
        else if (!strcmp(argv[i], "--stride"))      cfg.memory_stride    = next();
        else if (!strcmp(argv[i], "--duration"))    cfg.duration_sec     = next();
        else if (!strcmp(argv[i], "--poll"))        cfg.poll_interval_ms = next();
        else if (!strcmp(argv[i], "--no-cuda"))     cfg.enable_cuda      = false;
        else if (!strcmp(argv[i], "--no-vulkan"))   cfg.enable_vulkan    = false;
        else if (!strcmp(argv[i], "--output-dir"))  {
            if (i + 1 >= argc) throw std::invalid_argument("missing value");
            cfg.output_dir = argv[++i];
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); exit(0);
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            usage(argv[0]); exit(1);
        }
    }
    return cfg;
}

void print_config(const Config& cfg) {
    std::cout << "Config:\n"
              << "  block_size      = " << cfg.block_size       << "\n"
              << "  thread_count    = " << cfg.thread_count     << "\n"
              << "  memory_stride   = " << cfg.memory_stride    << "\n"
              << "  duration_sec    = " << (cfg.duration_sec ? std::to_string(cfg.duration_sec) + "s" : "until Ctrl+C") << "\n"
              << "  poll_interval   = " << cfg.poll_interval_ms << " ms\n"
              << "  enable_cuda     = " << cfg.enable_cuda      << "\n"
              << "  enable_vulkan   = " << cfg.enable_vulkan    << "\n"
              << "  output_dir      = " << cfg.output_dir       << "\n";
}
