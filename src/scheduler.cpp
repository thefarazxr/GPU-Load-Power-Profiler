#include "config.h"
#include "nvml_monitor.h"
#include "../include/heavy_hash.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

void run_vulkan_workload(uint32_t n, uint32_t iters, uint32_t stride,
                         const std::string& spv_path, uint32_t buf_mb);
std::string make_run_path(const std::string& output_dir);

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop.store(true, std::memory_order_relaxed); }

void run_scheduler(const Config& cfg) {
    std::signal(SIGINT, on_sigint);

    std::string csv_path = make_run_path(cfg.output_dir);
    std::cout << "Telemetry -> " << csv_path << "\n";
    if (cfg.duration_sec == 0)
        std::cout << "Running until Ctrl+C...\n";
    else
        std::cout << "Running for " << cfg.duration_sec << "s...\n";

    CudaContext cuda_ctx{};
    if (cfg.enable_cuda)
        cuda_ctx = heavy_hash_alloc(cfg.thread_count, cfg.block_size, /*buf_mb=*/64);

    NvmlMonitor monitor(cfg.poll_interval_ms, csv_path);
    monitor.start();

    using Clock = std::chrono::steady_clock;
    auto deadline = (cfg.duration_sec > 0)
                  ? Clock::now() + std::chrono::seconds(cfg.duration_sec)
                  : Clock::time_point::max();

    while (!g_stop && Clock::now() < deadline) {
        std::future<void> cuda_fut, vulkan_fut;

        if (cfg.enable_cuda) {
            cuda_fut = std::async(std::launch::async, [&] {
                heavy_hash_launch(cuda_ctx, 512, cfg.memory_stride);
            });
        }

        if (cfg.enable_vulkan) {
            vulkan_fut = std::async(std::launch::async, [&] {
                run_vulkan_workload(cfg.thread_count, 512, cfg.memory_stride,
                                    "vulkan/sha256.spv", /*buf_mb=*/64);
            });
        }

        if (cfg.enable_cuda   && cuda_fut.valid())   cuda_fut.get();
        if (cfg.enable_vulkan && vulkan_fut.valid()) vulkan_fut.get();
    }

    if (g_stop) std::cout << "\nInterrupted.\n";

    monitor.stop();

    if (cfg.enable_cuda)
        heavy_hash_free(cuda_ctx);

    std::cout << "Run complete. CSV saved to " << csv_path << "\n";
}
