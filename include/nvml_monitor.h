#pragma once
#include <atomic>
#include <string>
#include <thread>

struct TelemetrySample {
    uint64_t timestamp_ms;
    uint32_t power_mw;
    uint32_t temp_c;
    uint32_t clock_sm_mhz;
    uint32_t clock_mem_mhz;
    uint32_t utilization_gpu;
    uint32_t utilization_mem;
    uint64_t used_memory_bytes;
};

class NvmlMonitor {
public:
    explicit NvmlMonitor(uint32_t poll_interval_ms, const std::string& output_path);
    ~NvmlMonitor();

    void start();
    void stop();

private:
    void poll_loop();
    void pin_to_cpu(int cpu_id);

    uint32_t          poll_interval_ms_;
    std::string       output_path_;
    std::atomic<bool> running_{false};
    std::thread       thread_;
};
