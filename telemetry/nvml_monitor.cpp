#include "nvml_monitor.h"
#include <nvml.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>

NvmlMonitor::NvmlMonitor(uint32_t poll_interval_ms, const std::string& output_path)
    : poll_interval_ms_(poll_interval_ms), output_path_(output_path) {}

NvmlMonitor::~NvmlMonitor() { stop(); }

void NvmlMonitor::start() {
    running_ = true;
    thread_  = std::thread(&NvmlMonitor::poll_loop, this);
}

void NvmlMonitor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void NvmlMonitor::pin_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void NvmlMonitor::poll_loop() {
    pin_to_cpu(0);

    if (nvmlInit_v2() != NVML_SUCCESS)
        throw std::runtime_error("nvmlInit failed");

    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS)
        throw std::runtime_error("nvmlDeviceGetHandleByIndex failed");

    std::ofstream csv(output_path_);
    csv << "timestamp_ms,power_mw,temp_c,clock_sm_mhz,clock_mem_mhz,"
           "utilization_gpu,utilization_mem,used_memory_bytes\n";

    auto t0 = std::chrono::steady_clock::now();

    while (running_) {
        TelemetrySample s{};

        auto now = std::chrono::steady_clock::now();
        s.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();

        nvmlDeviceGetPowerUsage(dev, &s.power_mw);
        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &s.temp_c);
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM,  &s.clock_sm_mhz);
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &s.clock_mem_mhz);

        nvmlUtilization_t util{};
        nvmlDeviceGetUtilizationRates(dev, &util);
        s.utilization_gpu = util.gpu;
        s.utilization_mem = util.memory;

        nvmlMemory_t mem{};
        nvmlDeviceGetMemoryInfo(dev, &mem);
        s.used_memory_bytes = mem.used;

        csv << s.timestamp_ms    << ","
            << s.power_mw        << ","
            << s.temp_c          << ","
            << s.clock_sm_mhz    << ","
            << s.clock_mem_mhz   << ","
            << s.utilization_gpu << ","
            << s.utilization_mem << ","
            << s.used_memory_bytes << "\n";
        csv.flush();

        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms_));
    }

    nvmlShutdown();
}
