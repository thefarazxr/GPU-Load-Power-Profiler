# GPU Load & Power Profiler

A parallel GPU stress-testing and telemetry tool that runs concurrent CUDA and Vulkan compute workloads while continuously sampling power, temperature, clock speeds, and utilization via NVML. Telemetry is logged to CSV and analyzed with Python/Pandas.

---

## Coursework

This project was developed as part of:

| Course | Title |
|--------|-------|
| **COSC 5351** | Advanced Computer Architecture (Spr'25) |
| **COSC 5360** | Parallel Computing (Spr'25) |

It demonstrates concepts from both courses: GPU microarchitecture (SM scheduling, cache hierarchy, memory bandwidth), parallel workload decomposition (CUDA thread/block/grid model, Vulkan compute dispatch), and hardware telemetry analysis.

---

## System Architecture

```
CLI / Config Input
       │
       ▼
Main Controller (scheduler.cpp)
       │                    │
       ▼                    ▼
CUDA Engine            Vulkan Compute Engine
(heavy_hash.cu)        (sha256.comp / vulkan_runner.cpp)
       │                    │
       └────────┬───────────┘
                ▼
         GPU Hardware
                │
                ▼
      NVML Telemetry Layer  ──►  Logger (CSV)
      (nvml_monitor.cpp)              │
                                      ▼
                              Python Analysis
                              (analyze.py / plots.py)
```

**Key design points:**
- CUDA and Vulkan workloads run in parallel (`std::async`) to maximally stress the GPU
- NVML telemetry runs on a dedicated CPU-pinned thread (`pthread_setaffinity_np`) so polling never contends with compute
- The CUDA heavy-hash kernel chains four operation types per iteration (SHA-256 compression, FMA, integer muladd, XOR-rotate) to exercise different SM execution units simultaneously
- The Vulkan shader runs SHA-256 compression with strided reads from a 64MB device-local input buffer, driving memory bandwidth pressure independently of the CUDA engine

---

## Build

**Requirements:**
- CMake ≥ 3.20
- CUDA Toolkit ≥ 12.x (12.8+ for RTX 5000 / Blackwell)
- Vulkan SDK (includes `glslc` for SPIR-V compilation)
- NVML (ships with CUDA Toolkit)
- GCC/Clang with C++17

```bash
# Ada Lovelace (RTX 4000 series)
cmake -B build -DCUDA_ARCH=89
cmake --build build -j$(nproc)

# Blackwell (RTX 5000 series) — requires CUDA 12.8+
cmake -B build -DCUDA_ARCH=100
cmake --build build -j$(nproc)
```

> The SPIR-V shader (`vulkan/sha256.spv`) is compiled automatically by CMake if `glslc` is on your `PATH`. If not, compile manually:
> ```bash
> glslc -o vulkan/sha256.spv vulkan/sha256.comp
> ```

---

## Usage

```bash
./build/gpu_profiler [options]

  --block-size    N    Threads per CUDA block                    (default: 256)
  --threads       N    Total thread count; grid = N / block-size (default: 1048576)
  --stride        N    Memory access stride in elements          (default: 1)
  --duration      N    Run for N seconds; omit to run until Ctrl+C
  --poll          N    Telemetry poll interval in ms             (default: 200)
  --no-cuda            Disable CUDA engine
  --no-vulkan          Disable Vulkan engine
  --output-dir    DIR  Output directory for CSVs                 (default: data/)
```

By default the profiler runs until **Ctrl+C**. Pass `--duration N` to stop automatically after N seconds.

All three compute parameters (`--block-size`, `--threads`, `--stride`) are runtime flags — no recompile needed between experiments.

---

## Sweepable Parameters

Each parameter maps directly to a hardware behavior that appears in the telemetry CSV.

| Flag | Controls | What changes in telemetry |
|------|----------|--------------------------|
| `--block-size` | Threads per CUDA block → warp scheduling, register/shared-mem pressure | SM utilization, clock stability |
| `--threads` | Total thread count → grid occupancy | SM utilization, power draw |
| `--stride N` | Memory access stride in 4-byte elements | Memory controller utilization, clock variance |

**Stride and cache pressure:**

| `--stride` | Access pattern | Cache behaviour |
|-----------|----------------|-----------------|
| `1` | Sequential | L1/L2 friendly, low bandwidth pressure |
| `32` | One 128-byte cache line per step | L1 misses begin |
| `256+` | Far beyond cache line | L2 thrashing, high bandwidth pressure |

**Example — cache-pressure sweep (no recompile):**
```bash
for stride in 1 32 256 1024; do
    ./build/gpu_profiler --duration 60 --stride $stride
done
```

**Example — occupancy sweep:**
```bash
for bs in 64 128 256 512; do
    ./build/gpu_profiler --duration 60 --block-size $bs
done
```

**Example — 15-minute full saturation run:**
```bash
./build/gpu_profiler --duration 900 --block-size 256 --threads 1048576 --stride 1
```

---

## Live Monitoring

Run `live.py` in a second terminal **while the profiler is running**. It reads the CSV as it's being written and refreshes every 500ms:

```bash
python scripts/live.py                    # auto-picks newest CSV in data/
python scripts/live.py data/run_xyz.csv   # explicit path
python scripts/live.py --no-plot          # terminal table only (no GUI window)
```

Two outputs run simultaneously:

- **Terminal table** — in-place updating stats (current / mean / max) for all 7 metrics, with a live status line that shows `THROTTLING` or `TEMP >= 75°C` as soon as either condition is detected
- **Matplotlib window** — 4-panel live plot (Power, Temp, SM Clock, SM Util) over a rolling 120-second window; SM Clock panel includes a dashed throttle-threshold line; closes cleanly with the window's X button

If matplotlib is unavailable or there is no display, falls back to terminal-only automatically.

---

## Post-run Analysis

```bash
# Summary stats + throttle/temperature detection + SM utilization report
python scripts/analyze.py data/run_*.csv

# Generate PNG charts — two files per run:
#   run_<timestamp>.png          — 7-panel overview (power, temp, clocks, util, VRAM)
#   run_<timestamp>_throttle.png — dedicated throttle chart (SM clock + temp, dual axis,
#                                   throttle onset annotated with timestamp and % drop)
python scripts/plots.py data/run_*.csv
```

**Python dependencies:**
```bash
pip install pandas matplotlib
```

---

## Project Structure

```
gpu-profiler/
├── src/
│   ├── main.cpp              # Entry point
│   ├── scheduler.cpp         # Parallel workload orchestration
│   └── config.cpp            # CLI / parameter parsing
├── cuda/
│   └── heavy_hash.cu         # CUDA heavy-hash kernel (4-phase compute)
├── vulkan/
│   ├── vulkan_runner.cpp     # Vulkan setup, dispatch, and teardown
│   ├── sha256.comp           # GLSL compute shader (SHA-256)
│   └── sha256.spv            # Compiled SPIR-V (generated by CMake)
├── telemetry/
│   ├── nvml_monitor.cpp      # NVML polling on a pinned CPU thread
│   └── logger.cpp            # Timestamped CSV path generation
├── include/
│   ├── config.h
│   ├── nvml_monitor.h
│   └── heavy_hash.h          # CudaContext struct + alloc/launch/free interface
├── scripts/
│   ├── live.py               # Live terminal monitor (reads CSV while profiler runs)
│   ├── analyze.py            # Pandas stats, throttle detection, SM util
│   └── plots.py              # Matplotlib time-series charts
├── data/                     # Telemetry CSV output (run_YYYYMMDD_HHMMSS.csv)
└── CMakeLists.txt
```

---

## Telemetry Columns (CSV)

| Column | Description |
|--------|-------------|
| `timestamp_ms` | Milliseconds since run start |
| `power_mw` | GPU board power draw (milliwatts) |
| `temp_c` | GPU die temperature (°C) |
| `clock_sm_mhz` | SM clock frequency (MHz) |
| `clock_mem_mhz` | Memory clock frequency (MHz) |
| `utilization_gpu` | SM utilization — % of time ≥1 warp active on any SM |
| `utilization_mem` | Memory controller utilization (%) |
| `used_memory_bytes` | VRAM in use (bytes) |

> **Note on SM utilization:** `utilization_gpu` from `nvmlDeviceGetUtilizationRates()` is aggregate SM utilization averaged over the poll window. Per-SM or per-warp occupancy requires CUPTI.
